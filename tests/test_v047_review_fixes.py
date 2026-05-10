from __future__ import annotations

from pathlib import Path
import shutil
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import torch

from src.backtest.materialize import materialize_particle_pair
from src.model_layout.compiler import compile_parameter_layout
from src.policy.feasible_domain import compile_feasible_domain
from src.policy.policy_regime import build_policy_state_trace
from src.policy_deep.document_reader import read_policy_documents
from src.policy_deep.llm_bridge import resolve_llm_candidates
from src.training.tensor_bundle import compile_training_tensor_bundle


def _config() -> dict:
    return {
        "reward": {
            "baseline_strategy": "dynamic_lock_only",
            "baseline_position_ratios": [0.55],
            "cvar_alpha": 0.99,
            "lambda_tail": 0.65,
            "lambda_hedge": 0.18,
            "lambda_trade": 0.10,
            "lambda_violate": 1.0,
        },
        "score_kernel": {
            "contract_position_base_ratio": 0.50,
            "exposure_band_base_ratio": 0.00,
            "lt_settlement_weight": 0.00,
            "da_settlement_weight": 1.00,
            "hourly_limit": {
                "base_multiplier": 0.00,
                "shrink_multiplier": 0.00,
            },
        },
        "economics": {
            "retail_tariff_yuan_per_mwh": 430.0,
            "imbalance_penalty_multiplier": 1.0,
            "adjustment_cost_yuan_per_mwh": 0.6,
            "friction_cost_yuan_per_mwh": 1.2,
        },
        "policy_projection": {"mode": "policy_only", "clip_method": "projection_only", "violation_penalty_scale": 1.0},
        "policy_regime": {"pre_switch_window_days": 28, "countdown_cap_days": 60},
        "policy_feasible_domain": {
            "enabled": True,
            "strict_mode": True,
            "bind_upper_actions": True,
            "bind_lower_actions": True,
            "bind_settlement_mode": True,
            "non_negative_position_required": True,
            "contract_adjustment_ratio_limit": 0.30,
            "contract_adjustment_ratio_limit_linked": 0.12,
            "exposure_band_ratio_floor": 0.00,
            "exposure_band_ratio_cap": 0.25,
            "exposure_band_ratio_cap_ancillary_tight": 0.12,
            "hourly_hedge_share_cap": 1.0,
            "hourly_hedge_share_cap_ancillary_tight": 0.40,
            "hourly_ramp_share_cap": 1.0,
            "hourly_ramp_share_cap_renewable_active": 0.20,
        },
        "parameter_compiler": {
            "upper": {
                "weekly_feature_source": "agent_feature_columns",
                "policy_feature_source": "policy_state_numeric_columns",
                "blocks": {
                    "weekly_feature_weights": {"source": "weekly_feature_source"},
                    "policy_feature_weights": {
                        "source": "policy_feature_source",
                        "include": ["renewable_mechanism_active", "lt_price_linked_active"],
                    },
                    "contract_curve_latent": {"size": 4},
                    "action_head": {"size": 2},
                },
            },
            "lower": {
                "hourly_feature_groups": {
                    "spread_response": {"columns": ["price_spread"], "response_size": 4},
                    "load_deviation_response": {"columns": ["load_dev"], "response_size": 4},
                    "renewable_response": {"columns": ["renewable_dev"], "response_size": 4},
                    "policy_shrink_response": {"columns": ["ancillary_freq_reserve_tight"], "response_size": 4},
                }
            },
        },
    }


def _bundle() -> dict:
    week = pd.Timestamp("2026-02-02")
    return {
        "weekly_features": pd.DataFrame({"week_start": [week], "feature_a": [0.0]}),
        "weekly_metadata": pd.DataFrame(
            {
                "week_start": [week],
                "week_end": [week + pd.Timedelta(days=6, hours=23, minutes=45)],
                "forecast_weekly_net_demand_mwh": [100.0],
                "actual_weekly_net_demand_mwh": [100.0],
                "lt_price_w": [0.0],
                "hour_count": [2],
            }
        ),
        "policy_state_trace": pd.DataFrame(
            {
                "week_start": [week],
                "renewable_mechanism_active": [0.0],
                "lt_price_linked_active": [0.0],
                "ancillary_freq_reserve_tight": [0.0],
            }
        ),
        "hourly": pd.DataFrame(
            {
                "week_start": [week, week],
                "hour_index": [0, 1],
                "net_load_da": [50.0, 50.0],
                "net_load_id": [50.0, 50.0],
                "price_spread": [0.0, 0.0],
                "load_dev": [0.0, 0.0],
                "renewable_dev": [0.0, 0.0],
            }
        ),
        "quarter": pd.DataFrame(
            {
                "week_start": [week, week],
                "interval_index": [0, 1],
                "全网统一出清价格_日前": [1.0, 1.0],
                "全网统一出清价格_日内": [10.0, 10.0],
                "net_load_da_mwh": [50.0, 50.0],
                "net_load_id_mwh": [10.0, 90.0],
            }
        ),
        "agent_feature_columns": ["feature_a"],
    }


def _compiled_bundle() -> tuple[dict, object]:
    config = _config()
    bundle = _bundle()
    bundle["feasible_domain"] = compile_feasible_domain(
        config=config,
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=bundle["policy_state_trace"],
    )
    layout = compile_parameter_layout(config=config, bundle=bundle)
    bundle["compiled_parameter_layout"] = layout
    bundle["tensor_bundle"] = compile_training_tensor_bundle(bundle, device="cpu")
    return bundle, layout


def test_materialized_settlement_uses_real_15min_actual_need() -> None:
    bundle, layout = _compiled_bundle()
    upper = torch.zeros(layout.upper.total_dimension, dtype=torch.float32)
    lower = torch.zeros(layout.lower.total_dimension, dtype=torch.float32)

    result = materialize_particle_pair(
        tensor_bundle=bundle["tensor_bundle"],
        upper_particle=upper,
        lower_particle=lower,
        strategy_name="test",
        config=_config(),
        compiled_layout=layout,
    )

    assert result.settlement_results["actual_need_15m"].tolist() == [10.0, 90.0]


def test_materialized_weekly_results_expose_required_lock_ratio_fields() -> None:
    bundle, layout = _compiled_bundle()
    upper = torch.zeros(layout.upper.total_dimension, dtype=torch.float32)
    lower = torch.zeros(layout.lower.total_dimension, dtype=torch.float32)

    result = materialize_particle_pair(
        tensor_bundle=bundle["tensor_bundle"],
        upper_particle=upper,
        lower_particle=lower,
        strategy_name="test",
        config=_config(),
        compiled_layout=layout,
    )

    row = result.weekly_results.iloc[0]
    assert row["lock_ratio_base"] == 0.50
    assert row["delta_lock_ratio"] == 0.0
    assert row["lock_ratio_final"] == 0.50


def test_policy_state_trace_keeps_inside_week_rules_forward_until_next_decision() -> None:
    week_start = pd.Timestamp("2025-12-29")
    rule_table = pd.DataFrame(
        {
            "rule_id": ["r1"],
            "policy_name": ["新能源机制"],
            "publish_time": [pd.Timestamp("2025-11-05")],
            "effective_start": [pd.Timestamp("2026-01-01")],
            "effective_end": [pd.NaT],
            "rule_type": ["renewable_mechanism_execution"],
            "scope": ["湖南"],
            "state_group": ["renewable_mechanism"],
            "state_name": ["renewable_mechanism_active"],
            "state_value": [1.0],
            "source_file": ["policy.docx"],
            "note": ["2026-01-01生效"],
        }
    )
    trace = build_policy_state_trace(
        weekly_metadata=pd.DataFrame({"week_start": [week_start], "week_end": [pd.Timestamp("2026-01-04 23:45:00")]}),
        rule_table=rule_table,
        inventory=pd.DataFrame({"parse_status": [], "source_file": []}),
        config=_config(),
    )

    assert float(trace.loc[0, "renewable_mechanism_active"]) == 0.0
    assert float(trace.loc[0, "forward_mechanism_execution_in_window"]) == 1.0
    assert float(trace.loc[0, "forward_mechanism_execution_days"]) == 3.0


def test_feasible_domain_prefers_policy_trace_when_weekly_metadata_already_has_policy_columns() -> None:
    week = pd.Timestamp("2026-02-02")
    weekly_metadata = pd.DataFrame(
        {
            "week_start": [week],
            "hour_count": [24],
            "lt_price_linked_active": [0.0],
            "renewable_mechanism_active": [0.0],
            "ancillary_freq_reserve_tight": [0.0],
        }
    )
    policy_state_trace = pd.DataFrame(
        {
            "week_start": [week],
            "lt_price_linked_active": [1.0],
            "renewable_mechanism_active": [1.0],
            "ancillary_freq_reserve_tight": [1.0],
        }
    )

    domain = compile_feasible_domain(
        config=_config(),
        weekly_metadata=weekly_metadata,
        policy_state_trace=policy_state_trace,
    )

    weekly = domain.weekly_bounds.iloc[0]
    assert weekly["contract_adjustment_ratio_min"] == -0.12
    assert weekly["contract_adjustment_ratio_max"] == 0.12
    assert weekly["exposure_band_ratio_max"] == 0.12
    assert weekly["max_hourly_ramp_share"] == 0.20
    assert weekly["bound_reason_code"] == "lt_price_linked|ancillary_tight|renewable_active"
    assert domain.settlement_semantics.iloc[0]["settlement_mode"] == "linked_40_60"


def _workspace_case_dir(prefix: str) -> Path:
    root = Path(".cache") / "tests" / "pytest_review" / "cases" / f"{prefix}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def test_disabled_llm_candidate_parser_ignores_stale_cache() -> None:
    temp_dir = _workspace_case_dir("llm_cache")
    candidates = pd.DataFrame(
        {
            "candidate_id": ["candidate_0001"],
            "extractor": ["parsed_rule_bootstrap"],
            "confidence": [0.85],
        }
    )
    try:
        cache_dir = temp_dir / "policy_cache"
        cache_dir.mkdir()

        resolved = resolve_llm_candidates(candidates, enabled=False, cache_directory=cache_dir, cache_only=True)

        assert resolved["extractor"].tolist() == ["parsed_rule_bootstrap"]
        assert list(cache_dir.glob("llm_candidates_*.json")) == []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_minimal_docx(path: Path, text: str) -> None:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{escaped}</w:t></w:r></w:p></w:body></w:document>"
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)


def test_policy_document_units_include_source_text_not_only_rule_notes() -> None:
    temp_dir = _workspace_case_dir("policy_docs")
    try:
        policy_dir = temp_dir / "政策环境"
        policy_dir.mkdir()
        _write_minimal_docx(policy_dir / "2025.10.15湖南省电力中长期实施细则.docx", "中长期交易按每天24小时划分为24个时段。")

        result = read_policy_documents(policy_dir, project_root=temp_dir)

        assert "source_text" in set(result.document_units["unit_type"])
        assert result.document_units["text"].str.contains("24小时").any()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
