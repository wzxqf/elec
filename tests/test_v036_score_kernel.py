from __future__ import annotations

import pandas as pd
import torch

from src.model_layout.compiler import compile_parameter_layout
from src.training.score_kernel import batch_score_particles
from src.training.tensor_bundle import compile_training_tensor_bundle


def _bundle() -> dict:
    weeks = pd.to_datetime(["2026-01-05", "2026-01-12"])
    weekly_features = {"week_start": weeks}
    for index in range(1, 15):
        weekly_features[f"feature_{index:02d}"] = [float(index), float(index + 1)]
    return {
        "weekly_features": pd.DataFrame(weekly_features),
        "weekly_metadata": pd.DataFrame(
            {
                "week_start": weeks,
                "forecast_weekly_net_demand_mwh": [1000.0, 1200.0],
                "actual_weekly_net_demand_mwh": [1020.0, 1180.0],
                "lt_price_w": [290.0, 300.0],
            }
        ),
        "policy_state_trace": pd.DataFrame(
            {
                "week_start": weeks,
                "renewable_mechanism_active": [0.0, 1.0],
                "lt_price_linked_active": [0.0, 1.0],
                "forward_price_linkage_days": [12.0, 3.0],
                "forward_mechanism_execution_days": [20.0, 9.0],
            }
        ),
        "hourly": pd.DataFrame(
            {
                "week_start": [weeks[0], weeks[0], weeks[1], weeks[1]],
                "hour_index": [0, 1, 0, 1],
                "net_load_da": [100.0, 120.0, 110.0, 130.0],
                "net_load_id": [98.0, 122.0, 112.0, 129.0],
                "price_spread": [10.0, 12.0, 9.0, 8.0],
                "load_dev": [2.0, -2.0, 1.0, -1.0],
                "renewable_dev": [5.0, 4.0, 6.0, 3.0],
            }
        ),
        "quarter": pd.DataFrame(
            {
                "week_start": [weeks[0], weeks[0], weeks[1], weeks[1]],
                "interval_index": [0, 1, 0, 1],
                "全网统一出清价格_日前": [300.0, 302.0, 320.0, 321.0],
                "全网统一出清价格_日内": [310.0, 311.0, 330.0, 331.0],
            }
        ),
        "agent_feature_columns": [f"feature_{index:02d}" for index in range(1, 15)],
    }


def _config() -> dict:
    return {
        "reward": {
            "baseline_strategy": "dynamic_lock_only",
            "cvar_alpha": 0.99,
            "lambda_tail": 0.65,
            "lambda_hedge": 0.18,
            "lambda_trade": 0.10,
            "lambda_violate": 1.0,
        },
        "economics": {
            "retail_tariff_yuan_per_mwh": 430.0,
            "imbalance_penalty_multiplier": 1.0,
            "adjustment_cost_yuan_per_mwh": 0.6,
            "friction_cost_yuan_per_mwh": 1.2,
        },
        "policy_projection": {"mode": "policy_only", "clip_method": "projection_only", "violation_penalty_scale": 1.0},
        "parameter_compiler": {
            "upper": {
                "weekly_feature_source": "agent_feature_columns",
                "policy_feature_source": "policy_state_numeric_columns",
                "blocks": {
                    "weekly_feature_weights": {"source": "weekly_feature_source"},
                    "policy_feature_weights": {
                        "source": "policy_feature_source",
                        "include": [
                            "renewable_mechanism_active",
                            "lt_price_linked_active",
                            "forward_price_linkage_days",
                            "forward_mechanism_execution_days",
                        ],
                    },
                    "contract_curve_latent": {"size": 12},
                    "action_head": {"size": 6},
                },
            },
            "lower": {
                "hourly_feature_groups": {
                    "spread_response": {"columns": ["price_spread"], "response_size": 8},
                    "load_deviation_response": {"columns": ["load_dev"], "response_size": 8},
                    "renewable_response": {"columns": ["renewable_dev"], "response_size": 8},
                    "policy_shrink_response": {"columns": ["ancillary_freq_reserve_tight"], "response_size": 8},
                }
            },
        },
    }


def test_batch_score_particles_responds_to_late_declared_upper_features() -> None:
    bundle = _bundle()
    layout = compile_parameter_layout(config=_config(), bundle=bundle)
    bundle["compiled_parameter_layout"] = layout
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper_a = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    upper_b = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    late_index = layout.upper.blocks[0].slice_end - 1
    upper_b[0, late_index] = 5.0
    lower = torch.zeros((1, layout.lower.total_dimension), dtype=torch.float32)

    result_a = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper_a,
        lower_particles=lower,
        device="cpu",
        config=_config(),
        compiled_layout=layout,
    )
    result_b = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper_b,
        lower_particles=lower,
        device="cpu",
        config=_config(),
        compiled_layout=layout,
    )

    assert not torch.allclose(result_a.contract_adjustment_mwh_raw, result_b.contract_adjustment_mwh_raw)


def test_batch_score_particles_builds_curve_from_declared_curve_block() -> None:
    bundle = _bundle()
    layout = compile_parameter_layout(config=_config(), bundle=bundle)
    bundle["compiled_parameter_layout"] = layout
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower = torch.zeros((1, layout.lower.total_dimension), dtype=torch.float32)

    result = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper,
        lower_particles=lower,
        device="cpu",
        config=_config(),
        compiled_layout=layout,
    )

    assert result.contract_curve.shape[-1] == 24
    assert result.contract_curve.shape[:3] == (1, 1, 2)
