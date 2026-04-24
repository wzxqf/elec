from __future__ import annotations

import pandas as pd
import torch

from src.model_layout.compiler import compile_parameter_layout
from src.policy.feasible_domain import compile_feasible_domain
from src.training.score_kernel import batch_score_particles
from src.training.tensor_bundle import compile_training_tensor_bundle


def _bundle() -> dict:
    week = pd.Timestamp("2026-02-02")
    return {
        "weekly_features": pd.DataFrame(
            {
                "week_start": [week],
                "feature_a": [2.0],
                "feature_b": [1.0],
            }
        ),
        "weekly_metadata": pd.DataFrame(
            {
                "week_start": [week],
                "forecast_weekly_net_demand_mwh": [1000.0],
                "actual_weekly_net_demand_mwh": [980.0],
                "lt_price_w": [300.0],
                "hour_count": [2],
            }
        ),
        "policy_state_trace": pd.DataFrame(
            {
                "week_start": [week],
                "renewable_mechanism_active": [1.0],
                "lt_price_linked_active": [1.0],
                "forward_price_linkage_days": [2.0],
                "forward_mechanism_execution_days": [2.0],
                "ancillary_freq_reserve_tight": [1.0],
            }
        ),
        "hourly": pd.DataFrame(
            {
                "week_start": [week, week],
                "hour_index": [0, 1],
                "net_load_da": [100.0, 120.0],
                "net_load_id": [98.0, 121.0],
                "price_spread": [20.0, 20.0],
                "load_dev": [5.0, -5.0],
                "renewable_dev": [8.0, 8.0],
            }
        ),
        "quarter": pd.DataFrame(
            {
                "week_start": [week, week],
                "interval_index": [0, 1],
                "全网统一出清价格_日前": [320.0, 321.0],
                "全网统一出清价格_日内": [330.0, 331.0],
            }
        ),
        "agent_feature_columns": ["feature_a", "feature_b"],
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
        "policy_feasible_domain": {
            "enabled": True,
            "strict_mode": True,
            "bind_upper_actions": True,
            "bind_lower_actions": True,
            "bind_settlement_mode": True,
            "non_negative_position_required": True,
            "contract_adjustment_ratio_limit": 0.30,
            "contract_adjustment_ratio_limit_linked": 0.10,
            "exposure_band_ratio_floor": 0.05,
            "exposure_band_ratio_cap": 0.25,
            "exposure_band_ratio_cap_ancillary_tight": 0.12,
            "hourly_hedge_share_cap": 1.0,
            "hourly_hedge_share_cap_ancillary_tight": 0.40,
            "hourly_ramp_share_cap": 1.0,
            "hourly_ramp_share_cap_renewable_active": 0.15,
        },
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


def test_batch_score_particles_projects_actions_into_feasible_domain() -> None:
    bundle = _bundle()
    bundle["feasible_domain"] = compile_feasible_domain(
        config=_config(),
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=bundle["policy_state_trace"],
    )
    layout = compile_parameter_layout(config=_config(), bundle=bundle)
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower = torch.full((1, layout.lower.total_dimension), 8.0, dtype=torch.float32)
    action_head = next(block for block in layout.upper.blocks if block.name == "action_head")
    upper[0, action_head.slice_start] = 5.0
    upper[0, action_head.slice_start + 1] = 5.0

    result = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper,
        lower_particles=lower,
        device="cpu",
        config=_config(),
        compiled_layout=layout,
    )

    assert result.contract_adjustment_mwh_raw[0, 0, 0] > result.contract_adjustment_mwh_exec[0, 0, 0]
    assert result.exposure_band_mwh_raw[0, 0, 0] > result.exposure_band_mwh[0, 0, 0]
    assert result.contract_adjustment_mwh_exec[0, 0, 0] == 100.0
    assert result.exposure_band_mwh[0, 0, 0] == 120.0
    assert result.feasible_domain_clip_active[0, 0, 0] == 1.0
    assert result.feasible_domain_clip_gap[0, 0, 0] > 0.0
    assert result.weekly_policy_violation_penalty[0, 0, 0] > 0.0


def test_batch_score_particles_consumes_score_kernel_coefficients_from_config() -> None:
    bundle = _bundle()
    config = _config()
    config["score_kernel"] = {
        "contract_position_base_ratio": 0.50,
        "exposure_band_base_ratio": 0.10,
        "lt_settlement_weight": 0.70,
        "da_settlement_weight": 0.30,
    }
    bundle["feasible_domain"] = compile_feasible_domain(
        config=config,
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=bundle["policy_state_trace"],
    )
    layout = compile_parameter_layout(config=config, bundle=bundle)
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower = torch.zeros((1, layout.lower.total_dimension), dtype=torch.float32)

    result = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper,
        lower_particles=lower,
        device="cpu",
        config=config,
        compiled_layout=layout,
    )

    assert result.contract_position_mwh[0, 0, 0] == 500.0
    assert result.exposure_band_mwh_raw[0, 0, 0] == 100.0
    assert result.exposure_band_mwh[0, 0, 0] == 100.0


def test_signed_spot_hedge_reduces_scheduled_energy_when_negative() -> None:
    bundle = _bundle()
    config = _config()
    bundle["weekly_metadata"]["actual_weekly_net_demand_mwh"] = [500.0]
    config["score_kernel"] = {
        "contract_position_base_ratio": 0.60,
        "exposure_band_base_ratio": 0.25,
        "hourly_signal": {
            "spread_weight": 0.20,
            "load_dev_weight": 0.00,
            "renewable_weight": 0.00,
            "spread_abs_weight": 0.00,
            "renewable_abs_weight": 0.00,
        },
        "hourly_limit": {
            "base_multiplier": 1.00,
            "shrink_multiplier": 0.00,
        },
    }
    bundle["feasible_domain"] = compile_feasible_domain(
        config=config,
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=bundle["policy_state_trace"],
    )
    layout = compile_parameter_layout(config=config, bundle=bundle)
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower_positive = torch.full((1, layout.lower.total_dimension), 5.0, dtype=torch.float32)
    lower_negative = torch.full((1, layout.lower.total_dimension), -5.0, dtype=torch.float32)

    positive = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper,
        lower_particles=lower_positive,
        device="cpu",
        config=config,
        compiled_layout=layout,
    )
    negative = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper,
        lower_particles=lower_negative,
        device="cpu",
        config=config,
        compiled_layout=layout,
    )

    assert positive.spot_hedge_mwh[0, 0, 0].sum() > 0.0
    assert negative.spot_hedge_mwh[0, 0, 0].sum() < 0.0
    assert negative.weekly_hedge_error[0, 0, 0] < positive.weekly_hedge_error[0, 0, 0]
    assert negative.weekly_procurement_cost[0, 0, 0] < positive.weekly_procurement_cost[0, 0, 0]


def test_hourly_no_trade_gate_suppresses_small_signals() -> None:
    bundle = _bundle()
    config = _config()
    config["score_kernel"] = {
        "contract_position_base_ratio": 0.60,
        "exposure_band_base_ratio": 0.20,
        "hourly_signal": {
            "spread_weight": 0.001,
            "load_dev_weight": 0.000,
            "renewable_weight": 0.000,
            "spread_abs_weight": 0.000,
            "renewable_abs_weight": 0.000,
        },
        "hourly_limit": {
            "base_multiplier": 1.00,
            "shrink_multiplier": 0.00,
        },
        "hourly_gate": {
            "enabled": True,
            "signal_deadband": 0.10,
            "temperature": 0.02,
        },
    }
    bundle["feasible_domain"] = compile_feasible_domain(
        config=config,
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=bundle["policy_state_trace"],
    )
    layout = compile_parameter_layout(config=config, bundle=bundle)
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower = torch.ones((1, layout.lower.total_dimension), dtype=torch.float32)

    result = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper,
        lower_particles=lower,
        device="cpu",
        config=config,
        compiled_layout=layout,
    )

    assert torch.count_nonzero(result.spot_hedge_mwh.abs() > 1.0e-6).item() == 0


def test_reward_uses_best_profit_from_baseline_position_family() -> None:
    bundle = _bundle()
    config = _config()
    config["reward"]["baseline_position_ratios"] = [0.50, 0.55, 0.60]
    config["score_kernel"] = {
        "contract_position_base_ratio": 0.60,
        "exposure_band_base_ratio": 0.00,
        "baseline_projection_penalty_scale": 0.00,
        "lt_settlement_weight": 0.60,
        "da_settlement_weight": 0.40,
        "hourly_limit": {
            "base_multiplier": 0.00,
            "shrink_multiplier": 0.00,
        },
    }
    bundle["feasible_domain"] = compile_feasible_domain(
        config=config,
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=bundle["policy_state_trace"],
    )
    layout = compile_parameter_layout(config=config, bundle=bundle)
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower = torch.zeros((1, layout.lower.total_dimension), dtype=torch.float32)

    result = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper,
        lower_particles=lower,
        device="cpu",
        config=config,
        compiled_layout=layout,
    )

    assert result.weekly_profit_baseline[0, 0, 0] >= result.weekly_retail_revenue[0, 0, 0] - result.weekly_procurement_cost[0, 0, 0]

