from __future__ import annotations

import pandas as pd
import torch

from src.backtest.materialize import materialize_particle_pair
from src.model_layout.compiler import compile_parameter_layout
from src.training.score_kernel import batch_score_particles
from src.training.tensor_bundle import compile_training_tensor_bundle


def _bundle() -> dict:
    week = pd.Timestamp("2026-01-05")
    return {
        "weekly_features": pd.DataFrame({"week_start": [week], "feature_a": [1.0], "feature_b": [0.5]}),
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
                "ancillary_freq_reserve_tight": [0.0],
            }
        ),
        "hourly": pd.DataFrame(
            {
                "week_start": [week, week],
                "hour_index": [0, 1],
                "net_load_da": [100.0, 120.0],
                "net_load_id": [101.0, 119.0],
                "price_spread": [8.0, 9.0],
                "load_dev": [1.0, -1.0],
                "renewable_dev": [2.0, 3.0],
            }
        ),
        "quarter": pd.DataFrame(
            {
                "week_start": [week, week],
                "interval_index": [0, 1],
                "全网统一出清价格_日前": [320.0, 322.0],
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
        "parameter_compiler": {
            "upper": {
                "weekly_feature_source": "agent_feature_columns",
                "policy_feature_source": "policy_state_numeric_columns",
                "blocks": {
                    "weekly_feature_weights": {"source": "weekly_feature_source"},
                    "policy_feature_weights": {"source": "policy_feature_source", "include": ["renewable_mechanism_active", "lt_price_linked_active", "forward_price_linkage_days", "forward_mechanism_execution_days"]},
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


def test_materialize_particle_pair_matches_score_kernel_aggregate() -> None:
    bundle = _bundle()
    layout = compile_parameter_layout(config=_config(), bundle=bundle)
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower = torch.zeros((1, layout.lower.total_dimension), dtype=torch.float32)

    scored = batch_score_particles(tensor_bundle, upper, lower, device="cpu", config=_config(), compiled_layout=layout)
    materialized = materialize_particle_pair(
        tensor_bundle=tensor_bundle,
        upper_particle=upper[0],
        lower_particle=lower[0],
        strategy_name="consistency_check",
        config=_config(),
        compiled_layout=layout,
    )

    assert materialized.metrics["total_procurement_cost"] == float(scored.procurement_cost[0, 0])
    assert materialized.metrics["total_profit"] == float(scored.profit[0, 0])

