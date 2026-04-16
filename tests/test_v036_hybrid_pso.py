from __future__ import annotations

import pandas as pd

from src.agents.hybrid_pso import train_hybrid_pso_model
from src.model_layout.compiler import compile_parameter_layout
from src.training.tensor_bundle import compile_training_tensor_bundle


def _bundle() -> dict:
    weeks = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"])
    return {
        "weekly_features": pd.DataFrame(
            {
                "week_start": weeks,
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [0.5, 1.5, 2.5],
                "feature_c": [4.0, 5.0, 6.0],
            }
        ),
        "weekly_metadata": pd.DataFrame(
            {
                "week_start": weeks,
                "forecast_weekly_net_demand_mwh": [1000.0, 1200.0, 1150.0],
                "actual_weekly_net_demand_mwh": [1020.0, 1180.0, 1130.0],
                "da_price_mean": [300.0, 320.0, 315.0],
                "lt_price_w": [290.0, 300.0, 302.0],
            }
        ),
        "policy_state_trace": pd.DataFrame(
            {
                "week_start": weeks,
                "renewable_mechanism_active": [0.0, 1.0, 1.0],
                "lt_price_linked_active": [0.0, 1.0, 1.0],
                "forward_price_linkage_days": [12.0, 8.0, 3.0],
                "forward_mechanism_execution_days": [20.0, 15.0, 9.0],
            }
        ),
        "hourly": pd.DataFrame(
            {
                "week_start": [weeks[0], weeks[0], weeks[1], weeks[1], weeks[2], weeks[2]],
                "hour_index": [0, 1, 0, 1, 0, 1],
                "net_load_da": [100.0, 120.0, 110.0, 130.0, 108.0, 125.0],
                "net_load_id": [98.0, 122.0, 112.0, 129.0, 107.0, 124.0],
                "price_spread": [10.0, 12.0, 9.0, 8.0, 9.5, 7.5],
                "load_dev": [2.0, -2.0, 1.0, -1.0, 0.5, -0.5],
                "renewable_dev": [5.0, 4.0, 6.0, 3.0, 4.5, 2.0],
            }
        ),
        "quarter": pd.DataFrame(
            {
                "week_start": [weeks[0], weeks[0], weeks[1], weeks[1], weeks[2], weeks[2]],
                "interval_index": [0, 1, 0, 1, 0, 1],
                "全网统一出清价格_日前": [300.0, 302.0, 320.0, 321.0, 318.0, 316.0],
                "全网统一出清价格_日内": [310.0, 311.0, 330.0, 331.0, 327.0, 325.0],
            }
        ),
        "agent_feature_columns": ["feature_a", "feature_b", "feature_c"],
    }


def _config() -> dict:
    return {
        "training": {"device": "cpu"},
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
        "hybrid_pso": {
            "seed": 7,
            "upper": {"particles": 4, "iterations": 3, "dimension": 12},
            "lower": {"particles": 3, "iterations": 3, "dimension": 8},
        },
    }


def test_train_hybrid_pso_model_uses_compiled_layout_dimensions() -> None:
    bundle = _bundle()
    layout = compile_parameter_layout(config=_config(), bundle=bundle)
    bundle["compiled_parameter_layout"] = layout
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")

    result = train_hybrid_pso_model(tensor_bundle=tensor_bundle, config=_config(), compiled_layout=layout)

    assert result.model.upper_best.shape == (layout.upper.total_dimension,)
    assert result.model.lower_best.shape == (layout.lower.total_dimension,)
    assert result.runtime_profile["upper_dim"] == layout.upper.total_dimension
    assert result.runtime_profile["lower_dim"] == layout.lower.total_dimension
