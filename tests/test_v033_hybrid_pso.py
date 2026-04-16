from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents.hybrid_pso import (
    HybridPSOModel,
    load_hybrid_pso_model,
    save_hybrid_pso_model,
    train_hybrid_pso_model,
)
from src.training.tensor_bundle import compile_training_tensor_bundle


def _bundle() -> dict:
    weeks = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"])
    return {
        "weekly_features": pd.DataFrame(
            {
                "week_start": weeks,
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [0.5, 1.5, 2.5],
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
                "net_load_id_mwh": [25.0, 26.0, 31.0, 30.0, 28.0, 29.0],
                "全网统一出清价格_日前": [300.0, 302.0, 320.0, 321.0, 318.0, 316.0],
                "全网统一出清价格_日内": [310.0, 311.0, 330.0, 331.0, 327.0, 325.0],
            }
        ),
        "agent_feature_columns": ["feature_a", "feature_b"],
    }


def test_train_hybrid_pso_model_returns_upper_and_lower_best_particles() -> None:
    tensor_bundle = compile_training_tensor_bundle(_bundle(), device="cpu")
    config = {
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
        "upper_strategy": {"parameter_layout": {"contract_adjustment_weights": 3, "exposure_band_weights": 3, "contract_curve_weights": 4, "policy_gate_weights": 2}},
        "lower_strategy": {"parameter_layout": {"spread_response": 2, "load_deviation_response": 2, "renewable_response": 2, "policy_shrink_response": 2}},
        "hybrid_pso": {
            "seed": 7,
            "upper": {"particles": 4, "iterations": 3, "dimension": 12},
            "lower": {"particles": 3, "iterations": 3, "dimension": 8},
        },
    }

    result = train_hybrid_pso_model(tensor_bundle=tensor_bundle, config=config)

    assert result.model.upper_best.shape == (12,)
    assert result.model.lower_best.shape == (8,)
    assert result.runtime_profile["score_kernel_device"] == "cpu"
    assert result.training_trace.shape[0] == 3


def test_hybrid_pso_model_roundtrip() -> None:
    path = Path(".cache") / "tests" / "hybrid_pso_model.json"
    model = HybridPSOModel(
        upper_best=[0.1] * 6,
        lower_best=[0.2] * 5,
        best_score=123.4,
        metadata={"version": "v0.33"},
    )

    save_hybrid_pso_model(model, path)
    loaded = load_hybrid_pso_model(path)

    assert loaded.best_score == 123.4
    assert loaded.metadata["version"] == "v0.33"
    assert loaded.upper_best == [0.1] * 6
