from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.hourly_spot_experiments import (
    build_hourly_spot_experiment_grid,
    derive_hourly_spot_baseline,
    summarize_hourly_spot_guardrails,
)


def test_build_hourly_spot_experiment_grid_respects_disabled_flag() -> None:
    assert build_hourly_spot_experiment_grid({"hourly_spot_experiment": {"enabled": False}}) == []


def test_build_hourly_spot_experiment_grid_expands_values() -> None:
    candidates = build_hourly_spot_experiment_grid(
        {
            "hourly_spot_experiment": {
                "enabled": True,
                "signal_transform": ["raw", "forecast_load_normalized"],
                "gate_mode": ["hard"],
                "signal_clip_abs": [0.0],
                "signal_deadband": [0.05, 0.10],
                "temperature": [0.04],
                "hourly_limit_base_multiplier": [0.50],
                "hourly_limit_shrink_multiplier": [0.00, 0.50],
                "friction_cost_yuan_per_mwh": [1.50],
                "lambda_trade": [0.10],
            }
        }
    )
    assert len(candidates) == 8
    assert candidates[0].experiment_id == "hourly_spot_001"
    assert candidates[0].signal_transform == "raw"
    assert candidates[-1].signal_transform == "forecast_load_normalized"
    assert candidates[-1].hourly_limit_shrink_multiplier == 0.50


def test_summarize_hourly_spot_guardrails_marks_passed_candidate() -> None:
    frame = pd.DataFrame(
        [
            {
                "experiment_id": "weak",
                "signal_transform": "raw",
                "gate_mode": "hard",
                "signal_clip_abs": 0.0,
                "sum_excess_profit_w": -600.0,
                "mean_cvar99_w": 100.0,
                "mean_hedge_error_w": 0.60,
                "nonzero_hour_share": 0.20,
                "spot_abs_sum_mwh": 1000.0,
            },
            {
                "experiment_id": "strong",
                "signal_transform": "forecast_load_normalized",
                "gate_mode": "hard",
                "signal_clip_abs": 0.75,
                "sum_excess_profit_w": -400.0,
                "mean_cvar99_w": 102.0,
                "mean_hedge_error_w": 0.50,
                "nonzero_hour_share": 0.30,
                "spot_abs_sum_mwh": 1500.0,
            },
        ]
    )
    result = summarize_hourly_spot_guardrails(
        frame,
        baseline={
            "sum_excess_profit_w": -488.0,
            "mean_cvar99_w": 100.0,
            "mean_hedge_error_w": 0.55,
            "cvar_tolerance": 1.03,
        },
    )
    assert result.iloc[0]["experiment_id"] == "strong"
    assert "baseline_excess_improved" in result.columns
    assert "dynamic_lock_only_improved" not in result.columns
    assert bool(result.iloc[0]["candidate_passed"]) is True
    assert bool(result.iloc[1]["candidate_passed"]) is False


def test_summarize_hourly_spot_guardrails_requires_schema() -> None:
    with pytest.raises(KeyError):
        summarize_hourly_spot_guardrails(pd.DataFrame({"experiment_id": ["x"]}), baseline={})


def test_derive_hourly_spot_baseline_uses_current_config_candidate() -> None:
    frame = pd.DataFrame(
        [
            {
                "experiment_id": "other",
                "signal_transform": "forecast_load_normalized",
                "gate_mode": "soft",
                "signal_clip_abs": 0.75,
                "signal_deadband": 0.05,
                "gate_temperature": 0.08,
                "hourly_limit_base_multiplier": 0.75,
                "hourly_limit_shrink_multiplier": 0.25,
                "friction_cost_yuan_per_mwh": 1.50,
                "lambda_trade": 0.10,
                "sum_excess_profit_w": -100.0,
                "mean_cvar99_w": 4.0,
                "mean_hedge_error_w": 0.3,
            },
            {
                "experiment_id": "current",
                "signal_transform": "raw",
                "gate_mode": "hard",
                "signal_clip_abs": 0.0,
                "signal_deadband": 0.12,
                "gate_temperature": 0.04,
                "hourly_limit_base_multiplier": 0.50,
                "hourly_limit_shrink_multiplier": 0.50,
                "friction_cost_yuan_per_mwh": 1.50,
                "lambda_trade": 0.10,
                "sum_excess_profit_w": -300.0,
                "mean_cvar99_w": 5.0,
                "mean_hedge_error_w": 0.4,
            },
        ]
    )
    baseline = derive_hourly_spot_baseline(
        frame,
        config={
            "score_kernel": {
                "hourly_signal": {"transform": "raw", "signal_clip_abs": 0.0},
                "hourly_gate": {"mode": "hard", "signal_deadband": 0.12, "temperature": 0.04},
                "hourly_limit": {"base_multiplier": 0.50, "shrink_multiplier": 0.50},
            },
            "economics": {"friction_cost_yuan_per_mwh": 1.50},
            "reward": {"lambda_trade": 0.10},
        },
        cvar_tolerance=1.03,
    )

    assert baseline["sum_excess_profit_w"] == -300.0
    assert baseline["mean_cvar99_w"] == 5.0
    assert baseline["mean_hedge_error_w"] == 0.4
