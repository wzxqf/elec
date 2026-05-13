from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.hourly_spot_experiments import build_hourly_spot_experiment_grid, summarize_hourly_spot_guardrails


def test_build_hourly_spot_experiment_grid_respects_disabled_flag() -> None:
    assert build_hourly_spot_experiment_grid({"hourly_spot_experiment": {"enabled": False}}) == []


def test_build_hourly_spot_experiment_grid_expands_values() -> None:
    candidates = build_hourly_spot_experiment_grid(
        {
            "hourly_spot_experiment": {
                "enabled": True,
                "signal_deadband": [0.05, 0.10],
                "temperature": [0.04],
                "hourly_limit_base_multiplier": [0.50],
                "hourly_limit_shrink_multiplier": [0.00, 0.50],
                "friction_cost_yuan_per_mwh": [1.20],
                "lambda_trade": [0.10],
            }
        }
    )
    assert len(candidates) == 4
    assert candidates[0].experiment_id == "hourly_spot_001"
    assert candidates[-1].hourly_limit_shrink_multiplier == 0.50


def test_summarize_hourly_spot_guardrails_marks_passed_candidate() -> None:
    frame = pd.DataFrame(
        [
            {
                "experiment_id": "weak",
                "sum_excess_profit_w": -600.0,
                "mean_cvar99_w": 100.0,
                "mean_hedge_error_w": 0.60,
                "nonzero_hour_share": 0.20,
                "spot_abs_sum_mwh": 1000.0,
            },
            {
                "experiment_id": "strong",
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
    assert bool(result.iloc[0]["candidate_passed"]) is True
    assert bool(result.iloc[1]["candidate_passed"]) is False


def test_summarize_hourly_spot_guardrails_requires_schema() -> None:
    with pytest.raises(KeyError):
        summarize_hourly_spot_guardrails(pd.DataFrame({"experiment_id": ["x"]}), baseline={})
