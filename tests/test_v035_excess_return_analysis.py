from __future__ import annotations

import pandas as pd

from src.analysis.excess_return import build_policy_risk_adjusted_metrics, summarize_rolling_excess_return


def test_build_policy_risk_adjusted_metrics_emits_required_columns() -> None:
    weekly = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2026-01-05", "2026-01-12"]),
            "strategy": ["validation", "validation"],
            "excess_profit_w": [10.0, -2.0],
            "policy_projection_active": [1.0, 0.0],
            "policy_violation_penalty_w": [3.0, 1.0],
            "renewable_mechanism_active": [0.0, 1.0],
            "lt_price_linked_active": [0.0, 1.0],
            "forward_price_linkage_days": [5.0, 2.0],
            "forward_mechanism_execution_days": [10.0, 3.0],
        }
    )

    result = build_policy_risk_adjusted_metrics(weekly, epsilon=1.0e-6)

    assert "policy_risk_penalty_w" in result.columns
    assert "policy_risk_adjusted_excess_return_w" in result.columns


def test_summarize_rolling_excess_return_emits_conclusions() -> None:
    policy_metrics = pd.DataFrame(
        {
            "window_name": ["window_01", "window_01", "window_02", "window_02"],
            "policy_risk_adjusted_excess_return_w": [2.0, 1.0, -1.0, 3.0],
            "excess_profit_w": [3.0, 2.0, -1.0, 4.0],
        }
    )

    summary = summarize_rolling_excess_return(policy_metrics, epsilon=1.0e-6)

    assert "window_policy_risk_adjusted_sharpe" in summary.columns
    assert "dynamic_lock_only_outperformed" in summary.columns


def test_summarize_rolling_excess_return_clamps_flat_volatility_sharpe() -> None:
    policy_metrics = pd.DataFrame(
        {
            "window_name": ["window_01", "window_01"],
            "policy_risk_adjusted_excess_return_w": [2.0, 2.0],
            "excess_profit_w": [3.0, 3.0],
        }
    )

    summary = summarize_rolling_excess_return(policy_metrics, epsilon=1.0e-6)

    assert summary.loc[0, "window_policy_risk_adjusted_sharpe"] == 0.0
    assert bool(summary.loc[0, "active_excess_return_persistent"]) is True
