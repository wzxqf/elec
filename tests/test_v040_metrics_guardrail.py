from __future__ import annotations

import pandas as pd

from src.analysis.excess_return import summarize_rolling_excess_return


def test_summarize_rolling_excess_return_emits_guardrail_statistics() -> None:
    policy_metrics = pd.DataFrame(
        {
            "window_name": ["window_01", "window_01"],
            "policy_risk_adjusted_excess_return_w": [2.0, 2.0],
            "excess_profit_w": [3.0, 3.0],
        }
    )

    summary = summarize_rolling_excess_return(policy_metrics, epsilon=1.0e-6)

    assert "window_policy_risk_adjusted_median" in summary.columns
    assert "window_policy_risk_adjusted_cvar95" in summary.columns
    assert "window_policy_risk_adjusted_cvar99" in summary.columns
    assert "window_sharpe_guard_triggered" in summary.columns
    assert "window_metrics_tier" in summary.columns
    assert bool(summary.loc[0, "window_sharpe_guard_triggered"]) is True
    assert summary.loc[0, "window_metrics_tier"] == "appendix_only"

