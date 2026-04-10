from __future__ import annotations

import numpy as np
import pandas as pd


def cvar_95(series: pd.Series) -> float:
    threshold = float(series.quantile(0.95))
    tail = series[series >= threshold]
    if tail.empty:
        return threshold
    return float(tail.mean())


def max_drawdown(cost_series: pd.Series) -> float:
    equity = -cost_series.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    return float(drawdown.min())


def summarize_strategy_results(
    monthly_results: pd.DataFrame,
    interval_results: pd.DataFrame,
) -> dict[str, float]:
    cumulative_cost = float(monthly_results["procurement_cost_m"].sum())
    cost_volatility = float(monthly_results["procurement_cost_m"].std(ddof=0))
    hedge_error = float(monthly_results["hedge_error_m"].mean())
    avg_adjustment = float(monthly_results["avg_adjustment_mwh"].mean())
    interval_total = interval_results["total_cost_with_penalties"]
    return {
        "cumulative_procurement_cost": cumulative_cost,
        "cost_volatility": cost_volatility,
        "cvar95": cvar_95(interval_total),
        "hedge_error": hedge_error,
        "avg_adjustment_mwh": avg_adjustment,
        "max_drawdown": max_drawdown(monthly_results["procurement_cost_m"]),
        "mean_reward": float(monthly_results["reward"].mean()),
    }
