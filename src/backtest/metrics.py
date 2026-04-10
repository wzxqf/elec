from __future__ import annotations

import numpy as np
import pandas as pd


def cvar(series: pd.Series, alpha: float = 0.95) -> float:
    if series.empty:
        return 0.0
    threshold = float(series.quantile(alpha))
    tail = series.loc[series >= threshold]
    if tail.empty:
        return threshold
    return float(tail.mean())


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cumulative = series.cumsum()
    running_min = cumulative.cummin()
    return float((cumulative - running_min).max())


def summarize_strategy_results(
    weekly_results: pd.DataFrame,
    settlement_results: pd.DataFrame,
    strategy_name: str,
    cvar_alpha: float = 0.95,
) -> dict[str, float | str]:
    weekly_cost = weekly_results["procurement_cost_w"]
    return {
        "strategy": strategy_name,
        "total_procurement_cost": float(weekly_cost.sum()),
        "avg_weekly_cost": float(weekly_cost.mean()),
        "weekly_cost_volatility": float(weekly_cost.std(ddof=0)),
        "cvar": cvar(weekly_cost, alpha=cvar_alpha),
        "hedge_error": float(weekly_results["hedge_error_w"].mean()),
        "mean_reward": float(weekly_results["reward"].mean()),
        "avg_adjustment_mwh": float(weekly_results["avg_adjustment_mwh"].mean()),
        "total_trans_cost": float(weekly_results["trans_cost_w"].sum()),
        "max_drawdown": max_drawdown(weekly_cost),
        "settlement_records": int(len(settlement_results)),
    }
