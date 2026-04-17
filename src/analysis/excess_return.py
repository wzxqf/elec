from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cumulative = series.cumsum()
    running_peak = cumulative.cummax()
    return float((running_peak - cumulative).max())


def _lower_tail_mean(series: pd.Series, quantile: float) -> float:
    if series.empty:
        return 0.0
    threshold = float(series.quantile(quantile))
    tail = series.loc[series <= threshold]
    if tail.empty:
        return float(series.min())
    return float(tail.mean())


def build_policy_risk_adjusted_metrics(weekly_results: pd.DataFrame, epsilon: float = 1.0e-6) -> pd.DataFrame:
    frame = weekly_results.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame = frame.sort_values("week_start").reset_index(drop=True)
    active_columns = [column for column in frame.columns if column.endswith("_active")]
    if active_columns:
        switch_pressure = frame[active_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).diff().abs().sum(axis=1).fillna(0.0)
    else:
        switch_pressure = pd.Series(0.0, index=frame.index, dtype="float64")
    countdown_columns = [column for column in frame.columns if column.startswith("forward_") and column.endswith("_days")]
    countdown_pressure = pd.Series(0.0, index=frame.index, dtype="float64")
    for column in countdown_columns:
        countdown_pressure += 1.0 / (_numeric(frame, column).clip(lower=0.0) + 1.0)
    projection_pressure = _numeric(frame, "policy_projection_active")
    violation_pressure = _numeric(frame, "policy_violation_penalty_w")
    penalty = switch_pressure + 0.2 * countdown_pressure + projection_pressure + violation_pressure
    adjusted_excess = _numeric(frame, "excess_profit_w") - penalty
    frame["policy_risk_switch_pressure_w"] = switch_pressure
    frame["policy_risk_countdown_pressure_w"] = countdown_pressure
    frame["policy_risk_penalty_w"] = penalty
    frame["policy_risk_adjusted_excess_return_w"] = adjusted_excess
    frame["active_excess_return_positive"] = adjusted_excess > 0.0
    frame["policy_risk_adjusted_conclusion"] = np.where(
        adjusted_excess > 0.0,
        "政策风险调整后仍为正超额收益",
        "政策风险调整后未形成正超额收益",
    )
    frame["sharpe_epsilon"] = float(epsilon)
    return frame


def summarize_rolling_excess_return(policy_metrics: pd.DataFrame, epsilon: float = 1.0e-6) -> pd.DataFrame:
    frame = policy_metrics.copy()
    if "window_name" not in frame.columns:
        frame["window_name"] = "overall"
    rows: list[dict[str, object]] = []
    for window_name, group in frame.groupby("window_name", sort=True):
        adjusted = _numeric(group, "policy_risk_adjusted_excess_return_w")
        excess = _numeric(group, "excess_profit_w")
        sample_count = int(len(adjusted))
        adjusted_std = float(adjusted.std(ddof=0))
        adjusted_mean = float(adjusted.mean())
        adjusted_median = float(adjusted.median()) if not adjusted.empty else 0.0
        adjusted_p25 = float(adjusted.quantile(0.25)) if not adjusted.empty else 0.0
        adjusted_p75 = float(adjusted.quantile(0.75)) if not adjusted.empty else 0.0
        win_rate = float((excess > 0.0).mean()) if not excess.empty else 0.0
        stable_window = adjusted_std <= float(epsilon)
        insufficient_samples = sample_count < 3
        guard_triggered = stable_window or insufficient_samples
        if insufficient_samples:
            sharpe_warning = "insufficient_samples"
        elif stable_window:
            sharpe_warning = "flat_volatility"
        else:
            sharpe_warning = "ok"
        sharpe = 0.0 if guard_triggered else adjusted_mean / adjusted_std
        active_positive = adjusted_mean > 0.0
        active_persistent = active_positive and win_rate >= 0.5 and (stable_window or sharpe > 0.0)
        rows.append(
            {
                "window_name": window_name,
                "window_sample_count": sample_count,
                "window_excess_return_mean": float(excess.mean()) if not excess.empty else 0.0,
                "window_excess_return_volatility": float(excess.std(ddof=0)) if len(excess) > 1 else 0.0,
                "window_excess_return_win_rate": win_rate,
                "window_excess_return_max_drawdown": _max_drawdown(excess),
                "window_policy_risk_adjusted_mean": adjusted_mean,
                "window_policy_risk_adjusted_median": adjusted_median,
                "window_policy_risk_adjusted_p25": adjusted_p25,
                "window_policy_risk_adjusted_p75": adjusted_p75,
                "window_policy_risk_adjusted_volatility": adjusted_std,
                "window_policy_risk_adjusted_cvar95": _lower_tail_mean(adjusted, 0.05),
                "window_policy_risk_adjusted_cvar99": _lower_tail_mean(adjusted, 0.01),
                "window_policy_risk_adjusted_sharpe": sharpe,
                "window_sharpe_guard_triggered": guard_triggered,
                "window_sharpe_warning": sharpe_warning,
                "window_metrics_tier": "appendix_only" if guard_triggered else "main_text_safe",
                "active_excess_return_positive": active_positive,
                "active_excess_return_persistent": active_persistent,
                "dynamic_lock_only_outperformed": float(excess.mean()) > 0.0,
                "excess_return_conclusion": "持续跑赢 dynamic_lock_only" if active_persistent else "未持续跑赢 dynamic_lock_only",
            }
        )
    return pd.DataFrame(rows)
