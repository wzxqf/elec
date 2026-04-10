from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.metrics import summarize_strategy_results
from src.backtest.settlement import settle_week
from src.rules.hourly_hedge import apply_hourly_hedge_rule


def _get_week_frames(bundle: dict[str, Any], week_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    week_start = pd.Timestamp(week_start)
    quarter = bundle["quarter"].loc[bundle["quarter"]["week_start"] == week_start].copy()
    hourly = bundle["hourly"].loc[bundle["hourly"]["week_start"] == week_start].copy()
    metadata = bundle["weekly_metadata"].set_index("week_start").loc[week_start]
    return quarter, hourly, metadata


def _allocate_weekly_lt_to_hourly(hourly: pd.DataFrame, q_lt_target: float) -> pd.Series:
    weights = hourly["net_load_da"].clip(lower=0.0)
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(np.ones(len(hourly)), index=hourly.index, dtype=float)
    allocation = q_lt_target * weights / float(weights.sum())
    return allocation.astype(float)


def _week_risk_term(settlement: pd.DataFrame, config: dict[str, Any]) -> tuple[float, float]:
    per_interval = settlement["procurement_cost_15m"]
    sigma = float(per_interval.std(ddof=0))
    alpha = float(config["cost"]["cvar_alpha"])
    threshold = float(per_interval.quantile(alpha))
    tail = per_interval.loc[per_interval >= threshold]
    cvar = float(tail.mean()) if not tail.empty else threshold
    risk_term = (
        float(config["cost"]["risk_vol_weight"]) * sigma
        + float(config["cost"]["risk_cvar_weight"]) * cvar
    )
    return risk_term, cvar


def simulate_week(
    bundle: dict[str, Any],
    week_start: pd.Timestamp,
    action: tuple[float, float],
    config: dict[str, Any],
    previous_lock_ratio: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    quarter, hourly, metadata = _get_week_frames(bundle, week_start)
    lock_ratio_raw = float(action[0])
    hedge_intensity = float(np.clip(action[1], 0.0, 1.0))
    constraints = config["constraints"]

    lock_ratio = float(
        np.clip(lock_ratio_raw, float(constraints["lock_ratio_min"]), float(constraints["lock_ratio_max"]))
    )
    max_step = float(constraints["delta_h_max"])
    lock_ratio = float(np.clip(lock_ratio, previous_lock_ratio - max_step, previous_lock_ratio + max_step))
    lock_ratio = float(
        np.clip(lock_ratio, float(constraints["lock_ratio_min"]), float(constraints["lock_ratio_max"]))
    )

    forecast_weekly_net_demand = float(metadata["forecast_weekly_net_demand_mwh"])
    q_lt_target = max(lock_ratio * forecast_weekly_net_demand, 0.0)
    q_lt_hourly = _allocate_weekly_lt_to_hourly(hourly, q_lt_target)
    hourly_trace, rule_stats = apply_hourly_hedge_rule(
        hourly,
        q_lt_hourly=q_lt_hourly,
        hedge_intensity=hedge_intensity,
        rules_config=config["rules"],
    )

    lt_price_w = float(metadata["lt_price_w"]) if pd.notna(metadata["lt_price_w"]) else float(metadata["da_price_mean"])
    settlement = settle_week(quarter, hourly_trace, lt_price_w=lt_price_w, config=config)
    procurement_cost = float(settlement["procurement_cost_15m"].sum())
    risk_term, cvar_value = _week_risk_term(settlement, config)
    trans_cost = float(
        config["cost"]["transaction_linear"] * hourly_trace["delta_q"].abs().sum()
        + config["cost"]["transaction_quadratic"] * (hourly_trace["delta_q"] ** 2).sum()
        + config["cost"]["lt_adjust_cost"] * abs(q_lt_target - previous_lock_ratio * forecast_weekly_net_demand)
    )
    hedge_error = float(hourly_trace["hedge_error_abs"].mean())

    budget_gap = max(cvar_value - float(constraints["cvar_budget"]), 0.0)
    cvar_penalty = budget_gap * float(constraints["cvar_budget_penalty"])
    reward_raw = -(
        procurement_cost
        + float(config["cost"]["lambda_risk"]) * risk_term
        + float(config["cost"]["lambda_tc"]) * trans_cost
        + float(config["cost"]["lambda_he"]) * hedge_error
        + cvar_penalty
    ) / float(config["env"]["reward_scale"])
    reward = float(np.clip(reward_raw, -20.0, 20.0))

    week_summary = {
        "week_start": pd.Timestamp(week_start),
        "is_partial_week": bool(metadata["is_partial_week"]),
        "lock_ratio": lock_ratio,
        "hedge_intensity": hedge_intensity,
        "q_lt_target_w": q_lt_target,
        "forecast_weekly_net_demand_mwh": forecast_weekly_net_demand,
        "actual_weekly_net_demand_mwh": float(metadata["actual_weekly_net_demand_mwh"]),
        "lt_price_w": lt_price_w,
        "lt_price_source": metadata["lt_price_source"],
        "procurement_cost_w": procurement_cost,
        "risk_term_w": risk_term,
        "trans_cost_w": trans_cost,
        "hedge_error_w": hedge_error,
        "cvar_w": cvar_value,
        "reward_raw": reward_raw,
        "reward": reward,
        "avg_adjustment_mwh": float(hourly_trace["delta_q"].mean()),
        "bound_clip_count": int(rule_stats["bound_clip_count"]),
        "smooth_clip_count": int(rule_stats["smooth_clip_count"]),
        "non_negative_clip_count": int(rule_stats["non_negative_clip_count"]),
        "cvar_budget_excess": budget_gap,
    }

    hourly_trace["week_start"] = pd.Timestamp(week_start)
    settlement["week_start"] = pd.Timestamp(week_start)
    return week_summary, hourly_trace, settlement


def simulate_strategy(
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    action_source: dict[pd.Timestamp, tuple[float, float]] | Callable[[pd.Timestamp], tuple[float, float]],
    config: dict[str, Any],
    strategy_name: str,
    market_vol_scale: float = 1.0,
    price_cap_multiplier: float = 1.0,
    forecast_error_scale: float = 1.0,
) -> dict[str, Any]:
    weekly_records = []
    hourly_records = []
    settlement_records = []
    previous_lock_ratio = 0.0

    bundle_variant = bundle.copy()
    bundle_variant["hourly"] = bundle["hourly"].copy()
    bundle_variant["quarter"] = bundle["quarter"].copy()
    if market_vol_scale != 1.0:
        for column in ["price_spread", "load_dev", "renewable_dev"]:
            bundle_variant["hourly"][column] = bundle_variant["hourly"][column] * float(market_vol_scale)
    if price_cap_multiplier != 1.0:
        for column in ["全网统一出清价格_日前", "全网统一出清价格_日内"]:
            bundle_variant["quarter"][column] = bundle_variant["quarter"][column] * float(price_cap_multiplier)
    if forecast_error_scale != 1.0:
        for column in ["net_load_da", "net_load_da_mwh"]:
            bundle_variant["quarter"][column] = bundle_variant["quarter"][column] * float(forecast_error_scale)
        bundle_variant["hourly"]["net_load_da"] = bundle_variant["hourly"]["net_load_da"] * float(forecast_error_scale)
        bundle_variant["weekly_metadata"] = bundle["weekly_metadata"].copy()
        bundle_variant["weekly_metadata"]["forecast_weekly_net_demand_mwh"] = (
            bundle_variant["weekly_metadata"]["forecast_weekly_net_demand_mwh"] * float(forecast_error_scale)
        )
    else:
        bundle_variant["weekly_metadata"] = bundle["weekly_metadata"]

    for week in weeks:
        action = action_source(week) if callable(action_source) else action_source[pd.Timestamp(week)]
        summary, hourly_trace, settlement = simulate_week(
            bundle=bundle_variant,
            week_start=week,
            action=action,
            config=config,
            previous_lock_ratio=previous_lock_ratio,
        )
        summary["strategy"] = strategy_name
        hourly_trace["strategy"] = strategy_name
        settlement["strategy"] = strategy_name
        previous_lock_ratio = float(summary["lock_ratio"])

        weekly_records.append(summary)
        hourly_records.append(hourly_trace)
        settlement_records.append(settlement)

    weekly_results = pd.DataFrame(weekly_records).sort_values("week_start").reset_index(drop=True)
    hourly_results = pd.concat(hourly_records, ignore_index=True)
    settlement_results = pd.concat(settlement_records, ignore_index=True)
    metrics = summarize_strategy_results(
        weekly_results=weekly_results,
        settlement_results=settlement_results,
        strategy_name=strategy_name,
        cvar_alpha=float(config["cost"]["cvar_alpha"]),
    )
    return {
        "weekly_results": weekly_results,
        "hourly_results": hourly_results,
        "settlement_results": settlement_results,
        "metrics": metrics,
    }
