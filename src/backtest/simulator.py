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


def _resolve_lt_price(metadata: pd.Series) -> tuple[float, str]:
    if float(metadata.get("lt_price_linked_active", 0.0)) >= 0.5:
        fixed_ratio = float(metadata.get("fixed_price_ratio_max", 0.4) or 0.4)
        linked_ratio = float(metadata.get("linked_price_ratio_min", 0.6) or 0.6)
        total_ratio = fixed_ratio + linked_ratio
        if total_ratio <= 0.0:
            fixed_ratio, linked_ratio = 0.4, 0.6
            total_ratio = 1.0
        fixed_ratio /= total_ratio
        linked_ratio /= total_ratio
        lt_price = fixed_ratio * float(metadata["da_price_mean"]) + linked_ratio * float(metadata["id_price_mean"])
        return lt_price, "da_id_mixed_proxy"

    lt_price = float(metadata["lt_price_w"]) if pd.notna(metadata["lt_price_w"]) else float(metadata["da_price_mean"])
    return lt_price, str(metadata.get("lt_price_source", "estimated_prev_week_da_mean"))


def _robust_zscore(value: float, stats: dict[str, float], eps: float) -> float:
    median = float(stats.get("median", 0.0))
    iqr = float(stats.get("iqr", 1.0))
    return (float(value) - median) / max(iqr, eps)


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

    lt_price_w, lt_price_source = _resolve_lt_price(metadata)
    settlement = settle_week(quarter, hourly_trace, lt_price_w=lt_price_w, config=config)
    procurement_cost = float(settlement["procurement_cost_15m"].sum())
    risk_term, cvar_value = _week_risk_term(settlement, config)
    trans_cost = float(
        config["cost"]["transaction_linear"] * hourly_trace["delta_q"].abs().sum()
        + config["cost"]["transaction_quadratic"] * (hourly_trace["delta_q"] ** 2).sum()
        + config["cost"]["lt_adjust_cost"] * abs(q_lt_target - previous_lock_ratio * forecast_weekly_net_demand)
    )
    hedge_error = float(hourly_trace["hedge_error_abs"].mean())

    predicted_cost = max(float(metadata["forecast_weekly_net_demand_mwh"]) * max(float(metadata["da_price_mean"]), 1.0), 1.0)
    budget = max(
        float(config["cost"]["cvar_budget_ratio"]) * predicted_cost,
        float(config["cost"]["cvar_budget_min"]),
    )
    excess = max((cvar_value - budget) / (budget + float(config["cost"]["cvar_budget_eps"])), 0.0)
    penalty = float(np.log1p(excess))

    baseline_frame = bundle.get("reward_reference")
    baseline_row = None
    if isinstance(baseline_frame, pd.DataFrame) and not baseline_frame.empty:
        matched = baseline_frame.loc[baseline_frame["week_start"] == pd.Timestamp(week_start)]
        if not matched.empty:
            baseline_row = matched.iloc[0]

    delta_cost = procurement_cost - float(baseline_row["baseline_cost_w"]) if baseline_row is not None else procurement_cost
    delta_risk = risk_term - float(baseline_row["baseline_risk_w"]) if baseline_row is not None else risk_term

    robust_stats = bundle.get("reward_robust_stats", {})
    eps = float(config["reward"]["robust_eps"])
    z_delta_cost = _robust_zscore(delta_cost, robust_stats.get("delta_cost", {}), eps)
    z_delta_risk = _robust_zscore(delta_risk, robust_stats.get("delta_risk", {}), eps)
    z_trans_cost = _robust_zscore(trans_cost, robust_stats.get("trans_cost", {}), eps)
    z_hedge_error = _robust_zscore(hedge_error, robust_stats.get("hedge_error", {}), eps)

    reward_raw = -(
        float(config["reward"]["lambda_cost"]) * z_delta_cost
        + float(config["reward"]["lambda_risk"]) * z_delta_risk
        + float(config["reward"]["lambda_tc"]) * z_trans_cost
        + float(config["reward"]["lambda_he"]) * z_hedge_error
        + float(config["reward"]["lambda_penalty"]) * penalty
    )
    reward = float(np.tanh(reward_raw / float(config["env"]["reward_temperature"])))

    week_summary = {
        "week_start": pd.Timestamp(week_start),
        "is_partial_week": bool(metadata["is_partial_week"]),
        "lock_ratio": lock_ratio,
        "hedge_intensity": hedge_intensity,
        "q_lt_target_w": q_lt_target,
        "forecast_weekly_net_demand_mwh": forecast_weekly_net_demand,
        "actual_weekly_net_demand_mwh": float(metadata["actual_weekly_net_demand_mwh"]),
        "lt_price_w": lt_price_w,
        "lt_price_source": lt_price_source,
        "procurement_cost_w": procurement_cost,
        "risk_term_w": risk_term,
        "trans_cost_w": trans_cost,
        "hedge_error_w": hedge_error,
        "cvar_w": cvar_value,
        "baseline_cost_w": float(baseline_row["baseline_cost_w"]) if baseline_row is not None else np.nan,
        "baseline_risk_w": float(baseline_row["baseline_risk_w"]) if baseline_row is not None else np.nan,
        "delta_cost_w": delta_cost,
        "delta_risk_w": delta_risk,
        "reward_penalty_w": penalty,
        "reward_budget_w": budget,
        "reward_excess_w": excess,
        "z_delta_cost_w": z_delta_cost,
        "z_delta_risk_w": z_delta_risk,
        "z_trans_cost_w": z_trans_cost,
        "z_hedge_error_w": z_hedge_error,
        "reward_raw": reward_raw,
        "reward": reward,
        "avg_adjustment_mwh": float(hourly_trace["delta_q"].mean()),
        "bound_clip_count": int(rule_stats["bound_clip_count"]),
        "smooth_clip_count": int(rule_stats["smooth_clip_count"]),
        "non_negative_clip_count": int(rule_stats["non_negative_clip_count"]),
        "cvar_budget_excess": excess,
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
