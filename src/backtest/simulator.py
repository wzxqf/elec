from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.benchmarks import get_dynamic_lock_base_for_week
from src.backtest.runtime_cache import prepare_runtime_bundle
from src.backtest.metrics import summarize_strategy_results
from src.backtest.settlement import resolve_settlement_context, settle_week
from src.rules.hourly_hedge import apply_hourly_hedge_rule


def _get_week_frames(bundle: dict[str, Any], week_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    week_start = pd.Timestamp(week_start)
    quarter_lookup = bundle.get("quarter_by_week")
    hourly_lookup = bundle.get("hourly_by_week")
    metadata_lookup = bundle.get("weekly_metadata_by_week")

    if isinstance(quarter_lookup, dict) and isinstance(hourly_lookup, dict) and isinstance(metadata_lookup, dict):
        if week_start not in quarter_lookup or week_start not in hourly_lookup or week_start not in metadata_lookup:
            raise KeyError(f"未找到周度缓存切片: {week_start}")
        return (
            quarter_lookup[week_start].copy(),
            hourly_lookup[week_start].copy(),
            metadata_lookup[week_start].copy(),
        )

    quarter = bundle["quarter"].loc[bundle["quarter"]["week_start"] == week_start].copy()
    hourly = bundle["hourly"].loc[bundle["hourly"]["week_start"] == week_start].copy()
    metadata = bundle["weekly_metadata"].set_index("week_start").loc[week_start].copy()
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
    risk_term = float(config["cost"]["risk_vol_weight"]) * sigma + float(config["cost"]["risk_cvar_weight"]) * cvar
    return risk_term, cvar


def _robust_zscore(value: float, stats: dict[str, float], eps: float) -> float:
    median = float(stats.get("median", 0.0))
    iqr = float(stats.get("iqr", 1.0))
    return (float(value) - median) / max(iqr, eps)


def _resolve_action_payload(
    bundle: dict[str, Any],
    week_start: pd.Timestamp,
    action: tuple[float, float] | dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    delta_lock_cap = float(config["constraints"]["delta_lock_cap"])
    dynamic_base = get_dynamic_lock_base_for_week(
        bundle["weekly_features"],
        week_start,
        config,
        weekly_feature_by_week=bundle.get("weekly_feature_by_week"),
    )
    if isinstance(action, dict):
        mode = str(action.get("mode", "residual"))
        exposure_bandwidth = float(np.clip(action.get("exposure_bandwidth", 0.0), 0.0, 1.0))
        if mode == "absolute":
            target_lock_ratio = float(action.get("target_lock_ratio", dynamic_base))
            delta_lock_ratio = target_lock_ratio - dynamic_base
            delta_lock_ratio_raw = float(np.clip(delta_lock_ratio / max(delta_lock_cap, 1e-6), -1.0, 1.0))
        else:
            delta_lock_ratio_raw = float(np.clip(action.get("delta_lock_ratio_raw", 0.0), -1.0, 1.0))
            delta_lock_ratio = float(delta_lock_ratio_raw * delta_lock_cap)
            target_lock_ratio = dynamic_base + delta_lock_ratio
    else:
        delta_lock_ratio_raw = float(np.clip(action[0], -1.0, 1.0))
        exposure_bandwidth = float(np.clip(action[1], 0.0, 1.0))
        delta_lock_ratio = float(delta_lock_ratio_raw * delta_lock_cap)
        target_lock_ratio = dynamic_base + delta_lock_ratio
    return {
        "lock_ratio_base": dynamic_base,
        "delta_lock_ratio_raw": delta_lock_ratio_raw,
        "delta_lock_ratio": float(target_lock_ratio - dynamic_base),
        "target_lock_ratio": float(target_lock_ratio),
        "exposure_bandwidth": exposure_bandwidth,
    }


def simulate_week(
    bundle: dict[str, Any],
    week_start: pd.Timestamp,
    action: tuple[float, float] | dict[str, Any],
    config: dict[str, Any],
    previous_lock_ratio: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    quarter, hourly, metadata = _get_week_frames(bundle, week_start)
    action_payload = _resolve_action_payload(bundle, pd.Timestamp(week_start), action, config)
    lock_ratio_base = float(action_payload["lock_ratio_base"])
    delta_lock_ratio_raw = float(action_payload["delta_lock_ratio_raw"])
    delta_lock_ratio = float(action_payload["delta_lock_ratio"])
    exposure_bandwidth = float(action_payload["exposure_bandwidth"])
    constraints = config["constraints"]

    lock_ratio_preclip = float(
        np.clip(action_payload["target_lock_ratio"], float(constraints["lock_ratio_min"]), float(constraints["lock_ratio_max"]))
    )
    max_step = float(constraints["delta_h_max"])
    lock_ratio_final = float(np.clip(lock_ratio_preclip, previous_lock_ratio - max_step, previous_lock_ratio + max_step))
    lock_ratio_final = float(
        np.clip(lock_ratio_final, float(constraints["lock_ratio_min"]), float(constraints["lock_ratio_max"]))
    )

    forecast_weekly_net_demand = float(metadata["forecast_weekly_net_demand_mwh"])
    q_lt_target = max(lock_ratio_final * forecast_weekly_net_demand, 0.0)
    q_lt_hourly = _allocate_weekly_lt_to_hourly(hourly, q_lt_target)
    hourly_trace, rule_stats = apply_hourly_hedge_rule(
        hourly,
        q_lt_hourly=q_lt_hourly,
        exposure_bandwidth=exposure_bandwidth,
        policy_state=metadata,
        rules_config=config["rules"],
    )

    settlement_context = resolve_settlement_context(quarter, metadata, config)
    settlement = settle_week(quarter, hourly_trace, metadata=metadata, config=config)
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
    tail_penalty = float(np.log1p(excess))

    baseline_lookup = bundle.get("reward_reference_by_week")
    baseline_row = None
    if isinstance(baseline_lookup, dict):
        baseline_row = baseline_lookup.get(pd.Timestamp(week_start))
    if baseline_row is None:
        baseline_frame = bundle.get("reward_reference")
        if isinstance(baseline_frame, pd.DataFrame) and not baseline_frame.empty:
            matched = baseline_frame.loc[baseline_frame["week_start"] == pd.Timestamp(week_start)]
            if not matched.empty:
                baseline_row = matched.iloc[0]

    baseline_cost = float(baseline_row["baseline_cost_w"]) if baseline_row is not None else procurement_cost
    delta_cost = procurement_cost - baseline_cost
    robust_stats = bundle.get("reward_robust_stats", {})
    eps = float(config["reward"]["robust_eps"])
    z_delta_cost = _robust_zscore(delta_cost, robust_stats.get("delta_cost", {}), eps)

    action_smooth = abs(lock_ratio_final - previous_lock_ratio) / max(float(constraints["delta_h_max"]), eps)
    hourly_smooth = float(hourly_trace["delta_q"].diff().abs().fillna(0.0).mean()) / max(float(config["rules"]["gamma_max"]), eps)
    hedge_error_norm = hedge_error / max(float(hourly["net_load_id"].clip(lower=0.0).mean()), eps)
    exec_quality = (
        float(config["reward"]["lambda_action_smooth"]) * action_smooth
        + float(config["reward"]["lambda_hourly_smooth"]) * hourly_smooth
        + float(config["reward"]["lambda_hedge_error"]) * hedge_error_norm
    )
    reward_raw = -(
        float(config["reward"]["alpha_cost"]) * z_delta_cost
        + float(config["reward"]["beta_tail_risk"]) * tail_penalty
        + exec_quality
    )
    reward = float(np.tanh(reward_raw / float(config["env"]["reward_temperature"])))

    week_summary = {
        "week_start": pd.Timestamp(week_start),
        "is_partial_week": bool(metadata["is_partial_week"]),
        "lock_ratio_base": lock_ratio_base,
        "delta_lock_ratio_raw": delta_lock_ratio_raw,
        "delta_lock_ratio": delta_lock_ratio,
        "lock_ratio_final": lock_ratio_final,
        "lock_ratio": lock_ratio_final,
        "exposure_bandwidth": exposure_bandwidth,
        "q_lt_target_w": q_lt_target,
        "forecast_weekly_net_demand_mwh": forecast_weekly_net_demand,
        "actual_weekly_net_demand_mwh": float(metadata["actual_weekly_net_demand_mwh"]),
        "lt_price_w": float(settlement_context["lt_price_w"]),
        "lt_price_source": settlement_context["lt_price_regime"],
        "procurement_cost_w": procurement_cost,
        "risk_term_w": risk_term,
        "trans_cost_w": trans_cost,
        "hedge_error_w": hedge_error,
        "cvar_w": cvar_value,
        "baseline_cost_w": baseline_cost,
        "delta_cost_w": delta_cost,
        "tail_penalty_w": tail_penalty,
        "reward_budget_w": budget,
        "reward_excess_w": excess,
        "z_delta_cost_w": z_delta_cost,
        "action_smooth_w": action_smooth,
        "hourly_smooth_w": hourly_smooth,
        "hedge_error_norm_w": hedge_error_norm,
        "exec_quality_w": exec_quality,
        "reward_raw": reward_raw,
        "reward": reward,
        "avg_adjustment_mwh": float(hourly_trace["delta_q"].abs().mean()),
        "mean_bandwidth_mwh": float(hourly_trace["bandwidth_mwh"].mean()),
        "bound_clip_count": int(rule_stats["bound_clip_count"]),
        "smooth_clip_count": int(rule_stats["smooth_clip_count"]),
        "soft_clip_count": int(rule_stats.get("soft_clip_count", 0)),
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

    if any(scale != 1.0 for scale in (market_vol_scale, price_cap_multiplier, forecast_error_scale)):
        prepare_runtime_bundle(bundle_variant)

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
