from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.metrics import cvar_95, summarize_strategy_results
from src.rules.hourly_hedge import compute_hourly_hedge_adjustment


def _get_month_frames(bundle: dict, month: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    quarter = bundle["quarter"].loc[bundle["quarter"]["month"] == month].copy()
    hourly = bundle["hourly"].loc[bundle["hourly"]["month"] == month].copy()
    metadata = bundle["monthly_metadata"].set_index("month").loc[month]
    return quarter, hourly, metadata


def simulate_month(
    bundle: dict,
    month: pd.Timestamp,
    action: tuple[float, float],
    config: dict,
    market_vol_scale: float = 1.0,
    forecast_error_scale: float = 1.0,
    price_cap_multiplier: float = 1.0,
    hedge_intensity_scale: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    quarter, hourly, metadata = _get_month_frames(bundle, month)
    lock_ratio = float(np.clip(action[0], 0.0, 1.0))
    hedge_intensity = float(np.clip(action[1], 0.0, 1.0))
    interval_hours = 0.25

    quarter["forecast_energy_mwh"] = quarter["net_load_da"] * interval_hours
    quarter["actual_energy_mwh"] = quarter["net_load_id"] * interval_hours
    hourly["forecast_energy_mwh"] = hourly["net_load_da"]

    forecast_monthly_net_demand = float(metadata["forecast_monthly_net_demand_mwh"])
    lt_price = metadata["lt_price_m"]
    if pd.isna(lt_price):
        lt_price = float(metadata["da_price_mean"])

    q_lt_target = lock_ratio * forecast_monthly_net_demand
    positive_hourly_energy = hourly["forecast_energy_mwh"].clip(lower=0.0)
    total_positive_hourly_energy = float(positive_hourly_energy.sum())
    if total_positive_hourly_energy == 0.0:
        lt_alloc_hour = pd.Series(0.0, index=hourly.index)
    else:
        lt_alloc_hour = q_lt_target * positive_hourly_energy / total_positive_hourly_energy

    residual_exposure_hour = hourly["forecast_energy_mwh"] - lt_alloc_hour
    hedge_adjust_hour = compute_hourly_hedge_adjustment(
        hourly_frame=hourly,
        residual_exposure_mwh=residual_exposure_hour,
        hedge_intensity=hedge_intensity,
        rule_config=config["rules"],
        market_vol_scale=market_vol_scale,
        forecast_error_scale=forecast_error_scale,
        price_cap_multiplier=price_cap_multiplier,
        hedge_intensity_scale=hedge_intensity_scale,
    )
    hourly["lt_alloc_mwh"] = lt_alloc_hour
    hourly["hedge_adjust_mwh"] = hedge_adjust_hour

    quarter = quarter.merge(hourly[["hour", "lt_alloc_mwh", "hedge_adjust_mwh"]], on="hour", how="left")
    quarter["quarter_weight"] = quarter["forecast_energy_mwh"].clip(lower=0.0)
    hourly_weights = quarter.groupby("hour")["quarter_weight"].transform("sum")
    fallback_mask = hourly_weights == 0
    quarter.loc[fallback_mask, "quarter_weight"] = interval_hours
    hourly_weights = quarter.groupby("hour")["quarter_weight"].transform("sum")

    quarter["lt_alloc_q_mwh"] = np.where(
        hourly_weights > 0,
        quarter["lt_alloc_mwh"] * quarter["quarter_weight"] / hourly_weights,
        0.0,
    )
    quarter["hedge_adjust_q_mwh"] = quarter["hedge_adjust_mwh"] / 4.0
    quarter["da_residual_q_mwh"] = quarter["forecast_energy_mwh"] - quarter["lt_alloc_q_mwh"]
    quarter["imbalance_q_mwh"] = (
        quarter["actual_energy_mwh"] - quarter["forecast_energy_mwh"] - quarter["hedge_adjust_q_mwh"]
    )

    penalty_multiplier = config["cost"]["penalty_multiplier"]
    transaction_fee_rate = config["cost"]["transaction_fee_rate"]
    quarter["lt_cost"] = quarter["lt_alloc_q_mwh"] * lt_price
    quarter["dayahead_cost"] = quarter["da_residual_q_mwh"] * quarter["全网统一出清价格_日前"]
    quarter["intraday_adjust_cost"] = quarter["hedge_adjust_q_mwh"] * quarter["全网统一出清价格_日内"]
    quarter["imbalance_penalty_cost"] = (
        quarter["imbalance_q_mwh"].abs() * quarter["全网统一出清价格_日内"] * penalty_multiplier
    )
    quarter["trading_cost"] = quarter["hedge_adjust_q_mwh"].abs() * transaction_fee_rate
    quarter["procurement_cost"] = (
        quarter["lt_cost"]
        + quarter["dayahead_cost"]
        + quarter["intraday_adjust_cost"]
        + quarter["imbalance_penalty_cost"]
    )
    quarter["total_cost_with_penalties"] = quarter["procurement_cost"] + quarter["trading_cost"]

    risk_vol = float(quarter["total_cost_with_penalties"].std(ddof=0))
    risk_cvar = cvar_95(quarter["total_cost_with_penalties"])
    risk_term = (
        config["cost"]["risk_vol_weight"] * risk_vol
        + config["cost"]["risk_cvar_weight"] * risk_cvar
    )
    procurement_cost_m = float(quarter["procurement_cost"].sum())
    trading_cost_m = float(quarter["trading_cost"].sum())
    hedge_error_m = float(quarter["imbalance_q_mwh"].abs().mean())
    reward = -(
        procurement_cost_m
        + config["cost"]["lambda_risk"] * risk_term
        + config["cost"]["lambda_tc"] * trading_cost_m
        + config["cost"]["lambda_he"] * hedge_error_m
    ) / config["env"]["reward_scale"]

    summary = {
        "month": month,
        "lock_ratio": lock_ratio,
        "hedge_intensity": hedge_intensity,
        "q_lt_target_mwh": q_lt_target,
        "lt_price_m": float(lt_price),
        "procurement_cost_m": procurement_cost_m,
        "trading_cost_m": trading_cost_m,
        "risk_term_m": risk_term,
        "hedge_error_m": hedge_error_m,
        "avg_adjustment_mwh": float(quarter["hedge_adjust_q_mwh"].abs().mean()),
        "reward": reward,
    }
    quarter["month"] = month
    return summary, quarter


def simulate_strategy(
    bundle: dict,
    months: list[pd.Timestamp],
    action_source: dict[pd.Timestamp, tuple[float, float]] | Callable[[pd.Timestamp], tuple[float, float]],
    config: dict,
    strategy_name: str,
    market_vol_scale: float = 1.0,
    forecast_error_scale: float = 1.0,
    price_cap_multiplier: float = 1.0,
    hedge_intensity_scale: float = 1.0,
) -> dict:
    monthly_records = []
    interval_frames = []
    for month in months:
        action = action_source(month) if callable(action_source) else action_source[month]
        summary, interval_frame = simulate_month(
            bundle=bundle,
            month=month,
            action=action,
            config=config,
            market_vol_scale=market_vol_scale,
            forecast_error_scale=forecast_error_scale,
            price_cap_multiplier=price_cap_multiplier,
            hedge_intensity_scale=hedge_intensity_scale,
        )
        summary["strategy"] = strategy_name
        monthly_records.append(summary)
        interval_frame["strategy"] = strategy_name
        interval_frames.append(interval_frame)

    monthly_results = pd.DataFrame(monthly_records)
    interval_results = pd.concat(interval_frames, ignore_index=True) if interval_frames else pd.DataFrame()
    metrics = summarize_strategy_results(monthly_results, interval_results)
    metrics["strategy"] = strategy_name
    return {
        "monthly_results": monthly_results,
        "interval_results": interval_results,
        "metrics": metrics,
    }
