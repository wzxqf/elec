from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.model_layout.schema import CompiledParameterLayout
from src.training.score_kernel import batch_score_particles
from src.training.tensor_bundle import TrainingTensorBundle


@dataclass(frozen=True)
class MaterializedStrategyResult:
    weekly_results: pd.DataFrame
    hourly_results: pd.DataFrame
    settlement_results: pd.DataFrame
    metrics: dict[str, Any]


def _aggregate_hourly_profile_to_24(hourly_load: np.ndarray) -> np.ndarray:
    if hourly_load.size == 0:
        return np.zeros(24, dtype=np.float64)
    aggregated = np.zeros(24, dtype=np.float64)
    counts = np.zeros(24, dtype=np.float64)
    for hour_index, load_value in enumerate(hourly_load):
        slot = hour_index % 24
        aggregated[slot] += float(load_value)
        counts[slot] += 1.0
    counts[counts == 0.0] = 1.0
    return aggregated / counts


def _compute_metrics(
    weekly_results: pd.DataFrame,
    hourly_results: pd.DataFrame,
    settlement_results: pd.DataFrame,
    strategy_name: str,
) -> dict[str, Any]:
    total_procurement_cost = float(weekly_results["procurement_cost_w"].sum()) if not weekly_results.empty else 0.0
    weekly_cost_volatility = float(weekly_results["procurement_cost_w"].std(ddof=0)) if len(weekly_results) > 1 else 0.0
    total_profit = float(weekly_results["profit_w"].sum()) if not weekly_results.empty else 0.0
    max_drawdown = 0.0
    if not weekly_results.empty:
        cumulative = weekly_results["profit_w"].cumsum()
        running_peak = cumulative.cummax()
        max_drawdown = float((running_peak - cumulative).max())
    cvar99 = float(weekly_results["cvar99_w"].mean()) if "cvar99_w" in weekly_results and not weekly_results.empty else 0.0
    return {
        "strategy": strategy_name,
        "total_procurement_cost": total_procurement_cost,
        "total_profit": total_profit,
        "weekly_cost_volatility": weekly_cost_volatility,
        "cvar99": cvar99,
        "hedge_error": float(weekly_results["hedge_error_w"].mean()) if not weekly_results.empty else 0.0,
        "avg_adjustment_mwh": float(hourly_results["spot_hedge_mwh"].abs().mean()) if "spot_hedge_mwh" in hourly_results else 0.0,
        "mean_reward": float(weekly_results["reward_w"].mean()) if not weekly_results.empty else 0.0,
        "max_drawdown": max_drawdown,
    }


def materialize_particle_pair(
    *,
    tensor_bundle: TrainingTensorBundle,
    upper_particle: list[float] | torch.Tensor,
    lower_particle: list[float] | torch.Tensor,
    strategy_name: str,
    config: dict[str, Any] | None = None,
    compiled_layout: CompiledParameterLayout | None = None,
) -> MaterializedStrategyResult:
    upper = torch.as_tensor(upper_particle, dtype=torch.float32, device=tensor_bundle.weekly_feature_tensor.device).view(1, -1)
    lower = torch.as_tensor(lower_particle, dtype=torch.float32, device=tensor_bundle.weekly_feature_tensor.device).view(1, -1)
    scored = batch_score_particles(
        tensor_bundle,
        upper,
        lower,
        device=tensor_bundle.device,
        config=config,
        compiled_layout=compiled_layout,
    )
    contract_adjustment_raw = scored.contract_adjustment_mwh_raw[0, 0].detach().cpu().numpy()
    contract_adjustment_exec = scored.contract_adjustment_mwh_exec[0, 0].detach().cpu().numpy()
    contract_position = scored.contract_position_mwh[0, 0].detach().cpu().numpy()
    exposure_band = scored.exposure_band_mwh[0, 0].detach().cpu().numpy()
    policy_projection_active = scored.policy_projection_active[0, 0].detach().cpu().numpy()
    contract_curve = scored.contract_curve[0, 0].detach().cpu().numpy()
    spot_hedge = scored.spot_hedge_mwh[0, 0].detach().cpu().numpy()
    spot_hedge_limit = scored.spot_hedge_limit_mwh[0, 0].detach().cpu().numpy()
    cvar99 = scored.weekly_cvar99[0, 0].detach().cpu().numpy()
    profit = scored.weekly_profit[0, 0].detach().cpu().numpy()
    reward = scored.weekly_reward[0, 0].detach().cpu().numpy()
    procurement_cost = scored.weekly_procurement_cost[0, 0].detach().cpu().numpy()
    hedge_error = scored.weekly_hedge_error[0, 0].detach().cpu().numpy()
    profit_baseline = scored.weekly_profit_baseline[0, 0].detach().cpu().numpy()
    excess_profit = scored.weekly_excess_profit[0, 0].detach().cpu().numpy()
    retail_revenue = scored.weekly_retail_revenue[0, 0].detach().cpu().numpy()
    imbalance_cost = scored.weekly_imbalance_cost[0, 0].detach().cpu().numpy()
    adjustment_cost = scored.weekly_adjustment_cost[0, 0].detach().cpu().numpy()
    friction_cost = scored.weekly_friction_cost[0, 0].detach().cpu().numpy()
    violation_penalty = scored.weekly_policy_violation_penalty[0, 0].detach().cpu().numpy()
    actual_load = tensor_bundle.actual_weekly_load.detach().cpu().numpy()
    forecast_load = tensor_bundle.forecast_weekly_load.detach().cpu().numpy()
    hourly_tensor = tensor_bundle.hourly_tensor.detach().cpu().numpy()
    price_tensor = tensor_bundle.quarter_price_tensor.detach().cpu().numpy()
    hour_mask = tensor_bundle.hourly_valid_mask.detach().cpu().numpy()
    quarter_mask = tensor_bundle.quarter_valid_mask.detach().cpu().numpy()
    week_index = tensor_bundle.week_index

    weekly_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    settlement_rows: list[dict[str, Any]] = []
    for week_pos, week_start in enumerate(week_index):
        weekly_row = {
            "week_start": pd.Timestamp(week_start),
            "contract_adjustment_mwh_raw": float(contract_adjustment_raw[week_pos]),
            "contract_adjustment_mwh_exec": float(contract_adjustment_exec[week_pos]),
            "contract_position_mwh": float(contract_position[week_pos]),
            "exposure_band_mwh": float(exposure_band[week_pos]),
            "policy_projection_active": float(policy_projection_active[week_pos]),
            "policy_violation_penalty_w": float(violation_penalty[week_pos]),
            "retail_revenue_w": float(retail_revenue[week_pos]),
            "procurement_cost_w": float(procurement_cost[week_pos]),
            "imbalance_cost_w": float(imbalance_cost[week_pos]),
            "adjustment_cost_w": float(adjustment_cost[week_pos]),
            "friction_cost_w": float(friction_cost[week_pos]),
            "profit_w": float(profit[week_pos]),
            "profit_baseline_w": float(profit_baseline[week_pos]),
            "excess_profit_w": float(excess_profit[week_pos]),
            "cvar99_w": float(cvar99[week_pos]),
            "hedge_error_w": float(hedge_error[week_pos]),
            "reward_w": float(reward[week_pos]),
            "strategy": strategy_name,
        }
        forecast_weekly_load = max(float(forecast_load[week_pos]), 1.0)
        lock_ratio_proxy = float(contract_position[week_pos]) / forecast_weekly_load
        valid_hour_mask = hour_mask[week_pos].astype(bool)
        if valid_hour_mask.any():
            load_shape = _aggregate_hourly_profile_to_24(hourly_tensor[week_pos, valid_hour_mask, 0])
            load_shape = load_shape / max(float(load_shape.sum()), 1.0e-6)
            curve_slice = contract_curve[week_pos, :24]
            curve_slice = curve_slice / max(float(curve_slice.sum()), 1.0e-6)
            curve_match_score = float(np.clip(1.0 - 0.5 * np.abs(load_shape - curve_slice).sum(), 0.0, 1.0))
        else:
            curve_match_score = 0.0
        cvar_scale = float(cvar99[week_pos]) / max(float(np.mean(cvar99)), 1.0e-6)
        stability_score = float(
            np.clip(
                0.35 * np.clip(lock_ratio_proxy, 0.0, 1.0)
                + 0.35 * curve_match_score
                + 0.15 * np.clip(1.0 - float(hedge_error[week_pos]), 0.0, 1.0)
                + 0.15 * (1.0 / (1.0 + cvar_scale)),
                0.0,
                1.0,
            )
        )
        weekly_row["lock_ratio_proxy_w"] = lock_ratio_proxy
        weekly_row["curve_match_score_w"] = curve_match_score
        weekly_row["stability_score_w"] = stability_score
        for curve_idx in range(contract_curve.shape[-1]):
            weekly_row[f"contract_curve_h{curve_idx + 1}"] = float(contract_curve[week_pos, curve_idx])
        weekly_rows.append(weekly_row)
        for hour_index in range(len(hour_mask[week_pos])):
            if not bool(hour_mask[week_pos, hour_index]):
                continue
            hourly_rows.append(
                {
                    "week_start": pd.Timestamp(week_start),
                    "hour_index": int(hour_index),
                    "spot_hedge_mwh": float(spot_hedge[week_pos, hour_index]),
                    "spot_hedge_limit_mwh": float(spot_hedge_limit[week_pos, hour_index]),
                    "strategy": strategy_name,
                }
            )
        valid_intervals = max(int(quarter_mask[week_pos].sum()), 1)
        scheduled_15m = float((contract_position[week_pos] + abs(spot_hedge[week_pos]).sum()) / valid_intervals)
        actual_15m = float(actual_load[week_pos] / valid_intervals)
        for interval_index in range(len(quarter_mask[week_pos])):
            if not bool(quarter_mask[week_pos, interval_index]):
                continue
            da_price_15m = float(price_tensor[week_pos, interval_index, 0])
            id_price_15m = float(price_tensor[week_pos, interval_index, 1])
            interval_cost = scheduled_15m * da_price_15m + abs(actual_15m - scheduled_15m) * id_price_15m
            settlement_rows.append(
                {
                    "week_start": pd.Timestamp(week_start),
                    "interval_index": int(interval_index),
                    "scheduled_energy_15m": scheduled_15m,
                    "actual_need_15m": actual_15m,
                    "procurement_cost_15m": interval_cost,
                    "contract_position_mwh": float(contract_position[week_pos]),
                    "spot_hedge_energy_15m": float(abs(spot_hedge[week_pos]).sum() / valid_intervals),
                    "strategy": strategy_name,
                }
            )

    weekly_results = pd.DataFrame(weekly_rows)
    hourly_results = pd.DataFrame(hourly_rows)
    settlement_results = pd.DataFrame(settlement_rows)
    metrics = _compute_metrics(weekly_results, hourly_results, settlement_results, strategy_name)
    return MaterializedStrategyResult(
        weekly_results=weekly_results,
        hourly_results=hourly_results,
        settlement_results=settlement_results,
        metrics=metrics,
    )
