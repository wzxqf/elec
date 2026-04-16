from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.training.tensor_bundle import TrainingTensorBundle


@dataclass(frozen=True)
class ParticleScoreResult:
    total_score: torch.Tensor
    procurement_cost: torch.Tensor
    profit: torch.Tensor
    reward: torch.Tensor
    cvar99: torch.Tensor
    hedge_error: torch.Tensor
    retail_revenue: torch.Tensor
    imbalance_cost: torch.Tensor
    adjustment_cost: torch.Tensor
    friction_cost: torch.Tensor
    policy_violation_penalty: torch.Tensor
    profit_baseline: torch.Tensor
    excess_profit: torch.Tensor
    contract_adjustment_mwh_raw: torch.Tensor
    contract_adjustment_mwh_exec: torch.Tensor
    contract_position_mwh: torch.Tensor
    exposure_band_mwh: torch.Tensor
    policy_projection_active: torch.Tensor
    contract_curve: torch.Tensor
    spot_hedge_mwh: torch.Tensor
    spot_hedge_limit_mwh: torch.Tensor
    weekly_procurement_cost: torch.Tensor
    weekly_profit: torch.Tensor
    weekly_reward: torch.Tensor
    weekly_cvar99: torch.Tensor
    weekly_hedge_error: torch.Tensor
    weekly_profit_baseline: torch.Tensor
    weekly_excess_profit: torch.Tensor
    weekly_retail_revenue: torch.Tensor
    weekly_imbalance_cost: torch.Tensor
    weekly_adjustment_cost: torch.Tensor
    weekly_friction_cost: torch.Tensor
    weekly_policy_violation_penalty: torch.Tensor


def _resolve_device(device: str) -> str:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _cfg(config: dict[str, Any] | None, path: list[str], default: float) -> float:
    current: Any = config or {}
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return float(default)
        current = current[key]
    return float(current)


def batch_score_particles(
    tensor_bundle: TrainingTensorBundle,
    upper_particles: torch.Tensor,
    lower_particles: torch.Tensor,
    device: str = "cpu",
    config: dict[str, Any] | None = None,
) -> ParticleScoreResult:
    score_device = _resolve_device(device)
    weekly_features = torch.nan_to_num(tensor_bundle.weekly_feature_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    policy_tensor = torch.nan_to_num(tensor_bundle.policy_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    hourly_tensor = torch.nan_to_num(tensor_bundle.hourly_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    forecast_load = torch.nan_to_num(tensor_bundle.forecast_weekly_load.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    actual_load = torch.nan_to_num(tensor_bundle.actual_weekly_load.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    lt_price = torch.nan_to_num(tensor_bundle.lt_weekly_price.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    price_tensor = torch.nan_to_num(tensor_bundle.quarter_price_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    hour_mask = tensor_bundle.hourly_valid_mask.to(score_device).unsqueeze(0).unsqueeze(0)
    quarter_mask = tensor_bundle.quarter_valid_mask.to(score_device).unsqueeze(0).unsqueeze(0)

    upper = upper_particles.to(score_device)
    lower = lower_particles.to(score_device)
    upper_count, lower_count = upper.shape[0], lower.shape[0]
    week_count, hour_count = weekly_features.shape[0], hourly_tensor.shape[1]
    interval_count = price_tensor.shape[1]

    feature_dim = min(weekly_features.shape[1], upper.shape[1])
    feature_signal = weekly_features[:, :feature_dim] @ upper[:, :feature_dim].T
    feature_signal = feature_signal.transpose(0, 1).unsqueeze(1).expand(upper_count, lower_count, week_count)

    policy_dim = min(policy_tensor.shape[1], upper.shape[1])
    if policy_dim > 0:
        policy_signal = policy_tensor[:, :policy_dim] @ upper[:, :policy_dim].T
        policy_signal = policy_signal.transpose(0, 1).unsqueeze(1).expand(upper_count, lower_count, week_count)
        policy_projection_active = (policy_tensor[:, :policy_dim].abs().mean(dim=1) > 0).float().view(1, 1, week_count)
    else:
        policy_signal = torch.zeros((upper_count, lower_count, week_count), dtype=torch.float32, device=score_device)
        policy_projection_active = torch.zeros((1, 1, week_count), dtype=torch.float32, device=score_device)

    # Upper layer: weekly contract adjustment, contract position, exposure band.
    contract_adjustment_mwh_raw = 0.30 * torch.tanh(0.15 * feature_signal + 0.05 * policy_signal) * forecast_load.view(1, 1, week_count)
    policy_scale = torch.clamp(1.0 - 0.15 * torch.sigmoid(policy_signal), min=0.60, max=1.0)
    contract_adjustment_mwh_exec = contract_adjustment_mwh_raw * policy_scale
    contract_position_mwh = torch.relu(0.60 * forecast_load.view(1, 1, week_count) + contract_adjustment_mwh_exec)
    exposure_band_mwh = torch.relu(0.20 * forecast_load.view(1, 1, week_count) * (1.0 + torch.tanh(0.10 * feature_signal)))

    # Build a smooth 24h contract curve from four shape factors inferred from the upper particle.
    curve_base = torch.linspace(-1.0, 1.0, steps=24, device=score_device, dtype=torch.float32)
    upper_curve_seed = upper[:, -4:] if upper.shape[1] >= 4 else torch.nn.functional.pad(upper, (max(0, 4 - upper.shape[1]), 0))
    curve_components = torch.stack(
        [
            torch.ones_like(curve_base),
            curve_base,
            torch.cos(torch.pi * curve_base),
            torch.sin(torch.pi * curve_base),
        ],
        dim=0,
    )
    contract_curve_logits = torch.einsum("uc,ch->uh", upper_curve_seed[:, :4], curve_components)
    contract_curve = torch.softmax(contract_curve_logits, dim=-1).unsqueeze(1).unsqueeze(2).expand(upper_count, lower_count, week_count, 24)

    # Lower layer: hourly spot hedge responses within the exposure band.
    spread = hourly_tensor[..., 2].view(1, 1, week_count, hour_count)
    load_dev = hourly_tensor[..., 3].view(1, 1, week_count, hour_count)
    renewable_dev = hourly_tensor[..., 4].view(1, 1, week_count, hour_count)
    lower_groups = lower.view(lower_count, 4, -1).mean(dim=-1)
    spread_coef = lower_groups[:, 0].view(1, lower_count, 1, 1)
    load_coef = lower_groups[:, 1].view(1, lower_count, 1, 1)
    renewable_coef = lower_groups[:, 2].view(1, lower_count, 1, 1)
    policy_shrink = torch.sigmoid(lower_groups[:, 3]).view(1, lower_count, 1, 1)
    spot_hedge_limit_mwh = (exposure_band_mwh.unsqueeze(-1) / max(hour_count, 1)) * (0.50 + 0.50 * policy_shrink)
    spot_hedge_limit_mwh = spot_hedge_limit_mwh.expand(upper_count, lower_count, week_count, hour_count)
    raw_hourly_signal = 0.02 * spread_coef * spread + 0.01 * load_coef * load_dev + 0.01 * renewable_coef * renewable_dev
    spot_hedge_mwh = torch.tanh(raw_hourly_signal) * spot_hedge_limit_mwh * hour_mask

    # 15-minute proxy settlement and CVaR99.
    avg_da_price = torch.nan_to_num(price_tensor[..., 0].mean(dim=1), nan=0.0).view(1, 1, week_count)
    avg_id_price = torch.nan_to_num(price_tensor[..., 1].mean(dim=1), nan=0.0).view(1, 1, week_count)
    valid_intervals = quarter_mask.sum(dim=-1).clamp_min(1.0)
    spot_energy_total = (spot_hedge_mwh.abs() * hour_mask).sum(dim=-1)
    scheduled_energy = contract_position_mwh + spot_energy_total
    imbalance_energy = torch.abs(actual_load.view(1, 1, week_count) - scheduled_energy)

    per_interval_contract = contract_position_mwh.unsqueeze(-1) / valid_intervals.unsqueeze(-1)
    per_interval_spot = spot_energy_total.unsqueeze(-1) / valid_intervals.unsqueeze(-1)
    scheduled_interval = (per_interval_contract + per_interval_spot) * quarter_mask
    actual_interval = (actual_load.view(1, 1, week_count, 1) / valid_intervals.unsqueeze(-1)) * quarter_mask
    da_price = price_tensor[..., 0].view(1, 1, week_count, interval_count)
    id_price = price_tensor[..., 1].view(1, 1, week_count, interval_count)
    lt_interval_price = lt_price.view(1, 1, week_count, 1)
    interval_cost = scheduled_interval * (0.6 * lt_interval_price + 0.4 * da_price)
    interval_cost = interval_cost + torch.abs(actual_interval - scheduled_interval) * id_price * _cfg(config, ["economics", "imbalance_penalty_multiplier"], 1.0)
    interval_cost = interval_cost * quarter_mask

    interval_threshold = torch.quantile(interval_cost, q=_cfg(config, ["reward", "cvar_alpha"], 0.99), dim=-1, keepdim=True)
    interval_tail = torch.where(interval_cost >= interval_threshold, interval_cost, torch.zeros_like(interval_cost))
    interval_tail_count = torch.clamp((interval_cost >= interval_threshold).sum(dim=-1), min=1)
    cvar99 = interval_tail.sum(dim=-1) / interval_tail_count

    procurement_cost_weekly = interval_cost.sum(dim=-1)
    retail_revenue = actual_load.view(1, 1, week_count) * _cfg(config, ["economics", "retail_tariff_yuan_per_mwh"], 430.0)
    adjustment_cost = torch.abs(contract_adjustment_mwh_raw - contract_adjustment_mwh_exec) * _cfg(config, ["economics", "adjustment_cost_yuan_per_mwh"], 0.6)
    friction_cost = spot_hedge_mwh.abs().sum(dim=-1) * _cfg(config, ["economics", "friction_cost_yuan_per_mwh"], 1.2)
    imbalance_cost = imbalance_energy * avg_id_price
    policy_violation_penalty = torch.abs(contract_adjustment_mwh_raw - contract_adjustment_mwh_exec) * _cfg(config, ["policy_projection", "violation_penalty_scale"], 1.0)

    # Dynamic-lock baseline remains the reference strategy for reward.
    baseline_position = 0.55 * forecast_load.view(1, 1, week_count) * (1.0 - 0.05 * policy_projection_active)
    baseline_interval = (baseline_position.unsqueeze(-1) / valid_intervals.unsqueeze(-1)) * quarter_mask
    baseline_interval_cost = baseline_interval * (0.6 * lt_interval_price + 0.4 * da_price)
    baseline_interval_cost = baseline_interval_cost + torch.abs(actual_interval - baseline_interval) * id_price
    baseline_procurement_cost = baseline_interval_cost.sum(dim=-1)
    profit = retail_revenue - procurement_cost_weekly - adjustment_cost - friction_cost
    profit_baseline = retail_revenue - baseline_procurement_cost
    excess_profit = profit - profit_baseline
    hedge_error = imbalance_energy / actual_load.view(1, 1, week_count).clamp_min(1.0)
    reward = (
        excess_profit
        - _cfg(config, ["reward", "lambda_tail"], 0.65) * cvar99
        - _cfg(config, ["reward", "lambda_hedge"], 0.18) * hedge_error
        - _cfg(config, ["reward", "lambda_trade"], 0.10) * friction_cost
        - _cfg(config, ["reward", "lambda_violate"], 1.0) * policy_violation_penalty
    )

    procurement_cost = procurement_cost_weekly.sum(dim=-1)
    total_score = -(reward.sum(dim=-1))

    return ParticleScoreResult(
        total_score=total_score,
        procurement_cost=procurement_cost,
        profit=profit.sum(dim=-1),
        reward=reward.sum(dim=-1),
        cvar99=cvar99.mean(dim=-1),
        hedge_error=hedge_error.mean(dim=-1),
        retail_revenue=retail_revenue.sum(dim=-1),
        imbalance_cost=imbalance_cost.sum(dim=-1),
        adjustment_cost=adjustment_cost.sum(dim=-1),
        friction_cost=friction_cost.sum(dim=-1),
        policy_violation_penalty=policy_violation_penalty.sum(dim=-1),
        profit_baseline=profit_baseline.sum(dim=-1),
        excess_profit=excess_profit.sum(dim=-1),
        contract_adjustment_mwh_raw=contract_adjustment_mwh_raw,
        contract_adjustment_mwh_exec=contract_adjustment_mwh_exec,
        contract_position_mwh=contract_position_mwh,
        exposure_band_mwh=exposure_band_mwh,
        policy_projection_active=policy_projection_active.expand(upper_count, lower_count, week_count),
        contract_curve=contract_curve,
        spot_hedge_mwh=spot_hedge_mwh,
        spot_hedge_limit_mwh=spot_hedge_limit_mwh,
        weekly_procurement_cost=procurement_cost_weekly,
        weekly_profit=profit,
        weekly_reward=reward,
        weekly_cvar99=cvar99,
        weekly_hedge_error=hedge_error,
        weekly_profit_baseline=profit_baseline,
        weekly_excess_profit=excess_profit,
        weekly_retail_revenue=retail_revenue,
        weekly_imbalance_cost=imbalance_cost,
        weekly_adjustment_cost=adjustment_cost,
        weekly_friction_cost=friction_cost,
        weekly_policy_violation_penalty=policy_violation_penalty,
    )
