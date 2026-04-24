from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.model_layout.schema import CompiledParameterLayout, ParameterBlockSpec
from src.policy.projection import project_hourly_hedge_tensor
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
    exposure_band_mwh_raw: torch.Tensor
    contract_position_mwh: torch.Tensor
    exposure_band_mwh: torch.Tensor
    policy_projection_active: torch.Tensor
    feasible_domain_clip_gap: torch.Tensor
    feasible_domain_clip_active: torch.Tensor
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


def _score_cfg(config: dict[str, Any] | None, key: str, default: float) -> float:
    return _cfg(config, ["score_kernel", key], default)


def _score_nested_cfg(config: dict[str, Any] | None, group: str, key: str, default: float) -> float:
    return _cfg(config, ["score_kernel", group, key], default)


def _find_block(layout_blocks: list[ParameterBlockSpec], block_name: str) -> ParameterBlockSpec:
    for block in layout_blocks:
        if block.name == block_name:
            return block
    raise KeyError(block_name)


def _block_slice(particles: torch.Tensor, block: ParameterBlockSpec) -> torch.Tensor:
    return particles[:, block.slice_start:block.slice_end]


def _column_indices(available_columns: list[str], required_columns: list[str]) -> list[int]:
    lookup = {column: idx for idx, column in enumerate(available_columns)}
    missing = [column for column in required_columns if column not in lookup]
    if missing:
        raise KeyError(", ".join(missing))
    return [lookup[column] for column in required_columns]


def _curve_basis(curve_dim: int, device: str) -> torch.Tensor:
    hours = torch.linspace(0.0, 1.0, steps=24, device=device, dtype=torch.float32)
    components: list[torch.Tensor] = []
    for idx in range(curve_dim):
        if idx == 0:
            components.append(torch.ones_like(hours))
        elif idx % 2 == 1:
            components.append(torch.cos(torch.pi * (idx // 2 + 1) * hours))
        else:
            components.append(torch.sin(torch.pi * (idx // 2) * hours))
    return torch.stack(components[:curve_dim], dim=0)


def batch_score_particles(
    tensor_bundle: TrainingTensorBundle,
    upper_particles: torch.Tensor,
    lower_particles: torch.Tensor,
    device: str = "cpu",
    config: dict[str, Any] | None = None,
    compiled_layout: CompiledParameterLayout | None = None,
) -> ParticleScoreResult:
    score_device = _resolve_device(device)
    weekly_features = torch.nan_to_num(tensor_bundle.weekly_feature_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    policy_tensor = torch.nan_to_num(tensor_bundle.policy_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    hourly_tensor = torch.nan_to_num(tensor_bundle.hourly_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    weekly_bounds = torch.nan_to_num(tensor_bundle.weekly_bound_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    hourly_bounds = torch.nan_to_num(tensor_bundle.hourly_bound_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    forecast_load = torch.nan_to_num(tensor_bundle.forecast_weekly_load.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    actual_load = torch.nan_to_num(tensor_bundle.actual_weekly_load.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    lt_price = torch.nan_to_num(tensor_bundle.lt_weekly_price.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    price_tensor = torch.nan_to_num(tensor_bundle.quarter_price_tensor.to(score_device), nan=0.0, posinf=0.0, neginf=0.0)
    hour_mask = tensor_bundle.hourly_valid_mask.to(score_device).unsqueeze(0).unsqueeze(0).float()
    quarter_mask = tensor_bundle.quarter_valid_mask.to(score_device).unsqueeze(0).unsqueeze(0).float()

    upper = upper_particles.to(score_device)
    lower = lower_particles.to(score_device)
    upper_count, lower_count = upper.shape[0], lower.shape[0]
    week_count, hour_count = weekly_features.shape[0], hourly_tensor.shape[1]
    interval_count = price_tensor.shape[1]

    if compiled_layout is None:
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
        upper_curve_seed = upper[:, -4:] if upper.shape[1] >= 4 else torch.nn.functional.pad(upper, (max(0, 4 - upper.shape[1]), 0))
        curve_base = torch.linspace(-1.0, 1.0, steps=24, device=score_device, dtype=torch.float32)
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
        lower_groups = lower.view(lower_count, 4, -1).mean(dim=-1)
        spread_coef = lower_groups[:, 0].view(1, lower_count, 1, 1)
        load_coef = lower_groups[:, 1].view(1, lower_count, 1, 1)
        renewable_coef = lower_groups[:, 2].view(1, lower_count, 1, 1)
        policy_shrink = torch.sigmoid(lower_groups[:, 3]).view(1, lower_count, 1, 1)
        action_head = None
    else:
        weekly_block = _find_block(compiled_layout.upper.blocks, "weekly_feature_weights")
        weekly_indices = _column_indices(tensor_bundle.weekly_feature_columns, weekly_block.columns)
        weekly_feature_matrix = weekly_features[:, weekly_indices]
        weekly_weight_block = _block_slice(upper, weekly_block)
        feature_signal = weekly_feature_matrix @ weekly_weight_block.T
        feature_signal = feature_signal.transpose(0, 1).unsqueeze(1).expand(upper_count, lower_count, week_count)

        policy_block = _find_block(compiled_layout.upper.blocks, "policy_feature_weights")
        policy_indices = _column_indices(tensor_bundle.policy_columns, policy_block.columns)
        policy_feature_matrix = policy_tensor[:, policy_indices]
        policy_weight_block = _block_slice(upper, policy_block)
        policy_signal = policy_feature_matrix @ policy_weight_block.T
        policy_signal = policy_signal.transpose(0, 1).unsqueeze(1).expand(upper_count, lower_count, week_count)
        policy_projection_active = (policy_feature_matrix.abs().mean(dim=1) > 0).float().view(1, 1, week_count)

        curve_block = _find_block(compiled_layout.upper.blocks, "contract_curve_latent")
        upper_curve_seed = _block_slice(upper, curve_block)
        curve_components = _curve_basis(upper_curve_seed.shape[1], score_device)
        contract_curve_logits = torch.einsum("uc,ch->uh", upper_curve_seed, curve_components)

        spread_block = _block_slice(lower, _find_block(compiled_layout.lower.blocks, "spread_response"))
        load_block = _block_slice(lower, _find_block(compiled_layout.lower.blocks, "load_deviation_response"))
        renewable_block = _block_slice(lower, _find_block(compiled_layout.lower.blocks, "renewable_response"))
        shrink_block = _block_slice(lower, _find_block(compiled_layout.lower.blocks, "policy_shrink_response"))
        spread_coef = spread_block.mean(dim=1).view(1, lower_count, 1, 1)
        load_coef = load_block.mean(dim=1).view(1, lower_count, 1, 1)
        renewable_coef = renewable_block.mean(dim=1).view(1, lower_count, 1, 1)
        policy_shrink = torch.sigmoid(shrink_block.mean(dim=1)).view(1, lower_count, 1, 1)
        action_head = _block_slice(upper, _find_block(compiled_layout.upper.blocks, "action_head"))

    weekly_bound_lookup = {name: idx for idx, name in enumerate(tensor_bundle.weekly_bound_columns)}
    contract_adjustment_ratio_min = weekly_bounds[:, weekly_bound_lookup["contract_adjustment_ratio_min"]].view(1, 1, week_count)
    contract_adjustment_ratio_max = weekly_bounds[:, weekly_bound_lookup["contract_adjustment_ratio_max"]].view(1, 1, week_count)
    exposure_band_ratio_min = weekly_bounds[:, weekly_bound_lookup["exposure_band_ratio_min"]].view(1, 1, week_count)
    exposure_band_ratio_max = weekly_bounds[:, weekly_bound_lookup["exposure_band_ratio_max"]].view(1, 1, week_count)
    non_negative_position_required = weekly_bounds[:, weekly_bound_lookup["non_negative_position_required"]].view(1, 1, week_count)
    feasible_domain_triggered = weekly_bounds[:, weekly_bound_lookup["feasible_domain_triggered"]].view(1, 1, week_count)

    if action_head is not None and action_head.shape[1] > 0:
        contract_bias = action_head[:, 0].view(upper_count, 1, 1)
        exposure_bias = action_head[:, 1].view(upper_count, 1, 1) if action_head.shape[1] > 1 else 0.0
    else:
        contract_bias = 0.0
        exposure_bias = 0.0
    contract_adjustment_mwh_raw = (
        _score_cfg(config, "contract_adjustment_scale_ratio", 0.30)
        * torch.tanh(
            _score_cfg(config, "contract_adjustment_feature_scale", 0.15) * feature_signal
            + _score_cfg(config, "contract_adjustment_policy_scale", 0.05) * policy_signal
            + contract_bias
        )
        * forecast_load.view(1, 1, week_count)
    )
    contract_adjustment_mwh_exec = torch.clamp(
        contract_adjustment_mwh_raw,
        min=contract_adjustment_ratio_min * forecast_load.view(1, 1, week_count),
        max=contract_adjustment_ratio_max * forecast_load.view(1, 1, week_count),
    )
    exposure_band_mwh_raw = torch.relu(
        _score_cfg(config, "exposure_band_base_ratio", 0.20)
        * forecast_load.view(1, 1, week_count)
        * (1.0 + torch.tanh(_score_cfg(config, "exposure_band_feature_scale", 0.10) * feature_signal + exposure_bias))
    )
    exposure_band_mwh = torch.clamp(
        exposure_band_mwh_raw,
        min=exposure_band_ratio_min * forecast_load.view(1, 1, week_count),
        max=exposure_band_ratio_max * forecast_load.view(1, 1, week_count),
    )
    contract_position_raw = _score_cfg(config, "contract_position_base_ratio", 0.60) * forecast_load.view(1, 1, week_count) + contract_adjustment_mwh_exec
    contract_position_mwh = torch.where(
        non_negative_position_required > 0.5,
        torch.clamp_min(contract_position_raw, 0.0),
        contract_position_raw,
    )

    contract_curve = torch.softmax(contract_curve_logits, dim=-1).unsqueeze(1).unsqueeze(2).expand(upper_count, lower_count, week_count, 24)

    hourly_feature_lookup = {name: idx for idx, name in enumerate(tensor_bundle.hourly_feature_columns)}

    def _hourly_feature(column: str, default: float = 0.0) -> torch.Tensor:
        column_idx = hourly_feature_lookup.get(column)
        if column_idx is None:
            return torch.full((1, 1, week_count, hour_count), float(default), dtype=torch.float32, device=score_device)
        return hourly_tensor[..., column_idx].view(1, 1, week_count, hour_count)

    spread = _hourly_feature("price_spread")
    spread_abs = _hourly_feature("price_spread_abs")
    load_dev = _hourly_feature("load_dev")
    renewable_dev = _hourly_feature("renewable_dev")
    renewable_dev_abs = _hourly_feature("renewable_dev_abs")
    business_hour_flag = _hourly_feature("business_hour_flag")
    peak_hour_flag = _hourly_feature("peak_hour_flag")
    valley_hour_flag = _hourly_feature("valley_hour_flag")
    settlement_effective_flag = torch.clamp_min(_hourly_feature("settlement_effective_flag", default=1.0), 0.0)
    hourly_bound_lookup = {name: idx for idx, name in enumerate(tensor_bundle.hourly_bound_columns)}
    hourly_share_cap = hourly_bounds[..., hourly_bound_lookup["max_hourly_hedge_share"]].view(1, 1, week_count, hour_count)
    hourly_ramp_share = hourly_bounds[..., hourly_bound_lookup["max_hourly_ramp_share"]].view(1, 1, week_count, hour_count)
    raw_spot_hedge_limit_mwh = (exposure_band_mwh_raw.unsqueeze(-1) / max(hour_count, 1)) * (
        _score_nested_cfg(config, "hourly_limit", "base_multiplier", 0.50)
        + _score_nested_cfg(config, "hourly_limit", "shrink_multiplier", 0.50) * policy_shrink
    )
    raw_spot_hedge_limit_mwh = raw_spot_hedge_limit_mwh.expand(upper_count, lower_count, week_count, hour_count)
    session_signal = (
        _score_nested_cfg(config, "session_weights", "business_hour", 0.50) * business_hour_flag
        + _score_nested_cfg(config, "session_weights", "peak_hour", 0.30) * peak_hour_flag
        + _score_nested_cfg(config, "session_weights", "valley_hour", -0.20) * valley_hour_flag
    )
    raw_hourly_signal = (
        _score_nested_cfg(config, "hourly_signal", "spread_weight", 0.02) * spread_coef * spread
        + _score_nested_cfg(config, "hourly_signal", "load_dev_weight", 0.01) * load_coef * load_dev
        + _score_nested_cfg(config, "hourly_signal", "renewable_weight", 0.01) * renewable_coef * renewable_dev
    )
    raw_hourly_signal = raw_hourly_signal + _score_nested_cfg(config, "hourly_signal", "spread_abs_weight", 0.005) * spread_coef * spread_abs * session_signal
    raw_hourly_signal = raw_hourly_signal + _score_nested_cfg(config, "hourly_signal", "renewable_abs_weight", 0.004) * renewable_coef * renewable_dev_abs * (
        _score_nested_cfg(config, "session_weights", "renewable_valley_mix", 0.50) * valley_hour_flag
        + _score_nested_cfg(config, "session_weights", "renewable_business_mix", 0.50) * business_hour_flag
    )
    raw_hourly_signal = raw_hourly_signal * torch.where(
        settlement_effective_flag > 0.0,
        torch.ones_like(settlement_effective_flag),
        torch.zeros_like(settlement_effective_flag),
    )
    gate_cfg = config.get("score_kernel", {}).get("hourly_gate", {}) if isinstance(config, dict) else {}
    if bool(gate_cfg.get("enabled", False)):
        signal_deadband = float(gate_cfg.get("signal_deadband", 0.0))
        gate_temperature = max(float(gate_cfg.get("temperature", 0.05)), 1.0e-6)
        signal_edge = raw_hourly_signal.abs()
        soft_gate = torch.sigmoid((signal_edge - signal_deadband) / gate_temperature)
        hard_gate = torch.where(signal_edge >= signal_deadband, soft_gate, torch.zeros_like(soft_gate))
    else:
        hard_gate = torch.ones_like(raw_hourly_signal)
    raw_spot_hedge_mwh = hard_gate * torch.tanh(raw_hourly_signal) * raw_spot_hedge_limit_mwh * hour_mask
    spot_hedge_mwh, hourly_projection_gap = project_hourly_hedge_tensor(
        raw_spot_hedge_mwh,
        projected_exposure_band_mwh=exposure_band_mwh,
        hourly_share_cap=hourly_share_cap,
        hourly_ramp_share=hourly_ramp_share,
        hour_mask=hour_mask,
    )
    valid_hours = hour_mask.sum(dim=-1).clamp_min(1.0)
    spot_hedge_limit_mwh = (exposure_band_mwh.unsqueeze(-1) / valid_hours.unsqueeze(-1)) * hourly_share_cap

    avg_da_price = torch.nan_to_num(price_tensor[..., 0].mean(dim=1), nan=0.0).view(1, 1, week_count)
    avg_id_price = torch.nan_to_num(price_tensor[..., 1].mean(dim=1), nan=0.0).view(1, 1, week_count)
    valid_intervals = quarter_mask.sum(dim=-1).clamp_min(1.0)
    spot_energy_net = (spot_hedge_mwh * hour_mask).sum(dim=-1)
    spot_energy_abs = (spot_hedge_mwh.abs() * hour_mask).sum(dim=-1)
    scheduled_energy_raw = contract_position_mwh + spot_energy_net
    scheduled_energy = torch.clamp_min(scheduled_energy_raw, 0.0)
    imbalance_energy = torch.abs(actual_load.view(1, 1, week_count) - scheduled_energy)

    scheduled_interval = (scheduled_energy.unsqueeze(-1) / valid_intervals.unsqueeze(-1)) * quarter_mask
    actual_interval = (actual_load.view(1, 1, week_count, 1) / valid_intervals.unsqueeze(-1)) * quarter_mask
    da_price = price_tensor[..., 0].view(1, 1, week_count, interval_count)
    id_price = price_tensor[..., 1].view(1, 1, week_count, interval_count)
    lt_interval_price = lt_price.view(1, 1, week_count, 1)
    interval_cost = scheduled_interval * (
        _score_cfg(config, "lt_settlement_weight", 0.60) * lt_interval_price
        + _score_cfg(config, "da_settlement_weight", 0.40) * da_price
    )
    interval_cost = interval_cost + torch.abs(actual_interval - scheduled_interval) * id_price * _cfg(config, ["economics", "imbalance_penalty_multiplier"], 1.0)
    interval_cost = interval_cost * quarter_mask

    interval_threshold = torch.quantile(interval_cost, q=_cfg(config, ["reward", "cvar_alpha"], 0.99), dim=-1, keepdim=True)
    interval_tail = torch.where(interval_cost >= interval_threshold, interval_cost, torch.zeros_like(interval_cost))
    interval_tail_count = torch.clamp((interval_cost >= interval_threshold).sum(dim=-1), min=1)
    cvar99 = interval_tail.sum(dim=-1) / interval_tail_count

    procurement_cost_weekly = interval_cost.sum(dim=-1)
    retail_revenue = actual_load.view(1, 1, week_count) * _cfg(config, ["economics", "retail_tariff_yuan_per_mwh"], 430.0)
    adjustment_cost = torch.abs(contract_adjustment_mwh_raw - contract_adjustment_mwh_exec) * _cfg(config, ["economics", "adjustment_cost_yuan_per_mwh"], 0.6)
    friction_cost = spot_energy_abs * _cfg(config, ["economics", "friction_cost_yuan_per_mwh"], 1.2)
    imbalance_cost = imbalance_energy * avg_id_price
    feasible_domain_clip_gap = (
        torch.abs(contract_adjustment_mwh_raw - contract_adjustment_mwh_exec)
        + torch.abs(exposure_band_mwh_raw - exposure_band_mwh)
        + hourly_projection_gap
    )
    feasible_domain_clip_active = (feasible_domain_clip_gap > 1.0e-6).float()
    policy_projection_active = torch.maximum(policy_projection_active.expand(upper_count, lower_count, week_count), feasible_domain_triggered.expand(upper_count, lower_count, week_count))
    policy_violation_penalty = feasible_domain_clip_gap * _cfg(config, ["policy_projection", "violation_penalty_scale"], 1.0)

    profit = retail_revenue - procurement_cost_weekly - adjustment_cost - friction_cost
    reward_cfg = config.get("reward", {}) if isinstance(config, dict) else {}
    baseline_ratios = reward_cfg.get("baseline_position_ratios")
    if not baseline_ratios:
        baseline_ratios = [_score_cfg(config, "baseline_position_ratio", 0.55)]
    baseline_ratio_tensor = torch.as_tensor(
        [float(value) for value in baseline_ratios],
        dtype=torch.float32,
        device=score_device,
    ).view(-1, 1, 1, 1)
    baseline_projection_multiplier = 1.0 - _score_cfg(config, "baseline_projection_penalty_scale", 0.05) * policy_projection_active
    baseline_positions = (
        baseline_ratio_tensor
        * forecast_load.view(1, 1, 1, week_count)
        * baseline_projection_multiplier.unsqueeze(0)
    )
    baseline_interval = (baseline_positions.unsqueeze(-1) / valid_intervals.view(1, 1, 1, week_count, 1)) * quarter_mask.view(1, 1, 1, week_count, interval_count)
    baseline_interval_cost = baseline_interval * (
        _score_cfg(config, "lt_settlement_weight", 0.60) * lt_interval_price.view(1, 1, 1, week_count, 1)
        + _score_cfg(config, "da_settlement_weight", 0.40) * da_price.view(1, 1, 1, week_count, interval_count)
    )
    baseline_actual_interval = actual_interval.unsqueeze(0)
    baseline_interval_cost = baseline_interval_cost + torch.abs(baseline_actual_interval - baseline_interval) * id_price.view(1, 1, 1, week_count, interval_count)
    baseline_procurement_candidates = baseline_interval_cost.sum(dim=-1)
    baseline_procurement_cost = baseline_procurement_candidates.min(dim=0).values
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
        exposure_band_mwh_raw=exposure_band_mwh_raw,
        contract_position_mwh=contract_position_mwh,
        exposure_band_mwh=exposure_band_mwh,
        policy_projection_active=policy_projection_active,
        feasible_domain_clip_gap=feasible_domain_clip_gap,
        feasible_domain_clip_active=feasible_domain_clip_active,
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
