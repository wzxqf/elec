from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from src.backtest.benchmarks import get_dynamic_lock_base_for_week
from src.backtest.metrics import summarize_strategy_results
from src.backtest.runtime_cache import prepare_runtime_bundle
from src.backtest.settlement import resolve_settlement_context, settle_week
from src.backtest.simulator import _allocate_weekly_lt_to_hourly, _get_week_frames, _week_risk_term


ObjectiveFn = Callable[[torch.Tensor], torch.Tensor]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class HPSOSettings:
    particles: int
    iterations: int
    inertia_weight: float
    cognitive_factor: float
    social_factor: float
    initial_temperature: float
    cooling_rate: float
    perturbation_scale: float
    stagnation_window: int
    seed: int
    device: str
    allow_cpu: bool
    backprop_steps: int = 0
    backprop_learning_rate: float = 0.0
    backprop_clip_norm: float = 1.0


@dataclass
class HPSOModel:
    device: str
    config: dict[str, Any]


def _settings_from_config(config: dict[str, Any], scope: str, seed_offset: int = 0) -> HPSOSettings:
    hpso_cfg = config["hpso"]
    scope_cfg = hpso_cfg[scope]
    requested_device = str(hpso_cfg.get("device", config.get("device", "cpu")))
    allow_cpu = bool(hpso_cfg.get("allow_cpu", True))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("HPSO 配置要求 CUDA，但当前环境不可用，且 allow_cpu=false。")
        device = "cpu"
    else:
        device = requested_device
    return HPSOSettings(
        particles=int(scope_cfg["particles"]),
        iterations=int(scope_cfg["iterations"]),
        inertia_weight=float(scope_cfg["inertia_weight"]),
        cognitive_factor=float(scope_cfg["cognitive_factor"]),
        social_factor=float(scope_cfg["social_factor"]),
        initial_temperature=float(scope_cfg["initial_temperature"]),
        cooling_rate=float(scope_cfg["cooling_rate"]),
        perturbation_scale=float(scope_cfg["perturbation_scale"]),
        stagnation_window=int(scope_cfg["stagnation_window"]),
        seed=int(hpso_cfg.get("seed", config.get("seed", 42))) + int(seed_offset),
        device=device,
        allow_cpu=allow_cpu,
        backprop_steps=int(scope_cfg.get("backprop_steps", hpso_cfg.get("backprop_steps", 0))),
        backprop_learning_rate=float(scope_cfg.get("backprop_learning_rate", hpso_cfg.get("backprop_learning_rate", 0.0))),
        backprop_clip_norm=float(scope_cfg.get("backprop_clip_norm", hpso_cfg.get("backprop_clip_norm", 1.0))),
    )


class HybridParticleSwarmOptimizer:
    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        settings: HPSOSettings,
        objective: ObjectiveFn,
        bounded_mask: torch.Tensor | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if lower.shape != upper.shape:
            raise ValueError("HPSO lower/upper bounds must have the same shape.")
        self.lower = lower.to(settings.device, dtype=torch.float64)
        self.upper = upper.to(settings.device, dtype=torch.float64)
        if bounded_mask is None:
            self.bounded_mask = torch.ones_like(self.lower, dtype=torch.bool, device=settings.device)
        else:
            if bounded_mask.shape != lower.shape:
                raise ValueError("HPSO bounded_mask must have the same shape as lower/upper bounds.")
            self.bounded_mask = bounded_mask.to(settings.device, dtype=torch.bool)
        self.settings = settings
        self.objective = objective
        self.progress_callback = progress_callback

    def _emit_progress(
        self,
        iteration: int,
        total_iterations: int,
        best_score: float,
        temperature: float,
        stagnated: bool,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            {
                "iteration": int(iteration),
                "total_iterations": int(total_iterations),
                "best_score": float(best_score),
                "temperature": float(temperature),
                "stagnated": bool(stagnated),
            }
        )

    def _clamp_positions(self, positions: torch.Tensor) -> torch.Tensor:
        clipped = torch.minimum(torch.maximum(positions, self.lower), self.upper)
        return torch.where(self.bounded_mask.unsqueeze(0), clipped, positions)

    def _backprop_refine(self, positions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
        settings = self.settings
        if settings.backprop_steps <= 0 or settings.backprop_learning_rate <= 0.0:
            return positions, {
                "bp_steps": 0,
                "bp_loss_before": float("nan"),
                "bp_loss_after": float("nan"),
                "bp_grad_norm_mean": 0.0,
                "bp_improved_particles": 0,
            }

        candidate = positions.detach()
        before_scores = self.objective(candidate).detach()
        grad_norms: list[float] = []
        clip_norm = max(float(settings.backprop_clip_norm), 1e-12)

        for _ in range(settings.backprop_steps):
            candidate = candidate.detach().clone().requires_grad_(True)
            scores = self.objective(candidate)
            scores.sum().backward()
            gradient = candidate.grad
            if gradient is None:
                break
            grad_norm = torch.linalg.vector_norm(gradient.detach(), dim=1, ord=2)
            grad_norms.append(float(grad_norm.mean().item()))
            scale = torch.clamp(clip_norm / torch.clamp(grad_norm, min=1e-12), max=1.0).unsqueeze(1)
            candidate = candidate.detach() - float(settings.backprop_learning_rate) * gradient.detach() * scale
            candidate = self._clamp_positions(candidate)

        refined = candidate.detach()
        after_scores = self.objective(refined).detach()
        improved = after_scores < before_scores
        keep_refined = improved.unsqueeze(1)
        refined_positions = torch.where(keep_refined, refined, positions.detach())
        final_scores = torch.where(improved, after_scores, before_scores)
        return refined_positions, {
            "bp_steps": int(settings.backprop_steps),
            "bp_loss_before": float(before_scores.mean().item()),
            "bp_loss_after": float(final_scores.mean().item()),
            "bp_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "bp_improved_particles": int(improved.sum().item()),
        }

    def optimize(self) -> tuple[torch.Tensor, float, pd.DataFrame]:
        settings = self.settings
        generator = torch.Generator(device=settings.device)
        generator.manual_seed(settings.seed)
        span = torch.clamp(self.upper - self.lower, min=1e-12)
        shape = (settings.particles, int(self.lower.numel()))
        positions = self.lower + torch.rand(shape, generator=generator, device=settings.device, dtype=torch.float64) * span
        velocities = torch.zeros_like(positions)
        positions, bp_stats = self._backprop_refine(positions)

        scores = self.objective(positions)
        pbest_positions = positions.clone()
        pbest_scores = scores.clone()
        best_index = int(torch.argmin(scores).item())
        gbest_position = positions[best_index].clone()
        gbest_score = float(scores[best_index].item())
        last_improved_iteration = 0
        records = [
            {
                "iteration": 0,
                "best_score": gbest_score,
                "temperature": float(settings.initial_temperature),
                "stagnated": False,
                **bp_stats,
            }
        ]
        self._emit_progress(
            iteration=0,
            total_iterations=settings.iterations,
            best_score=gbest_score,
            temperature=float(settings.initial_temperature),
            stagnated=False,
        )

        for iteration in range(1, settings.iterations + 1):
            r1 = torch.rand(shape, generator=generator, device=settings.device, dtype=torch.float64)
            r2 = torch.rand(shape, generator=generator, device=settings.device, dtype=torch.float64)
            velocities = (
                settings.inertia_weight * velocities
                + settings.cognitive_factor * r1 * (pbest_positions - positions)
                + settings.social_factor * r2 * (gbest_position.unsqueeze(0) - positions)
            )
            positions = positions + velocities
            temperature = settings.initial_temperature * (settings.cooling_rate ** iteration)
            stagnated = iteration - last_improved_iteration >= settings.stagnation_window
            if settings.perturbation_scale > 0.0:
                noise = torch.randn(shape, generator=generator, device=settings.device, dtype=torch.float64)
                trigger = 1.0 if stagnated else 0.35
                positions = positions + noise * span * settings.perturbation_scale * temperature * trigger
            positions = self._clamp_positions(positions)
            positions, bp_stats = self._backprop_refine(positions)

            scores = self.objective(positions)
            improved = scores < pbest_scores
            pbest_positions = torch.where(improved.unsqueeze(1), positions, pbest_positions)
            pbest_scores = torch.where(improved, scores, pbest_scores)
            best_index = int(torch.argmin(pbest_scores).item())
            candidate_score = float(pbest_scores[best_index].item())
            if candidate_score < gbest_score - 1e-12:
                gbest_score = candidate_score
                gbest_position = pbest_positions[best_index].clone()
                last_improved_iteration = iteration

            records.append(
                {
                    "iteration": iteration,
                    "best_score": gbest_score,
                    "temperature": float(temperature),
                    "stagnated": bool(stagnated),
                    **bp_stats,
                }
            )
            self._emit_progress(
                iteration=iteration,
                total_iterations=settings.iterations,
                best_score=gbest_score,
                temperature=float(temperature),
                stagnated=bool(stagnated),
            )

        return gbest_position.detach().cpu(), gbest_score, pd.DataFrame(records)


def _to_tensor(values: list[float] | np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.array(values, dtype=float, copy=True), dtype=torch.float64, device=device)


def _apply_weekly_lock_constraints(
    raw_locks: torch.Tensor,
    config: dict[str, Any],
    previous_lock_ratio: float = 0.0,
) -> torch.Tensor:
    constraints = config["constraints"]
    min_lock = float(constraints["lock_ratio_min"])
    max_lock = float(constraints["lock_ratio_max"])
    max_step = float(constraints["delta_h_max"])
    constrained = []
    previous = torch.full((raw_locks.shape[0],), float(previous_lock_ratio), dtype=raw_locks.dtype, device=raw_locks.device)
    for index in range(raw_locks.shape[1]):
        target = torch.clamp(raw_locks[:, index], min_lock, max_lock)
        final = torch.clamp(target, previous - max_step, previous + max_step)
        final = torch.clamp(final, min_lock, max_lock)
        constrained.append(final)
        previous = final
    return torch.stack(constrained, dim=1)


def _build_weekly_locks_from_residual_actions(
    delta_lock_ratio_raw: torch.Tensor,
    base_locks: torch.Tensor,
    config: dict[str, Any],
    previous_lock_ratio: float = 0.0,
) -> torch.Tensor:
    return base_locks + delta_lock_ratio_raw


def _upper_objective_components(
    delta_lock_ratio_raw: torch.Tensor,
    bandwidth: torch.Tensor,
    forecast_t: torch.Tensor,
    da_t: torch.Tensor,
    lt_t: torch.Tensor,
    spread_t: torch.Tensor,
    renewable_t: torch.Tensor,
    base_t: torch.Tensor,
    config: dict[str, Any],
    previous_lock_ratio: float,
) -> dict[str, torch.Tensor]:
    locks = _build_weekly_locks_from_residual_actions(
        delta_lock_ratio_raw,
        base_t,
        config,
        previous_lock_ratio=previous_lock_ratio,
    )
    q_lt = locks * forecast_t
    q_spot = forecast_t - q_lt
    lock_change = torch.abs(torch.diff(locks, dim=1, prepend=torch.full_like(locks[:, :1], previous_lock_ratio)))
    base_gap = torch.abs(locks - base_t)
    contract_quadratic = float(config["hpso"]["objective_weights"].get("contract_adjustment_quadratic", 5.0))
    return {
        "procurement_cost": (q_lt * lt_t + q_spot * da_t).sum(dim=1),
        "price_risk": (torch.abs(q_spot) * spread_t).sum(dim=1),
        "renewable_risk": (torch.abs(q_spot) * renewable_t * 100.0).sum(dim=1),
        "weekly_lock_change": lock_change.sum(dim=1) * forecast_t.mean(),
        "baseline_gap": base_gap.sum(dim=1) * forecast_t.mean(),
        "bandwidth_cost": (bandwidth * forecast_t).sum(dim=1),
        "contract_adjustment_cost": (
            torch.abs(delta_lock_ratio_raw).sum(dim=1) + contract_quadratic * (delta_lock_ratio_raw**2).sum(dim=1)
        )
        * forecast_t.mean(),
    }


def _score_upper_objective(components: dict[str, torch.Tensor], weights: dict[str, Any]) -> torch.Tensor:
    return (
        components["procurement_cost"]
        + float(weights["risk"]) * (components["price_risk"] + components["renewable_risk"])
        + float(weights["weekly_lock_change"]) * components["weekly_lock_change"]
        + float(weights["hedge_error"]) * components["baseline_gap"]
        + float(weights["transaction"]) * (components["bandwidth_cost"] + components["contract_adjustment_cost"])
    )


def optimize_upper_actions(
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    config: dict[str, Any],
    previous_lock_ratio: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = [pd.Timestamp(week) for week in weeks]
    settings = _settings_from_config(config, "upper")
    metadata_index = bundle["weekly_metadata"].set_index("week_start").sort_index()
    feature_lookup = bundle.get("weekly_feature_by_week")
    forecast = []
    da_price = []
    lt_price = []
    spread = []
    renewable = []
    dynamic_base = []
    for week in weeks:
        row = metadata_index.loc[week]
        forecast.append(float(row["forecast_weekly_net_demand_mwh"]))
        da_price.append(float(row["da_price_mean"]))
        lt_price.append(float(row["lt_price_w"]))
        if isinstance(feature_lookup, dict) and week in feature_lookup:
            feature_row = feature_lookup[week]
        else:
            feature_row = bundle["weekly_features"].set_index("week_start").loc[week]
        spread.append(abs(float(feature_row.get("prev_spread_mean", 0.0))))
        renewable.append(abs(float(feature_row.get("prev_renewable_ratio_da_mean", 0.0))))
        dynamic_base.append(get_dynamic_lock_base_for_week(bundle["weekly_features"], week, config, feature_lookup))

    n_weeks = len(weeks)
    contract_initial_radius = float(config["hpso"]["upper"].get("contract_adjustment_initial_radius", 1.0))
    lower = torch.tensor(
        [-contract_initial_radius] * n_weeks
        + [float(config["hpso"]["upper"]["exposure_bandwidth_min"])] * n_weeks,
        dtype=torch.float64,
    )
    upper = torch.tensor(
        [contract_initial_radius] * n_weeks
        + [float(config["hpso"]["upper"]["exposure_bandwidth_max"])] * n_weeks,
        dtype=torch.float64,
    )
    bounded_mask = torch.tensor([False] * n_weeks + [True] * n_weeks, dtype=torch.bool)

    forecast_t = _to_tensor(forecast, settings.device).unsqueeze(0)
    da_t = _to_tensor(da_price, settings.device).unsqueeze(0)
    lt_t = _to_tensor(lt_price, settings.device).unsqueeze(0)
    spread_t = _to_tensor(spread, settings.device).unsqueeze(0)
    renewable_t = _to_tensor(renewable, settings.device).unsqueeze(0)
    base_t = _to_tensor(dynamic_base, settings.device).unsqueeze(0)
    weights = config["hpso"]["objective_weights"]

    def objective(positions: torch.Tensor) -> torch.Tensor:
        components = _upper_objective_components(
            delta_lock_ratio_raw=positions[:, :n_weeks],
            bandwidth=positions[:, n_weeks:],
            forecast_t=forecast_t,
            da_t=da_t,
            lt_t=lt_t,
            spread_t=spread_t,
            renewable_t=renewable_t,
            base_t=base_t,
            config=config,
            previous_lock_ratio=previous_lock_ratio,
        )
        return _score_upper_objective(components, weights)

    optimizer = HybridParticleSwarmOptimizer(lower, upper, settings, objective, bounded_mask=bounded_mask)
    best, _, convergence = optimizer.optimize()
    best_batch = best.reshape(1, -1).to(settings.device)
    delta_raw = best_batch[:, :n_weeks][0]
    final_locks = _build_weekly_locks_from_residual_actions(
        best_batch[:, :n_weeks],
        base_t,
        config,
        previous_lock_ratio=previous_lock_ratio,
    )[0]
    bandwidths = best_batch[:, n_weeks:][0]
    rows = []
    for index, week in enumerate(weeks):
        final_lock = float(final_locks[index].detach().cpu().item())
        base_lock = float(dynamic_base[index])
        raw_action = float(delta_raw[index].detach().cpu().item())
        rows.append(
            {
                "week_start": week,
                "lock_ratio_base": base_lock,
                "delta_lock_ratio_raw": raw_action,
                "delta_lock_ratio": final_lock - base_lock,
                "lock_ratio_final": final_lock,
                "target_lock_ratio": final_lock,
                "exposure_bandwidth": float(bandwidths[index].detach().cpu().item()),
            }
        )
    convergence["scope"] = "upper"
    return pd.DataFrame(rows), convergence


def _signal(values: np.ndarray, threshold: float) -> np.ndarray:
    scaled = values / max(threshold, 1e-6)
    return np.clip(scaled, -1.0, 1.0)


def _compute_hourly_bandwidth(
    q_base: np.ndarray,
    exposure_bandwidth: float,
    policy_state: pd.Series,
    rules_config: dict[str, Any],
    composite_signal: np.ndarray,
) -> tuple[np.ndarray, float]:
    exposure = float(np.clip(exposure_bandwidth, 0.0, 1.0))
    ancillary_multiplier = 1.0
    if float(policy_state.get("ancillary_freq_reserve_tight", 0.0)) >= 0.5:
        ancillary_multiplier *= float(rules_config["ancillary_tight_multiplier"])
    if float(policy_state.get("ancillary_peak_shaving_pause", 0.0)) >= 0.5:
        ancillary_multiplier *= float(rules_config["peak_shaving_pause_multiplier"])
    volatility_multiplier = np.clip(1.0 - float(rules_config["price_spike_shrink"]) * np.abs(composite_signal), 0.1, 1.0)
    if exposure <= 0.0:
        return np.zeros_like(q_base, dtype=float), float(ancillary_multiplier)
    demand_scale = np.abs(q_base)
    base_band = demand_scale * float(rules_config["band_base_multiplier"]) * exposure
    bandwidth = np.clip(
        base_band * ancillary_multiplier * volatility_multiplier,
        float(rules_config["band_floor_mwh"]),
        demand_scale * float(rules_config["band_cap_ratio"]),
    )
    return np.clip(np.minimum(bandwidth, demand_scale), 0.0, None), float(ancillary_multiplier)


def _smooth_delta(delta_values: np.ndarray, smooth_limit: float) -> tuple[np.ndarray, int]:
    smoothed = np.empty_like(delta_values, dtype=float)
    previous = 0.0
    hits = 0
    for index, value in enumerate(delta_values):
        clipped = float(np.clip(value, previous - smooth_limit, previous + smooth_limit))
        if not np.isclose(clipped, value, atol=1e-9):
            hits += 1
        smoothed[index] = clipped
        previous = clipped
    return smoothed, hits


def _build_hourly_trace_from_delta(
    hourly_frame: pd.DataFrame,
    q_lt_hourly: pd.Series,
    exposure_bandwidth: float,
    policy_state: pd.Series,
    rules_config: dict[str, Any],
    raw_delta: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = hourly_frame.copy().reset_index(drop=True)
    q_lt = np.asarray(q_lt_hourly, dtype=float)
    net_load_da = frame["net_load_da"].to_numpy(dtype=float, copy=False)
    price_spread = frame["price_spread"].to_numpy(dtype=float, copy=False)
    load_dev = frame["load_dev"].to_numpy(dtype=float, copy=False)
    renewable_dev = frame["renewable_dev"].to_numpy(dtype=float, copy=False)
    q_base = net_load_da - q_lt

    spread_signal = _signal(price_spread, float(rules_config["spread_threshold"]))
    load_signal = _signal(load_dev, float(rules_config["load_dev_threshold"]))
    renewable_signal = _signal(-renewable_dev, float(rules_config["renewable_dev_threshold"]))
    composite = np.clip(
        float(rules_config["price_spread_weight"]) * spread_signal
        + float(rules_config["load_dev_weight"]) * load_signal
        + float(rules_config["renewable_dev_weight"]) * renewable_signal,
        -float(rules_config["signal_clip"]),
        float(rules_config["signal_clip"]),
    )
    bandwidth_mwh, ancillary_multiplier = _compute_hourly_bandwidth(q_base, exposure_bandwidth, policy_state, rules_config, composite)

    lower_band = q_base - bandwidth_mwh
    upper_band = q_base + bandwidth_mwh
    delta_after_smoothing = np.asarray(raw_delta, dtype=float)
    smooth_hits = 0
    q_spot_raw = q_base + delta_after_smoothing
    q_spot_band_clipped = q_spot_raw
    clipped_by_bound = np.zeros(len(frame), dtype=int)
    q_spot = q_spot_raw
    clipped_non_negative = np.zeros(len(frame), dtype=int)
    delta_q = q_spot - q_base
    hedge_error_abs = np.abs(q_spot - (frame["net_load_id"].to_numpy(dtype=float, copy=False) - q_lt))

    derived = pd.DataFrame(
        {
            "q_lt_hourly": q_lt,
            "spot_need": q_base,
            "q_base": q_base,
            "signal_spread": spread_signal,
            "signal_load_dev": load_signal,
            "signal_renewable_dev": renewable_signal,
            "signal_composite": composite,
            "bandwidth_mwh": bandwidth_mwh,
            "bandwidth_multiplier": np.full(len(frame), ancillary_multiplier, dtype=float),
            "delta_q_target": raw_delta,
            "delta_q_after_smoothing": delta_after_smoothing,
            "smoothing_mode": np.full(len(frame), "hpso_unbounded", dtype=object),
            "lower_band": lower_band,
            "upper_band": upper_band,
            "q_spot_raw": q_spot_raw,
            "q_spot_band_clipped": q_spot_band_clipped,
            "clipped_by_bound": clipped_by_bound,
            "q_spot": q_spot,
            "clipped_non_negative": clipped_non_negative,
            "delta_q": delta_q,
            "a_t_raw": np.divide(delta_q, np.maximum(bandwidth_mwh, 1e-9)),
            "a_t": np.divide(delta_q, np.maximum(bandwidth_mwh, 1e-9)),
            "soft_clipped": np.zeros(len(frame), dtype=int),
            "hedge_error_abs": hedge_error_abs,
        },
        index=frame.index,
    )
    frame = pd.concat([frame, derived], axis=1)
    stats = {
        "bound_clip_count": int(clipped_by_bound.sum()),
        "smooth_clip_count": int(smooth_hits),
        "soft_clip_count": 0,
        "non_negative_clip_count": int(clipped_non_negative.sum()),
    }
    return frame, stats


def _lower_objective_components(
    positions: torch.Tensor,
    q_base_t: torch.Tensor,
    da_t: torch.Tensor,
    spread_t: torch.Tensor,
    load_dev_t: torch.Tensor,
    renewable_t: torch.Tensor,
    net_target_t: torch.Tensor,
    spot_quadratic_weight: float,
) -> dict[str, torch.Tensor]:
    q_spot = q_base_t + positions
    return {
        "spot_cost": (q_spot * da_t).sum(dim=1),
        "risk": (torch.abs(positions) * (spread_t + 0.1 * load_dev_t + 0.05 * renewable_t)).sum(dim=1),
        "hourly_smooth": torch.abs(torch.diff(positions, dim=1, prepend=torch.zeros_like(positions[:, :1]))).sum(dim=1),
        "hedge_error": torch.abs(q_spot - net_target_t).sum(dim=1),
        "transaction": torch.abs(positions).sum(dim=1) + float(spot_quadratic_weight) * (positions**2).sum(dim=1),
    }


def _score_lower_objective(components: dict[str, torch.Tensor], weights: dict[str, Any]) -> torch.Tensor:
    return (
        components["spot_cost"]
        + float(weights["risk"]) * components["risk"]
        + float(weights["hourly_smooth"]) * components["hourly_smooth"]
        + float(weights["hedge_error"]) * components["hedge_error"]
        + float(weights["transaction"]) * components["transaction"]
    )


def optimize_hourly_delta(
    hourly_frame: pd.DataFrame,
    q_lt_hourly: pd.Series,
    exposure_bandwidth: float,
    policy_state: pd.Series,
    config: dict[str, Any],
    seed_offset: int,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    settings = _settings_from_config(config, "lower", seed_offset=seed_offset)
    rules_config = config["rules"]
    frame = hourly_frame.copy().reset_index(drop=True)
    q_lt = np.asarray(q_lt_hourly, dtype=float)
    q_base = frame["net_load_da"].to_numpy(dtype=float, copy=False) - q_lt
    price_spread = frame["price_spread"].to_numpy(dtype=float, copy=False)
    load_dev = frame["load_dev"].to_numpy(dtype=float, copy=False)
    renewable_dev = frame["renewable_dev"].to_numpy(dtype=float, copy=False)
    composite = np.clip(
        float(rules_config["price_spread_weight"]) * _signal(price_spread, float(rules_config["spread_threshold"]))
        + float(rules_config["load_dev_weight"]) * _signal(load_dev, float(rules_config["load_dev_threshold"]))
        + float(rules_config["renewable_dev_weight"]) * _signal(-renewable_dev, float(rules_config["renewable_dev_threshold"])),
        -float(rules_config["signal_clip"]),
        float(rules_config["signal_clip"]),
    )
    bandwidth, _ = _compute_hourly_bandwidth(q_base, exposure_bandwidth, policy_state, rules_config, composite)
    initial_radius_multiplier = float(config["hpso"]["lower"].get("spot_delta_initial_radius_multiplier", 1.0))
    initial_radius = np.maximum(np.abs(q_base), 1.0) * initial_radius_multiplier
    lower = torch.as_tensor(-initial_radius, dtype=torch.float64)
    upper = torch.as_tensor(initial_radius, dtype=torch.float64)
    bounded_mask = torch.zeros_like(lower, dtype=torch.bool)

    q_base_t = _to_tensor(q_base, settings.device).unsqueeze(0)
    da_t = _to_tensor(frame["全网统一出清价格_日前"].to_numpy(dtype=float), settings.device).unsqueeze(0)
    spread_t = _to_tensor(np.abs(price_spread), settings.device).unsqueeze(0)
    load_dev_t = _to_tensor(np.abs(load_dev), settings.device).unsqueeze(0)
    renewable_t = _to_tensor(np.abs(renewable_dev), settings.device).unsqueeze(0)
    net_target_t = _to_tensor(frame["net_load_id"].to_numpy(dtype=float, copy=False) - q_lt, settings.device).unsqueeze(0)
    weights = config["hpso"]["objective_weights"]

    def objective(positions: torch.Tensor) -> torch.Tensor:
        return _score_lower_objective(
            _lower_objective_components(
                positions=positions,
                q_base_t=q_base_t,
                da_t=da_t,
                spread_t=spread_t,
                load_dev_t=load_dev_t,
                renewable_t=renewable_t,
                net_target_t=net_target_t,
                spot_quadratic_weight=float(weights.get("spot_delta_quadratic", 1.0)),
            ),
            weights,
        )

    optimizer = HybridParticleSwarmOptimizer(lower, upper, settings, objective, bounded_mask=bounded_mask)
    best, _, convergence = optimizer.optimize()
    trace, stats = _build_hourly_trace_from_delta(
        hourly_frame=hourly_frame,
        q_lt_hourly=q_lt_hourly,
        exposure_bandwidth=exposure_bandwidth,
        policy_state=policy_state,
        rules_config=rules_config,
        raw_delta=best.numpy().astype(float, copy=False),
    )
    convergence["scope"] = "lower"
    return trace, stats, convergence


def simulate_hpso_week(
    bundle: dict[str, Any],
    week_start: pd.Timestamp,
    upper_action: pd.Series,
    config: dict[str, Any],
    seed_offset: int,
    previous_lock_ratio: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quarter, hourly, metadata = _get_week_frames(bundle, week_start)
    lock_ratio_final = float(upper_action["lock_ratio_final"])
    exposure_bandwidth = float(upper_action["exposure_bandwidth"])
    forecast_weekly_net_demand = float(metadata["forecast_weekly_net_demand_mwh"])
    q_lt_target = lock_ratio_final * forecast_weekly_net_demand
    q_lt_hourly = _allocate_weekly_lt_to_hourly(hourly, q_lt_target)

    hourly_trace, hpso_stats, lower_convergence = optimize_hourly_delta(
        hourly,
        q_lt_hourly=q_lt_hourly,
        exposure_bandwidth=exposure_bandwidth,
        policy_state=metadata,
        config=config,
        seed_offset=seed_offset,
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
    hourly_smooth = float(hourly_trace["delta_q"].diff().abs().fillna(0.0).mean()) / max(float(config["rules"]["gamma_max"]), 1e-6)
    action_smooth = abs(lock_ratio_final - previous_lock_ratio) / max(float(config["constraints"]["delta_h_max"]), 1e-6)
    objective_value = (
        procurement_cost
        + float(config["hpso"]["objective_weights"]["risk"]) * risk_term
        + trans_cost
        + float(config["hpso"]["objective_weights"]["weekly_lock_change"]) * action_smooth
        + float(config["hpso"]["objective_weights"]["hourly_smooth"]) * hourly_smooth
        + float(config["hpso"]["objective_weights"]["hedge_error"]) * hedge_error
    )
    reward_raw = -objective_value / max(forecast_weekly_net_demand * max(float(metadata["da_price_mean"]), 1.0), 1.0)
    reward = float(np.tanh(reward_raw / float(config["env"]["reward_temperature"])))
    lock_ratio_base = float(upper_action["lock_ratio_base"])
    delta_lock_ratio = lock_ratio_final - lock_ratio_base
    delta_lock_ratio_raw = float(upper_action.get("delta_lock_ratio_raw", delta_lock_ratio))

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
        "baseline_cost_w": np.nan,
        "delta_cost_w": np.nan,
        "tail_penalty_w": 0.0,
        "reward_budget_w": 0.0,
        "reward_excess_w": 0.0,
        "z_delta_cost_w": 0.0,
        "action_smooth_w": action_smooth,
        "hourly_smooth_w": hourly_smooth,
        "hedge_error_norm_w": hedge_error / max(float(hourly["net_load_id"].clip(lower=0.0).mean()), 1e-6),
        "exec_quality_w": hourly_smooth + hedge_error,
        "reward_raw": reward_raw,
        "reward": reward,
        "hpso_objective_w": objective_value,
        "avg_adjustment_mwh": float(hourly_trace["delta_q"].abs().mean()),
        "mean_bandwidth_mwh": float(hourly_trace["bandwidth_mwh"].mean()),
        "bound_clip_count": int(hpso_stats["bound_clip_count"]),
        "smooth_clip_count": int(hpso_stats["smooth_clip_count"]),
        "soft_clip_count": int(hpso_stats["soft_clip_count"]),
        "non_negative_clip_count": int(hpso_stats["non_negative_clip_count"]),
        "cvar_budget_excess": 0.0,
    }
    hourly_trace["week_start"] = pd.Timestamp(week_start)
    settlement["week_start"] = pd.Timestamp(week_start)
    lower_convergence["week_start"] = pd.Timestamp(week_start)
    return week_summary, hourly_trace, settlement, lower_convergence


def simulate_hpso_strategy(
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    config: dict[str, Any],
    strategy_name: str = "hpso",
    market_vol_scale: float = 1.0,
    price_cap_multiplier: float = 1.0,
    forecast_error_scale: float = 1.0,
) -> dict[str, Any]:
    weeks = [pd.Timestamp(week) for week in weeks]
    bundle_variant = bundle.copy()
    bundle_variant["hourly"] = bundle["hourly"].copy()
    bundle_variant["quarter"] = bundle["quarter"].copy()
    bundle_variant["weekly_metadata"] = bundle["weekly_metadata"].copy()
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
        bundle_variant["weekly_metadata"]["forecast_weekly_net_demand_mwh"] = (
            bundle_variant["weekly_metadata"]["forecast_weekly_net_demand_mwh"] * float(forecast_error_scale)
        )
    if any(scale != 1.0 for scale in (market_vol_scale, price_cap_multiplier, forecast_error_scale)):
        prepare_runtime_bundle(bundle_variant)

    upper_actions, upper_convergence = optimize_upper_actions(bundle_variant, weeks, config)
    action_lookup = upper_actions.set_index("week_start")
    weekly_records = []
    hourly_records = []
    settlement_records = []
    lower_convergence_records = []
    actions: dict[pd.Timestamp, dict[str, float | str]] = {}
    previous_lock_ratio = 0.0

    for index, week in enumerate(weeks):
        upper_action = action_lookup.loc[week]
        summary, hourly_trace, settlement, lower_convergence = simulate_hpso_week(
            bundle=bundle_variant,
            week_start=week,
            upper_action=upper_action,
            config=config,
            seed_offset=1000 + index * 97,
            previous_lock_ratio=previous_lock_ratio,
        )
        summary["strategy"] = strategy_name
        hourly_trace["strategy"] = strategy_name
        settlement["strategy"] = strategy_name
        lower_convergence["strategy"] = strategy_name
        previous_lock_ratio = float(summary["lock_ratio_final"])
        actions[week] = {
            "mode": "absolute",
            "target_lock_ratio": float(summary["lock_ratio_final"]),
            "exposure_bandwidth": float(summary["exposure_bandwidth"]),
        }
        weekly_records.append(summary)
        hourly_records.append(hourly_trace)
        settlement_records.append(settlement)
        lower_convergence_records.append(lower_convergence)

    weekly_results = pd.DataFrame(weekly_records).sort_values("week_start").reset_index(drop=True)
    hourly_results = pd.concat(hourly_records, ignore_index=True)
    settlement_results = pd.concat(settlement_records, ignore_index=True)
    metrics = summarize_strategy_results(
        weekly_results=weekly_results,
        settlement_results=settlement_results,
        strategy_name=strategy_name,
        cvar_alpha=float(config["cost"]["cvar_alpha"]),
    )
    upper_convergence["strategy"] = strategy_name
    return {
        "weekly_results": weekly_results,
        "hourly_results": hourly_results,
        "settlement_results": settlement_results,
        "metrics": metrics,
        "actions": actions,
        "upper_actions": upper_actions,
        "upper_convergence": upper_convergence,
        "lower_convergence": pd.concat(lower_convergence_records, ignore_index=True),
    }


def train_hpso_model(
    bundle: dict[str, Any],
    train_weeks: list[pd.Timestamp],
    val_weeks: list[pd.Timestamp],
    config: dict[str, Any],
    output_paths: dict[str, Path],
    run_name: str = "hpso",
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    unique_train_weeks = sorted({pd.Timestamp(week) for week in train_weeks})
    train_result = simulate_hpso_strategy(bundle, unique_train_weeks, config, strategy_name=f"{run_name}_train")
    if persist_artifacts:
        train_result["weekly_results"].to_csv(output_paths["metrics"] / f"{run_name}_weekly_practice_data.csv", index=False)
        train_result["hourly_results"].to_csv(output_paths["metrics"] / f"{run_name}_hourly_delta_q.csv", index=False)
        train_result["upper_actions"].to_csv(output_paths["metrics"] / f"{run_name}_upper_weekly_actions.csv", index=False)
        pd.concat(
            [train_result["upper_convergence"], train_result["lower_convergence"]],
            ignore_index=True,
        ).to_csv(output_paths["metrics"] / f"{run_name}_convergence_curve.csv", index=False)
    device = _settings_from_config(config, "upper").device
    return {
        "model": HPSOModel(device=device, config=config),
        "model_path": None,
        "best_model_path": None,
        "train_metrics": train_result["weekly_results"],
        "eval_metrics": pd.DataFrame(),
        "device": device,
        "gpu_used": device.startswith("cuda"),
        "run_name": run_name,
        "train_result": train_result,
    }


def evaluate_hpso_policy(
    model: HPSOModel,
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    config: dict[str, Any],
    strategy_name: str = "hpso",
) -> dict[str, Any]:
    return simulate_hpso_strategy(bundle=bundle, weeks=weeks, config=config, strategy_name=strategy_name)
