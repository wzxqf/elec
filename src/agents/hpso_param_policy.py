from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.backtest.benchmarks import get_dynamic_lock_base_for_week
from src.backtest.contract_curve import allocate_weekly_contract_curve, build_base_24h_profile
from src.backtest.metrics import summarize_strategy_results
from src.backtest.settlement import resolve_settlement_context, settle_week


THETA_DIMENSION = 64
_TRAINING_WORKER_CONTEXT: dict[str, Any] = {}


@dataclass(frozen=True)
class UpperTheta:
    lock_delta_weights: np.ndarray
    bandwidth_weights: np.ndarray
    curve_shape_weights: np.ndarray
    policy_gate_weights: np.ndarray


@dataclass(frozen=True)
class LowerTheta:
    spread_response: np.ndarray
    load_forecast_response: np.ndarray
    renewable_response: np.ndarray
    smoothing_response: np.ndarray
    policy_band_shrink: np.ndarray


@dataclass(frozen=True)
class SplitTheta:
    upper: UpperTheta
    lower: LowerTheta


@dataclass(frozen=True)
class HPSOParamPolicyModel:
    theta: np.ndarray
    objective_value: float
    metadata: dict[str, Any]


def _slice(values: np.ndarray, start: int, width: int) -> tuple[np.ndarray, int]:
    end = start + width
    return values[start:end].astype(float, copy=True), end


def split_theta(theta: np.ndarray | list[float]) -> SplitTheta:
    values = np.asarray(theta, dtype=float)
    if values.shape != (THETA_DIMENSION,):
        raise ValueError(f"HPSO parameter policy theta must have shape ({THETA_DIMENSION},), got {values.shape}.")

    cursor = 0
    lock_delta_weights, cursor = _slice(values, cursor, 16)
    bandwidth_weights, cursor = _slice(values, cursor, 12)
    curve_shape_weights, cursor = _slice(values, cursor, 8)
    policy_gate_weights, cursor = _slice(values, cursor, 4)
    spread_response, cursor = _slice(values, cursor, 6)
    load_forecast_response, cursor = _slice(values, cursor, 6)
    renewable_response, cursor = _slice(values, cursor, 4)
    smoothing_response, cursor = _slice(values, cursor, 4)
    policy_band_shrink, cursor = _slice(values, cursor, 4)
    if cursor != THETA_DIMENSION:
        raise ValueError(f"Internal theta layout consumed {cursor} parameters, expected {THETA_DIMENSION}.")

    return SplitTheta(
        upper=UpperTheta(
            lock_delta_weights=lock_delta_weights,
            bandwidth_weights=bandwidth_weights,
            curve_shape_weights=curve_shape_weights,
            policy_gate_weights=policy_gate_weights,
        ),
        lower=LowerTheta(
            spread_response=spread_response,
            load_forecast_response=load_forecast_response,
            renewable_response=renewable_response,
            smoothing_response=smoothing_response,
            policy_band_shrink=policy_band_shrink,
        ),
    )


def _feature(row: pd.Series, name: str, scale: float = 1.0) -> float:
    value = row.get(name, 0.0)
    if isinstance(value, pd.Series):
        if value.empty:
            value = 0.0
        else:
            value = value.dropna().iloc[-1] if value.dropna().shape[0] > 0 else 0.0
    if pd.isna(value):
        value = 0.0
    return float(value) / max(float(scale), 1e-9)


def _upper_feature_vector(row: pd.Series, width: int) -> np.ndarray:
    base = np.array(
        [
            1.0,
            _feature(row, "prev_spread_mean", 100.0),
            _feature(row, "prev_da_price_mean", 500.0),
            _feature(row, "prev_id_price_mean", 500.0),
            _feature(row, "prev_load_dev_std", 1000.0),
            _feature(row, "prev_renewable_ratio_da_mean", 1.0),
            _feature(row, "renewable_mechanism_active", 1.0),
            _feature(row, "lt_price_linked_active", 1.0),
            _feature(row, "forward_price_linkage_days", 60.0),
            _feature(row, "forward_mechanism_execution_days", 60.0),
            _feature(row, "forward_ancillary_coupling_days", 60.0),
            _feature(row, "forward_info_forecast_boundary_days", 60.0),
            _feature(row, "ancillary_freq_reserve_tight", 1.0),
            _feature(row, "ancillary_peak_shaving_pause", 1.0),
            _feature(row, "forecast_weekly_net_demand_mwh", 5_000_000.0),
            _feature(row, "lt_price_w", 500.0),
        ],
        dtype=float,
    )
    if len(base) >= width:
        return base[:width]
    return np.pad(base, (0, width - len(base)))


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(-value))


def infer_upper_action(
    theta: np.ndarray | list[float],
    feature_row: pd.Series,
    lock_ratio_base: float,
    previous_lock_ratio: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parts = split_theta(theta)
    policy_cfg = (config or {}).get("policy", {}).get("upper", {})
    constraints = (config or {}).get("constraints", {})
    lock_delta_scale = float(policy_cfg.get("lock_delta_scale", constraints.get("delta_lock_cap", 0.12)))
    bandwidth_min = float(policy_cfg.get("bandwidth_min", 0.0))
    bandwidth_max = float(policy_cfg.get("bandwidth_max", 0.85))
    min_lock = float(constraints.get("lock_ratio_min", 0.10))
    max_lock = float(constraints.get("lock_ratio_max", 0.85))
    max_step = float(constraints.get("delta_h_max", 0.12))

    lock_features = _upper_feature_vector(feature_row, len(parts.upper.lock_delta_weights))
    band_features = _upper_feature_vector(feature_row, len(parts.upper.bandwidth_weights))
    policy_features = _upper_feature_vector(feature_row, len(parts.upper.policy_gate_weights))

    policy_gate = float(np.tanh(np.dot(policy_features, parts.upper.policy_gate_weights)))
    delta_signal = float(np.tanh(np.dot(lock_features, parts.upper.lock_delta_weights) + 0.25 * policy_gate))
    delta_lock_ratio = delta_signal * lock_delta_scale
    target_lock = float(lock_ratio_base) + delta_lock_ratio
    lock_ratio_pre_step = float(np.clip(target_lock, min_lock, max_lock))
    lock_ratio_final = float(np.clip(lock_ratio_pre_step, previous_lock_ratio - max_step, previous_lock_ratio + max_step))
    lock_ratio_final = float(np.clip(lock_ratio_final, min_lock, max_lock))

    band_signal = float(np.dot(band_features, parts.upper.bandwidth_weights) - 0.10 * policy_gate)
    exposure_bandwidth = bandwidth_min + (bandwidth_max - bandwidth_min) * _sigmoid(band_signal)
    return {
        "lock_ratio_base": float(lock_ratio_base),
        "delta_lock_ratio_raw": float(delta_signal),
        "delta_lock_ratio": float(lock_ratio_final - float(lock_ratio_base)),
        "lock_ratio_final": lock_ratio_final,
        "target_lock_ratio": lock_ratio_final,
        "exposure_bandwidth": float(np.clip(exposure_bandwidth, bandwidth_min, bandwidth_max)),
        "curve_shape_params": parts.upper.curve_shape_weights.astype(float, copy=True),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_hpso_param_policy(model: HPSOParamPolicyModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "theta": np.asarray(model.theta, dtype=float).tolist(),
        "objective_value": float(model.objective_value),
        "metadata": dict(model.metadata),
        "theta_layout": {
            "upper_total": 40,
            "lower_total": 24,
            "dimension": THETA_DIMENSION,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def load_hpso_param_policy(path: str | Path) -> HPSOParamPolicyModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HPSOParamPolicyModel(
        theta=np.asarray(payload["theta"], dtype=float),
        objective_value=float(payload["objective_value"]),
        metadata=dict(payload.get("metadata", {})),
    )


def theta_to_frame(theta: np.ndarray | list[float]) -> pd.DataFrame:
    values = np.asarray(theta, dtype=float)
    return pd.DataFrame({"parameter_index": np.arange(len(values)), "value": values})


def resolve_parallel_worker_count(config: dict[str, Any], particle_count: int) -> int:
    configured = int(config.get("hpso", {}).get("parallel", {}).get("worker_count", 1))
    cpu_count = os.cpu_count() or 1
    return max(1, min(configured, particle_count, cpu_count))


def _get_week_frames(bundle: dict[str, Any], week_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    week_start = pd.Timestamp(week_start)
    quarter_lookup = bundle.get("quarter_by_week")
    hourly_lookup = bundle.get("hourly_by_week")
    metadata_lookup = bundle.get("weekly_metadata_by_week")
    if isinstance(quarter_lookup, dict) and isinstance(hourly_lookup, dict) and isinstance(metadata_lookup, dict):
        return quarter_lookup[week_start].copy(), hourly_lookup[week_start].copy(), metadata_lookup[week_start].copy()
    quarter = bundle["quarter"].loc[bundle["quarter"]["week_start"] == week_start].copy()
    hourly = bundle["hourly"].loc[bundle["hourly"]["week_start"] == week_start].copy()
    metadata = bundle["weekly_metadata"].set_index("week_start").loc[week_start].copy()
    return quarter, hourly, metadata


def _week_risk_term(settlement: pd.DataFrame, config: dict[str, Any]) -> tuple[float, float]:
    per_interval = settlement["procurement_cost_15m"]
    sigma = float(per_interval.std(ddof=0))
    alpha = float(config["cost"]["cvar_alpha"])
    threshold = float(per_interval.quantile(alpha))
    tail = per_interval.loc[per_interval >= threshold]
    cvar = float(tail.mean()) if not tail.empty else threshold
    risk_term = float(config["cost"]["risk_vol_weight"]) * sigma + float(config["cost"]["risk_cvar_weight"]) * cvar
    return risk_term, cvar


def _feature_row_for_week(bundle: dict[str, Any], week: pd.Timestamp) -> pd.Series:
    lookup = bundle.get("weekly_feature_by_week")
    if isinstance(lookup, dict) and pd.Timestamp(week) in lookup:
        return lookup[pd.Timestamp(week)].copy()
    return bundle["weekly_features"].set_index("week_start").loc[pd.Timestamp(week)].copy()


def _simulate_theta_score(
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    theta: np.ndarray,
    config: dict[str, Any],
    strategy_name: str,
) -> tuple[float, int]:
    result = simulate_param_policy_strategy(
        bundle=bundle,
        weeks=weeks,
        theta=theta,
        config=config,
        strategy_name=strategy_name,
    )
    score = float(result["weekly_results"]["hpso_param_objective_w"].mean())
    return score, len(result["weekly_results"])


def _initialize_training_worker(
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    config: dict[str, Any],
    strategy_name: str,
) -> None:
    global _TRAINING_WORKER_CONTEXT
    _TRAINING_WORKER_CONTEXT = {
        "bundle": bundle,
        "weeks": weeks,
        "config": config,
        "strategy_name": strategy_name,
    }


def _evaluate_theta_worker(task: tuple[int, np.ndarray]) -> dict[str, Any]:
    particle_index, theta = task
    score, weeks_evaluated = _simulate_theta_score(
        bundle=_TRAINING_WORKER_CONTEXT["bundle"],
        weeks=_TRAINING_WORKER_CONTEXT["weeks"],
        theta=np.asarray(theta, dtype=float),
        config=_TRAINING_WORKER_CONTEXT["config"],
        strategy_name=_TRAINING_WORKER_CONTEXT["strategy_name"],
    )
    return {
        "particle_index": int(particle_index),
        "score": float(score),
        "weeks_evaluated": int(weeks_evaluated),
    }


def simulate_param_policy_week(
    bundle: dict[str, Any],
    week_start: pd.Timestamp,
    theta: np.ndarray | list[float],
    config: dict[str, Any],
    previous_lock_ratio: float = 0.0,
    previous_reward: float = 0.0,
    strategy_name: str = "hpso_param_policy",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.rules.rolling_hedge import apply_causal_rolling_hedge

    week_start = pd.Timestamp(week_start)
    quarter, hourly, metadata = _get_week_frames(bundle, week_start)
    feature_row = _feature_row_for_week(bundle, week_start)
    enriched_feature = pd.Series(
        {
            **feature_row.to_dict(),
            **metadata.to_dict(),
            "previous_lock_ratio": previous_lock_ratio,
            "previous_reward": previous_reward,
        }
    )
    lock_ratio_base = get_dynamic_lock_base_for_week(
        bundle["weekly_features"],
        week_start,
        config,
        weekly_feature_by_week=bundle.get("weekly_feature_by_week"),
    )
    upper_action = infer_upper_action(
        theta=theta,
        feature_row=enriched_feature,
        lock_ratio_base=lock_ratio_base,
        previous_lock_ratio=previous_lock_ratio,
        config=config,
    )
    forecast_weekly_net_demand = float(metadata["forecast_weekly_net_demand_mwh"])
    q_lt_target = float(upper_action["lock_ratio_final"]) * forecast_weekly_net_demand
    base_profile = build_base_24h_profile(bundle["hourly"])
    contract_trace = allocate_weekly_contract_curve(
        hourly,
        q_lt_target=q_lt_target,
        base_profile=base_profile,
        curve_params=np.asarray(upper_action["curve_shape_params"], dtype=float),
        adjustment_scale=float(config.get("policy", {}).get("contract_curve", {}).get("shape_adjustment_scale", 0.20)),
        positive_floor=float(config.get("policy", {}).get("contract_curve", {}).get("positive_floor", 0.05)),
    )
    hourly_trace, audit = apply_causal_rolling_hedge(
        hourly_frame=hourly,
        q_lt_hourly=contract_trace["q_lt_hourly"],
        theta=theta,
        exposure_bandwidth=float(upper_action["exposure_bandwidth"]),
        policy_state=metadata,
        config=config,
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
    hedge_error = float(hourly_trace["hedge_error_abs"].dropna().mean()) if hourly_trace["hedge_error_abs"].notna().any() else 0.0
    hourly_smooth = float(hourly_trace["delta_q"].diff().abs().fillna(0.0).mean()) / max(
        float(config.get("policy", {}).get("lower", {}).get("smooth_limit_mwh", config["rules"].get("gamma_max", 420.0))),
        1e-6,
    )
    action_smooth = abs(float(upper_action["lock_ratio_final"]) - previous_lock_ratio) / max(
        float(config["constraints"]["delta_h_max"]),
        1e-6,
    )

    baseline_lookup = bundle.get("reward_reference_by_week")
    baseline_cost = procurement_cost
    if isinstance(baseline_lookup, dict) and week_start in baseline_lookup:
        baseline_cost = float(baseline_lookup[week_start]["baseline_cost_w"])
    objective_weights = config["hpso"]["objective_weights"]
    objective_value = (
        float(objective_weights.get("procurement_cost", 1.0)) * procurement_cost
        + float(objective_weights.get("dynamic_baseline_gap", 1.15)) * max(procurement_cost - baseline_cost, 0.0)
        + float(objective_weights.get("tail_risk", objective_weights.get("risk", 0.5))) * risk_term
        + float(objective_weights.get("hedge_error", 0.18)) * hedge_error
        + float(objective_weights.get("hourly_smooth", 0.08)) * hourly_smooth
        + float(objective_weights.get("action_smooth", 0.12)) * action_smooth
        + float(objective_weights.get("transaction", 1.0)) * trans_cost
    )
    reward_raw = -objective_value / max(forecast_weekly_net_demand * max(float(metadata["da_price_mean"]), 1.0), 1.0)
    reward = float(np.tanh(reward_raw / float(config["env"]["reward_temperature"])))
    week_summary = {
        "week_start": week_start,
        "is_partial_week": bool(metadata["is_partial_week"]),
        "lock_ratio_base": float(upper_action["lock_ratio_base"]),
        "delta_lock_ratio_raw": float(upper_action["delta_lock_ratio_raw"]),
        "delta_lock_ratio": float(upper_action["delta_lock_ratio"]),
        "lock_ratio_final": float(upper_action["lock_ratio_final"]),
        "lock_ratio": float(upper_action["lock_ratio_final"]),
        "exposure_bandwidth": float(upper_action["exposure_bandwidth"]),
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
        "delta_cost_w": procurement_cost - baseline_cost,
        "tail_penalty_w": 0.0,
        "reward_budget_w": 0.0,
        "reward_excess_w": 0.0,
        "z_delta_cost_w": 0.0,
        "action_smooth_w": action_smooth,
        "hourly_smooth_w": hourly_smooth,
        "hedge_error_norm_w": hedge_error / max(float(hourly["net_load_da"].clip(lower=0.0).mean()), 1e-6),
        "exec_quality_w": hourly_smooth + hedge_error,
        "reward_raw": reward_raw,
        "reward": reward,
        "hpso_param_objective_w": objective_value,
        "avg_adjustment_mwh": float(hourly_trace["delta_q"].abs().mean()),
        "mean_bandwidth_mwh": float(hourly_trace["bandwidth_mwh"].mean()),
        "bound_clip_count": 0,
        "smooth_clip_count": 0,
        "soft_clip_count": 0,
        "non_negative_clip_count": int(hourly_trace["clipped_non_negative"].sum()),
        "cvar_budget_excess": 0.0,
        "strategy": strategy_name,
    }
    hourly_trace["week_start"] = week_start
    hourly_trace["strategy"] = strategy_name
    settlement["week_start"] = week_start
    settlement["strategy"] = strategy_name
    contract_trace["week_start"] = week_start
    contract_trace["strategy"] = strategy_name
    audit["week_start"] = week_start
    audit["strategy"] = strategy_name
    return week_summary, hourly_trace, settlement, contract_trace, audit


def simulate_param_policy_strategy(
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    theta: np.ndarray | list[float],
    config: dict[str, Any],
    strategy_name: str = "hpso_param_policy",
) -> dict[str, Any]:
    weekly_records: list[dict[str, Any]] = []
    hourly_records: list[pd.DataFrame] = []
    settlement_records: list[pd.DataFrame] = []
    contract_records: list[pd.DataFrame] = []
    audit_records: list[pd.DataFrame] = []
    actions: dict[pd.Timestamp, dict[str, Any]] = {}
    previous_lock_ratio = 0.0
    previous_reward = 0.0
    for week in [pd.Timestamp(value) for value in weeks]:
        summary, hourly_trace, settlement, contract_trace, audit = simulate_param_policy_week(
            bundle=bundle,
            week_start=week,
            theta=theta,
            config=config,
            previous_lock_ratio=previous_lock_ratio,
            previous_reward=previous_reward,
            strategy_name=strategy_name,
        )
        previous_lock_ratio = float(summary["lock_ratio_final"])
        previous_reward = float(summary["reward"])
        actions[week] = {
            "mode": "absolute",
            "target_lock_ratio": float(summary["lock_ratio_final"]),
            "exposure_bandwidth": float(summary["exposure_bandwidth"]),
        }
        weekly_records.append(summary)
        hourly_records.append(hourly_trace)
        settlement_records.append(settlement)
        contract_records.append(contract_trace)
        audit_records.append(audit)
    weekly_results = pd.DataFrame(weekly_records).sort_values("week_start").reset_index(drop=True)
    hourly_results = pd.concat(hourly_records, ignore_index=True) if hourly_records else pd.DataFrame()
    settlement_results = pd.concat(settlement_records, ignore_index=True) if settlement_records else pd.DataFrame()
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
        "contract_curve": pd.concat(contract_records, ignore_index=True) if contract_records else pd.DataFrame(),
        "information_boundary_audit": pd.concat(audit_records, ignore_index=True) if audit_records else pd.DataFrame(),
        "metrics": metrics,
        "actions": actions,
        "theta": np.asarray(theta, dtype=float),
    }


def _settings_from_param_config(config: dict[str, Any]) -> Any:
    from src.agents.hpso import HPSOSettings

    hpso_cfg = config["hpso"]
    swarm = hpso_cfg["swarm"]
    requested_device = str(hpso_cfg.get("device", config.get("device", "cpu")))
    allow_cpu = bool(hpso_cfg.get("allow_cpu", True))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("HPSO_PARAM_POLICY requires CUDA, but CUDA is unavailable and allow_cpu=false.")
        device = "cpu"
    else:
        device = requested_device
    return HPSOSettings(
        particles=int(swarm["particles"]),
        iterations=int(swarm["iterations"]),
        inertia_weight=float(swarm["inertia_weight"]),
        cognitive_factor=float(swarm["cognitive_factor"]),
        social_factor=float(swarm["social_factor"]),
        initial_temperature=float(swarm["initial_temperature"]),
        cooling_rate=float(swarm["cooling_rate"]),
        perturbation_scale=float(swarm["perturbation_scale"]),
        stagnation_window=int(swarm["stagnation_window"]),
        seed=int(hpso_cfg.get("seed", config.get("seed", 42))),
        device=device,
        allow_cpu=allow_cpu,
        backprop_steps=int(swarm.get("backprop_steps", 0)),
        backprop_learning_rate=float(swarm.get("backprop_learning_rate", 0.0)),
        backprop_clip_norm=float(swarm.get("backprop_clip_norm", 5.0)),
    )


def train_hpso_param_policy(
    bundle: dict[str, Any],
    train_weeks: list[pd.Timestamp],
    config: dict[str, Any],
    output_paths: dict[str, Path],
    run_name: str = "hpso_param_policy",
    persist_artifacts: bool = True,
    status_tracker: Any | None = None,
) -> dict[str, Any]:
    from src.agents.hpso import HybridParticleSwarmOptimizer

    settings = _settings_from_param_config(config)
    train_sequence = [pd.Timestamp(week) for week in train_weeks]
    if not train_sequence:
        raise ValueError("HPSO_PARAM_POLICY training requires at least one training week.")

    lower = torch.full((THETA_DIMENSION,), -1.0, dtype=torch.float64)
    upper = torch.full((THETA_DIMENSION,), 1.0, dtype=torch.float64)

    rollout_records: list[dict[str, Any]] = []
    parallel_workers = resolve_parallel_worker_count(config, settings.particles)
    strategy_name = f"{run_name}_train"

    def _record_rollout(evaluations: list[dict[str, Any]]) -> None:
        for row in evaluations:
            rollout_records.append(
                {
                    "particle_index": int(row["particle_index"]),
                    "score": float(row["score"]),
                    "weeks_evaluated": int(row["weeks_evaluated"]),
                    "unique_weeks_evaluated": len(set(train_sequence)),
                }
            )

    def _evaluate_locally(positions_np: np.ndarray) -> list[dict[str, Any]]:
        evaluations: list[dict[str, Any]] = []
        for particle_index, position in enumerate(positions_np):
            score, weeks_evaluated = _simulate_theta_score(
                bundle=bundle,
                weeks=train_sequence,
                theta=np.asarray(position, dtype=float),
                config=config,
                strategy_name=strategy_name,
            )
            evaluations.append(
                {
                    "particle_index": int(particle_index),
                    "score": float(score),
                    "weeks_evaluated": int(weeks_evaluated),
                }
            )
        return evaluations

    executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
    parallel_backend = "serial"
    if parallel_workers > 1:
        try:
            executor = ProcessPoolExecutor(
                max_workers=parallel_workers,
                initializer=_initialize_training_worker,
                initargs=(bundle, train_sequence, config, strategy_name),
            )
            parallel_backend = "process"
        except (OSError, PermissionError):
            executor = ThreadPoolExecutor(max_workers=parallel_workers)
            parallel_backend = "thread"

    def objective(positions: torch.Tensor) -> torch.Tensor:
        positions_np = np.asarray(positions.detach().cpu().numpy(), dtype=float)
        if executor is None:
            evaluations = _evaluate_locally(positions_np)
        elif parallel_backend == "thread":
            evaluations = list(
                executor.map(
                    lambda item: {
                        "particle_index": int(item[0]),
                        "score": float(
                            _simulate_theta_score(
                                bundle=bundle,
                                weeks=train_sequence,
                                theta=np.asarray(item[1], dtype=float),
                                config=config,
                                strategy_name=strategy_name,
                            )[0]
                        ),
                        "weeks_evaluated": len(train_sequence),
                    },
                    [(index, position) for index, position in enumerate(positions_np)],
                )
            )
        else:
            evaluations = list(executor.map(_evaluate_theta_worker, [(index, position) for index, position in enumerate(positions_np)]))
        _record_rollout(evaluations)
        scores = [float(row["score"]) for row in evaluations]
        return torch.as_tensor(scores, dtype=torch.float64, device=settings.device)

    def _on_progress(payload: dict[str, Any]) -> None:
        if status_tracker is None:
            return
        iteration = int(payload.get("iteration", 0))
        total_iterations = max(int(payload.get("total_iterations", settings.iterations)), 1)
        phase_progress = min(max(iteration / total_iterations, 0.0), 1.0)
        status_tracker.update(
            stage="训练",
            phase_name="HPSO 参数训练",
            phase_progress=phase_progress,
            total_progress=phase_progress / 3.0,
            message=f"iter {iteration}/{total_iterations} best {float(payload.get('best_score', float('nan'))):.4f}",
        )

    try:
        optimizer = HybridParticleSwarmOptimizer(
            lower,
            upper,
            settings,
            objective,
            bounded_mask=torch.ones_like(lower, dtype=torch.bool),
            progress_callback=_on_progress,
        )
        best_theta, best_score, convergence = optimizer.optimize()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    model = HPSOParamPolicyModel(
        theta=best_theta.numpy().astype(float, copy=False),
        objective_value=float(best_score),
        metadata={
            "version": config["version"],
            "algorithm": "HPSO_PARAM_POLICY",
            "train_sequence_length": len(train_sequence),
            "unique_train_weeks": len(set(train_sequence)),
        },
    )
    train_result = simulate_param_policy_strategy(
        bundle=bundle,
        weeks=train_sequence,
        theta=model.theta,
        config=config,
        strategy_name=strategy_name,
    )
    model_path = output_paths["models"] / "hpso_param_policy.json"
    runtime_profile = {
        "optimizer_device": settings.device,
        "rollout_compute_device": "cpu",
        "parallel_workers": int(parallel_workers),
        "parallel_backend": parallel_backend,
        "requested_device": str(config.get("hpso", {}).get("device", config.get("device", "cpu"))),
        "gpu_optimizer_used": bool(settings.device.startswith("cuda")),
    }
    if persist_artifacts:
        save_hpso_param_policy(model, model_path)
        theta_to_frame(model.theta).to_csv(output_paths["metrics"] / "hpso_theta_trace.csv", index=False)
        convergence.to_csv(output_paths["metrics"] / "hpso_convergence_curve.csv", index=False)
        pd.DataFrame(rollout_records).to_csv(output_paths["metrics"] / "hpso_training_rollout.csv", index=False)
        train_result["weekly_results"].to_csv(output_paths["metrics"] / "hpso_weekly_practice_data.csv", index=False)
        train_result["hourly_results"].to_csv(output_paths["metrics"] / "hpso_policy_inference_trace.csv", index=False)
        train_result["contract_curve"].to_csv(output_paths["metrics"] / "contract_curve_24h.csv", index=False)
        train_result["information_boundary_audit"].to_csv(output_paths["metrics"] / "information_boundary_audit.csv", index=False)
    return {
        "model": model,
        "model_path": model_path,
        "best_model_path": model_path,
        "train_metrics": train_result["weekly_results"],
        "eval_metrics": pd.DataFrame(),
        "device": settings.device,
        "gpu_used": runtime_profile["gpu_optimizer_used"],
        "runtime_profile": runtime_profile,
        "run_name": run_name,
        "train_result": train_result,
    }
