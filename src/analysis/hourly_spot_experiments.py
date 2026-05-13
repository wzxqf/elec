from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HourlySpotExperimentCandidate:
    experiment_id: str
    signal_deadband: float
    gate_temperature: float
    hourly_limit_base_multiplier: float
    hourly_limit_shrink_multiplier: float
    friction_cost_yuan_per_mwh: float
    lambda_trade: float


def build_hourly_spot_experiment_grid(config: dict[str, Any]) -> list[HourlySpotExperimentCandidate]:
    cfg = config.get("hourly_spot_experiment", {})
    if not bool(cfg.get("enabled", False)):
        return []
    deadbands = [float(value) for value in cfg.get("signal_deadband", [0.10])]
    temperatures = [float(value) for value in cfg.get("temperature", [0.04])]
    base_multipliers = [float(value) for value in cfg.get("hourly_limit_base_multiplier", [0.50])]
    shrink_multipliers = [float(value) for value in cfg.get("hourly_limit_shrink_multiplier", [0.50])]
    friction_costs = [float(value) for value in cfg.get("friction_cost_yuan_per_mwh", [1.20])]
    lambda_trades = [float(value) for value in cfg.get("lambda_trade", [0.10])]
    candidates: list[HourlySpotExperimentCandidate] = []
    for index, values in enumerate(
        product(deadbands, temperatures, base_multipliers, shrink_multipliers, friction_costs, lambda_trades),
        start=1,
    ):
        deadband, temperature, base, shrink, friction, lambda_trade = values
        candidates.append(
            HourlySpotExperimentCandidate(
                experiment_id=f"hourly_spot_{index:03d}",
                signal_deadband=deadband,
                gate_temperature=temperature,
                hourly_limit_base_multiplier=base,
                hourly_limit_shrink_multiplier=shrink,
                friction_cost_yuan_per_mwh=friction,
                lambda_trade=lambda_trade,
            )
        )
    return candidates


def summarize_hourly_spot_guardrails(result_frame: pd.DataFrame, *, baseline: dict[str, float]) -> pd.DataFrame:
    required = {
        "experiment_id",
        "sum_excess_profit_w",
        "mean_cvar99_w",
        "mean_hedge_error_w",
        "nonzero_hour_share",
        "spot_abs_sum_mwh",
    }
    missing = sorted(required.difference(result_frame.columns))
    if missing:
        raise KeyError(", ".join(missing))
    frame = result_frame.copy()
    baseline_excess = float(baseline["sum_excess_profit_w"])
    baseline_cvar = float(baseline["mean_cvar99_w"])
    baseline_hedge = float(baseline["mean_hedge_error_w"])
    cvar_tolerance = float(baseline.get("cvar_tolerance", 1.03))
    frame["dynamic_lock_only_improved"] = frame["sum_excess_profit_w"].astype(float) > baseline_excess
    frame["tail_risk_guard_passed"] = frame["mean_cvar99_w"].astype(float) <= baseline_cvar * cvar_tolerance
    frame["hedge_error_guard_passed"] = frame["mean_hedge_error_w"].astype(float) <= baseline_hedge
    frame["nonzero_spot_guard_passed"] = frame["nonzero_hour_share"].astype(float) > 0.0
    frame["candidate_passed"] = (
        frame["dynamic_lock_only_improved"]
        & frame["tail_risk_guard_passed"]
        & frame["hedge_error_guard_passed"]
        & frame["nonzero_spot_guard_passed"]
    )
    return frame.sort_values(["candidate_passed", "sum_excess_profit_w"], ascending=[False, False]).reset_index(drop=True)
