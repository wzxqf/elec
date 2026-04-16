from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_DECISION_FIELDS = {
    "net_load_id",
    "net_load_id_mwh",
    "全网统一出清价格_日内",
    "price_spread",
    "load_dev",
    "renewable_dev",
}


def _lag_or_zero(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name in frame.columns:
        return frame[name].to_numpy(dtype=float, copy=False)
    source_name = name.removesuffix("_lag1")
    if source_name in frame.columns:
        return frame[source_name].shift(1).fillna(0.0).to_numpy(dtype=float, copy=False)
    return np.zeros(len(frame), dtype=float)


def _decision_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in sorted(set(frame.columns).union(FORBIDDEN_DECISION_FIELDS)):
        rows.append(
            {
                "field": field,
                "available_in_frame": bool(field in frame.columns),
                "used_for_decision": bool(field not in FORBIDDEN_DECISION_FIELDS and field in frame.columns),
            }
        )
    return pd.DataFrame(rows)


def apply_causal_rolling_hedge(
    hourly_frame: pd.DataFrame,
    q_lt_hourly: pd.Series,
    theta: np.ndarray | list[float],
    exposure_bandwidth: float,
    policy_state: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = hourly_frame.copy().reset_index(drop=True)
    lower_cfg = config.get("policy", {}).get("lower", {})
    non_negative_spot = bool(lower_cfg.get("non_negative_spot", True))
    smooth_limit = float(lower_cfg.get("smooth_limit_mwh", 420.0))
    q_lt = np.asarray(q_lt_hourly, dtype=float)
    if q_lt.shape[0] != len(frame):
        raise ValueError("q_lt_hourly length must match hourly_frame length.")

    net_load_da = frame["net_load_da"].to_numpy(dtype=float, copy=False)
    q_base = net_load_da - q_lt
    spread_lag = _lag_or_zero(frame, "price_spread_lag1")
    load_lag = _lag_or_zero(frame, "load_dev_lag1")
    renewable_lag = _lag_or_zero(frame, "renewable_dev_lag1")

    ancillary_tight = float(policy_state.get("ancillary_freq_reserve_tight", 0.0) or 0.0)
    peak_pause = float(policy_state.get("ancillary_peak_shaving_pause", 0.0) or 0.0)
    policy_shrink = float(np.clip(1.0 - 0.10 * ancillary_tight - 0.05 * peak_pause, 0.2, 1.0))
    exposure = float(np.clip(exposure_bandwidth, 0.0, 1.0))
    bandwidth = np.maximum(np.abs(q_base), 1.0) * exposure * policy_shrink

    deltas = []
    previous_delta = 0.0
    for index in range(len(frame)):
        raw_signal = (
            0.45 * np.clip(spread_lag[index] / 100.0, -1.0, 1.0)
            + 0.35 * np.clip(load_lag[index] / 1000.0, -1.0, 1.0)
            + 0.20 * np.clip(renewable_lag[index] / 1000.0, -1.0, 1.0)
            - 0.15 * np.clip(previous_delta / max(smooth_limit, 1.0), -1.0, 1.0)
        )
        target_delta = float(np.tanh(raw_signal) * bandwidth[index])
        delta = float(np.clip(target_delta, previous_delta - smooth_limit, previous_delta + smooth_limit))
        deltas.append(delta)
        previous_delta = delta

    delta_q = np.asarray(deltas, dtype=float)
    q_spot_raw = q_base + delta_q
    q_spot = np.clip(q_spot_raw, 0.0, None) if non_negative_spot else q_spot_raw
    trace = frame.copy()
    trace["q_lt_hourly"] = q_lt
    trace["spot_need"] = q_base
    trace["q_base"] = q_base
    trace["bandwidth_mwh"] = bandwidth
    trace["delta_q_target"] = delta_q
    trace["delta_q_after_smoothing"] = delta_q
    trace["smoothing_mode"] = "causal_param_policy"
    trace["lower_band"] = q_base - bandwidth
    trace["upper_band"] = q_base + bandwidth
    trace["q_spot_raw"] = q_spot_raw
    trace["q_spot_band_clipped"] = q_spot_raw
    trace["q_spot"] = q_spot
    trace["delta_q"] = q_spot - q_base
    trace["a_t"] = np.divide(trace["delta_q"], np.maximum(bandwidth, 1e-9))
    trace["signal_spread"] = np.clip(spread_lag / 100.0, -1.0, 1.0)
    trace["signal_load_dev"] = np.clip(load_lag / 1000.0, -1.0, 1.0)
    trace["signal_renewable_dev"] = np.clip(renewable_lag / 1000.0, -1.0, 1.0)
    trace["signal_composite"] = trace[["signal_spread", "signal_load_dev", "signal_renewable_dev"]].mean(axis=1)
    trace["clipped_by_bound"] = 0
    trace["clipped_non_negative"] = (q_spot_raw < 0.0).astype(int)
    trace["soft_clipped"] = 0
    if "net_load_id" in trace.columns:
        trace["hedge_error_abs"] = np.abs(trace["q_spot"] - (trace["net_load_id"].to_numpy(dtype=float) - q_lt))
    else:
        trace["hedge_error_abs"] = np.nan
    return trace, _decision_audit(frame)
