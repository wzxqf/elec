from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _signal(series: pd.Series, threshold: float) -> pd.Series:
    scaled = series / max(threshold, 1e-6)
    return scaled.clip(-1.0, 1.0)


def apply_hourly_hedge_rule(
    hourly_frame: pd.DataFrame,
    q_lt_hourly: pd.Series,
    exposure_bandwidth: float,
    policy_state: pd.Series,
    rules_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = hourly_frame.copy().reset_index(drop=True)
    q_lt = pd.Series(np.asarray(q_lt_hourly, dtype=float), index=frame.index, dtype=float).clip(lower=0.0)
    frame["q_lt_hourly"] = q_lt
    frame["spot_need"] = (frame["net_load_da"] - frame["q_lt_hourly"]).clip(lower=0.0)
    frame["q_base"] = frame["spot_need"]

    spread_signal = _signal(frame["price_spread"], float(rules_config["spread_threshold"]))
    load_signal = _signal(frame["load_dev"], float(rules_config["load_dev_threshold"]))
    renewable_signal = _signal(-frame["renewable_dev"], float(rules_config["renewable_dev_threshold"]))
    composite = (
        float(rules_config["price_spread_weight"]) * spread_signal
        + float(rules_config["load_dev_weight"]) * load_signal
        + float(rules_config["renewable_dev_weight"]) * renewable_signal
    ).clip(-float(rules_config["signal_clip"]), float(rules_config["signal_clip"]))

    frame["signal_spread"] = spread_signal
    frame["signal_load_dev"] = load_signal
    frame["signal_renewable_dev"] = renewable_signal
    frame["signal_composite"] = composite

    ancillary_tight = float(policy_state.get("ancillary_freq_reserve_tight", 0.0))
    peak_shaving_pause = float(policy_state.get("ancillary_peak_shaving_pause", 0.0))
    ancillary_multiplier = 1.0
    if ancillary_tight >= 0.5:
        ancillary_multiplier *= float(rules_config["ancillary_tight_multiplier"])
    if peak_shaving_pause >= 0.5:
        ancillary_multiplier *= float(rules_config["peak_shaving_pause_multiplier"])

    volatility_multiplier = 1.0 - float(rules_config["price_spike_shrink"]) * composite.abs()
    volatility_multiplier = volatility_multiplier.clip(lower=0.1, upper=1.0)
    base_band = frame["q_base"] * float(rules_config["band_base_multiplier"]) * float(np.clip(exposure_bandwidth, 0.0, 1.0))
    frame["bandwidth_mwh"] = (
        base_band * ancillary_multiplier * volatility_multiplier
    ).clip(
        lower=float(rules_config["band_floor_mwh"]),
        upper=frame["q_base"] * float(rules_config["band_cap_ratio"]),
    )
    frame["bandwidth_mwh"] = frame[["bandwidth_mwh", "q_base"]].min(axis=1).clip(lower=0.0)
    frame["bandwidth_multiplier"] = ancillary_multiplier

    target_delta = composite * frame["bandwidth_mwh"]
    smooth_limit = float(rules_config["gamma_max"])
    smoothing_mode = str(rules_config.get("smoothing_mode", "hard_clip"))
    delta_values: list[float] = []
    smooth_hits = 0
    previous_delta = 0.0
    for value in target_delta.tolist():
        if smoothing_mode == "soft_clip":
            clipped_value = float(previous_delta + smooth_limit * np.tanh((value - previous_delta) / max(smooth_limit, 1e-6)))
        else:
            clipped_value = float(np.clip(value, previous_delta - smooth_limit, previous_delta + smooth_limit))
        if not np.isclose(clipped_value, value, atol=1e-9):
            smooth_hits += 1
        delta_values.append(clipped_value)
        previous_delta = clipped_value

    frame["delta_q_target"] = target_delta
    frame["delta_q_after_smoothing"] = delta_values
    frame["smoothing_mode"] = smoothing_mode
    frame["lower_band"] = (frame["q_base"] - frame["bandwidth_mwh"]).clip(lower=0.0)
    frame["upper_band"] = frame["q_base"] + frame["bandwidth_mwh"]
    frame["q_spot_raw"] = frame["q_base"] + frame["delta_q_after_smoothing"]
    frame["q_spot_band_clipped"] = frame["q_spot_raw"].clip(lower=frame["lower_band"], upper=frame["upper_band"])
    frame["clipped_by_bound"] = (~np.isclose(frame["q_spot_raw"], frame["q_spot_band_clipped"])).astype(int)
    frame["q_spot"] = (
        frame["q_spot_band_clipped"].clip(lower=0.0) if bool(rules_config["non_negative_clip"]) else frame["q_spot_band_clipped"]
    )
    frame["clipped_non_negative"] = (frame["q_spot_band_clipped"] < 0.0).astype(int)
    frame["delta_q"] = frame["q_spot"] - frame["q_base"]
    frame["a_t_raw"] = composite
    frame["a_t"] = composite
    frame["soft_clipped"] = [
        int(not np.isclose(raw, clipped))
        for raw, clipped in zip(target_delta.tolist(), frame["delta_q_after_smoothing"].tolist(), strict=False)
    ]
    frame["hedge_error_abs"] = (frame["q_spot"] - frame["spot_need"]).abs()

    stats = {
        "bound_clip_count": int(frame["clipped_by_bound"].sum()),
        "smooth_clip_count": int(smooth_hits),
        "soft_clip_count": int(smooth_hits if smoothing_mode == "soft_clip" else 0),
        "non_negative_clip_count": int(frame["clipped_non_negative"].sum()),
    }
    return frame, stats
