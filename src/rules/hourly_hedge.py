from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _signal(values: np.ndarray, threshold: float) -> np.ndarray:
    scaled = values / max(threshold, 1e-6)
    return np.clip(scaled, -1.0, 1.0)


def apply_hourly_hedge_rule(
    hourly_frame: pd.DataFrame,
    q_lt_hourly: pd.Series,
    exposure_bandwidth: float,
    policy_state: pd.Series,
    rules_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = hourly_frame.copy().reset_index(drop=True)
    q_lt = np.clip(np.asarray(q_lt_hourly, dtype=float), 0.0, None)
    net_load_da = frame["net_load_da"].to_numpy(dtype=float, copy=False)
    price_spread = frame["price_spread"].to_numpy(dtype=float, copy=False)
    load_dev = frame["load_dev"].to_numpy(dtype=float, copy=False)
    renewable_dev = frame["renewable_dev"].to_numpy(dtype=float, copy=False)

    spot_need = np.clip(net_load_da - q_lt, 0.0, None)
    q_base = spot_need

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

    ancillary_tight = float(policy_state.get("ancillary_freq_reserve_tight", 0.0))
    peak_shaving_pause = float(policy_state.get("ancillary_peak_shaving_pause", 0.0))
    ancillary_multiplier = 1.0
    if ancillary_tight >= 0.5:
        ancillary_multiplier *= float(rules_config["ancillary_tight_multiplier"])
    if peak_shaving_pause >= 0.5:
        ancillary_multiplier *= float(rules_config["peak_shaving_pause_multiplier"])

    volatility_multiplier = np.clip(
        1.0 - float(rules_config["price_spike_shrink"]) * np.abs(composite),
        0.1,
        1.0,
    )
    exposure = float(np.clip(exposure_bandwidth, 0.0, 1.0))
    if exposure <= 0.0:
        bandwidth_mwh = np.zeros_like(q_base, dtype=float)
    else:
        base_band = q_base * float(rules_config["band_base_multiplier"]) * exposure
        bandwidth_mwh = np.clip(
            base_band * ancillary_multiplier * volatility_multiplier,
            float(rules_config["band_floor_mwh"]),
            q_base * float(rules_config["band_cap_ratio"]),
        )
        bandwidth_mwh = np.clip(np.minimum(bandwidth_mwh, q_base), 0.0, None)

    target_delta = composite * bandwidth_mwh
    smooth_limit = float(rules_config["gamma_max"])
    smoothing_mode = str(rules_config.get("smoothing_mode", "hard_clip"))
    delta_values = np.empty_like(target_delta)
    smooth_hits = 0
    previous_delta = 0.0
    for index, value in enumerate(target_delta):
        if smoothing_mode == "soft_clip":
            clipped_value = float(previous_delta + smooth_limit * np.tanh((value - previous_delta) / max(smooth_limit, 1e-6)))
        else:
            clipped_value = float(np.clip(value, previous_delta - smooth_limit, previous_delta + smooth_limit))
        if not np.isclose(clipped_value, value, atol=1e-9):
            smooth_hits += 1
        delta_values[index] = clipped_value
        previous_delta = clipped_value

    lower_band = np.clip(q_base - bandwidth_mwh, 0.0, None)
    upper_band = q_base + bandwidth_mwh
    q_spot_raw = q_base + delta_values
    q_spot_band_clipped = np.minimum(np.maximum(q_spot_raw, lower_band), upper_band)
    clipped_by_bound = (~np.isclose(q_spot_raw, q_spot_band_clipped)).astype(int)
    if bool(rules_config["non_negative_clip"]):
        q_spot = np.clip(q_spot_band_clipped, 0.0, None)
    else:
        q_spot = q_spot_band_clipped
    clipped_non_negative = (q_spot_band_clipped < 0.0).astype(int)
    delta_q = q_spot - q_base
    soft_clipped = (~np.isclose(target_delta, delta_values, atol=1e-9)).astype(int)
    hedge_error_abs = np.abs(q_spot - spot_need)

    derived = pd.DataFrame(
        {
            "q_lt_hourly": q_lt,
            "spot_need": spot_need,
            "q_base": q_base,
            "signal_spread": spread_signal,
            "signal_load_dev": load_signal,
            "signal_renewable_dev": renewable_signal,
            "signal_composite": composite,
            "bandwidth_mwh": bandwidth_mwh,
            "bandwidth_multiplier": np.full(len(frame), ancillary_multiplier, dtype=float),
            "delta_q_target": target_delta,
            "delta_q_after_smoothing": delta_values,
            "smoothing_mode": np.full(len(frame), smoothing_mode, dtype=object),
            "lower_band": lower_band,
            "upper_band": upper_band,
            "q_spot_raw": q_spot_raw,
            "q_spot_band_clipped": q_spot_band_clipped,
            "clipped_by_bound": clipped_by_bound,
            "q_spot": q_spot,
            "clipped_non_negative": clipped_non_negative,
            "delta_q": delta_q,
            "a_t_raw": composite,
            "a_t": composite,
            "soft_clipped": soft_clipped,
            "hedge_error_abs": hedge_error_abs,
        },
        index=frame.index,
    )
    frame = pd.concat([frame, derived], axis=1)

    stats = {
        "bound_clip_count": int(clipped_by_bound.sum()),
        "smooth_clip_count": int(smooth_hits),
        "soft_clip_count": int(smooth_hits if smoothing_mode == "soft_clip" else 0),
        "non_negative_clip_count": int(clipped_non_negative.sum()),
    }
    return frame, stats
