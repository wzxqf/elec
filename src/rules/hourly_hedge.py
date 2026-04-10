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
    hedge_intensity: float,
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

    raw_a = hedge_intensity * composite
    clipped_a = raw_a.clip(float(rules_config["a_min"]), float(rules_config["a_max"]))
    frame["a_t_raw"] = raw_a
    frame["a_t"] = clipped_a
    frame["clipped_by_bound"] = (raw_a != clipped_a).astype(int)

    raw_delta = frame["a_t"] * frame["q_base"]
    smooth_limit = float(rules_config["gamma_max"])
    delta_values: list[float] = []
    smooth_hits = 0
    previous_delta = 0.0
    for value in raw_delta.tolist():
        clipped_value = float(np.clip(value, previous_delta - smooth_limit, previous_delta + smooth_limit))
        if not np.isclose(clipped_value, value):
            smooth_hits += 1
        delta_values.append(clipped_value)
        previous_delta = clipped_value

    frame["delta_q"] = delta_values
    frame["smoothed"] = [
        int(not np.isclose(raw, clipped))
        for raw, clipped in zip(raw_delta.tolist(), frame["delta_q"].tolist(), strict=False)
    ]
    frame["q_spot_raw"] = frame["q_base"] + frame["delta_q"]
    if bool(rules_config["non_negative_clip"]):
        frame["q_spot"] = frame["q_spot_raw"].clip(lower=0.0)
    else:
        frame["q_spot"] = frame["q_spot_raw"]
    frame["clipped_non_negative"] = (frame["q_spot_raw"] < 0.0).astype(int)
    frame["hedge_error_abs"] = (frame["q_spot"] - frame["spot_need"]).abs()

    stats = {
        "bound_clip_count": int(frame["clipped_by_bound"].sum()),
        "smooth_clip_count": int(smooth_hits),
        "non_negative_clip_count": int(frame["clipped_non_negative"].sum()),
    }
    return frame, stats
