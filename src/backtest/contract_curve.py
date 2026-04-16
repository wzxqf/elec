from __future__ import annotations

import numpy as np
import pandas as pd


def build_base_24h_profile(hourly: pd.DataFrame) -> pd.Series:
    if "hour" not in hourly.columns:
        raise KeyError("hourly frame must include an 'hour' column.")
    values = hourly.copy()
    values["contract_hour"] = pd.to_datetime(values["hour"]).dt.hour
    if "net_load_da" in values.columns:
        source = values["net_load_da"].clip(lower=0.0)
    else:
        source = pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    profile = source.groupby(values["contract_hour"]).mean().reindex(range(24))
    if profile.isna().any():
        fill = float(profile.dropna().mean()) if profile.notna().any() else 1.0
        profile = profile.fillna(fill)
    mean_value = float(profile.mean())
    if mean_value <= 0.0:
        profile = pd.Series(np.ones(24), index=range(24), dtype=float)
    else:
        profile = profile / mean_value
    profile.index.name = "contract_hour"
    profile.name = "contract_profile_base"
    return profile.astype(float)


def _shape_adjustment(hours: pd.Series, curve_params: np.ndarray, scale: float) -> np.ndarray:
    params = np.asarray(curve_params, dtype=float)
    if params.shape[0] < 8:
        params = np.pad(params, (0, 8 - params.shape[0]))
    hour_values = hours.to_numpy(dtype=float, copy=False)
    features = np.vstack(
        [
            np.ones(len(hours)),
            np.sin(2 * np.pi * hour_values / 24.0),
            np.cos(2 * np.pi * hour_values / 24.0),
            np.sin(4 * np.pi * hour_values / 24.0),
            np.cos(4 * np.pi * hour_values / 24.0),
            np.isin(hour_values, [8, 9, 10, 11, 18, 19, 20, 21]).astype(float),
            np.isin(hour_values, [0, 1, 2, 3, 4, 5]).astype(float),
            ((hour_values >= 12) & (hour_values <= 16)).astype(float),
        ]
    ).T
    return scale * np.tanh(features @ params[:8])


def allocate_weekly_contract_curve(
    hourly: pd.DataFrame,
    q_lt_target: float,
    base_profile: pd.Series | None = None,
    curve_params: np.ndarray | list[float] | None = None,
    adjustment_scale: float = 0.20,
    positive_floor: float = 0.05,
) -> pd.DataFrame:
    frame = hourly.copy().reset_index(drop=True)
    if "hour" not in frame.columns:
        raise KeyError("hourly frame must include an 'hour' column.")
    frame["contract_hour"] = pd.to_datetime(frame["hour"]).dt.hour
    profile = build_base_24h_profile(frame) if base_profile is None else base_profile.astype(float)
    params = np.zeros(8, dtype=float) if curve_params is None else np.asarray(curve_params, dtype=float)
    base_values = frame["contract_hour"].map(profile).astype(float).to_numpy()
    adjustment = _shape_adjustment(frame["contract_hour"], params, adjustment_scale)
    shaped = np.clip(base_values * (1.0 + adjustment), positive_floor, None)
    raw_total = float(shaped.sum())
    normalization = 0.0 if raw_total <= 0.0 else float(q_lt_target) / raw_total
    q_lt_hourly = shaped * normalization
    frame["contract_profile_base"] = base_values
    frame["contract_profile_adjustment"] = adjustment
    frame["contract_profile_final"] = shaped
    frame["curve_normalization_factor"] = normalization
    frame["q_lt_hourly"] = q_lt_hourly
    return frame
