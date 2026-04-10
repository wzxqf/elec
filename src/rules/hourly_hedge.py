from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def compute_hourly_hedge_adjustment(
    hourly_frame: pd.DataFrame,
    residual_exposure_mwh: pd.Series,
    hedge_intensity: float,
    rule_config: dict,
    market_vol_scale: float = 1.0,
    forecast_error_scale: float = 1.0,
    price_cap_multiplier: float = 1.0,
    hedge_intensity_scale: float = 1.0,
) -> pd.Series:
    spread_signal = zscore(hourly_frame["price_spread"] * market_vol_scale)
    load_signal = zscore(hourly_frame["load_dev"] * forecast_error_scale)
    renewable_signal = zscore(hourly_frame["renewable_dev"] * forecast_error_scale)

    raw_signal = (
        rule_config["price_spread_weight"] * spread_signal
        + rule_config["load_dev_weight"] * load_signal
        + rule_config["renewable_dev_weight"] * renewable_signal
    )
    clipped_signal = np.clip(raw_signal, -1.0, 1.0)

    price_threshold = hourly_frame["全网统一出清价格_日内"].quantile(rule_config["price_gate_quantile"])
    price_gate = np.where(
        hourly_frame["全网统一出清价格_日内"] > price_threshold * price_cap_multiplier,
        rule_config["price_gate_penalty"],
        1.0,
    )

    max_adjust_share = rule_config["max_adjust_share"]
    adjustment = (
        clipped_signal
        * np.clip(hedge_intensity, 0.0, 1.0)
        * hedge_intensity_scale
        * max_adjust_share
        * residual_exposure_mwh
        * price_gate
    )
    return pd.Series(adjustment, index=hourly_frame.index)
