from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_derived_features(frame: pd.DataFrame, policy_events: dict[str, str]) -> pd.DataFrame:
    data = frame.copy()
    data["net_load_da"] = data["省调负荷_日前"] - data["新能源负荷-总加_日前"]
    data["net_load_id"] = data["省调负荷_日内"] - data["新能源负荷-总加_日内"]
    data["price_spread"] = data["全网统一出清价格_日内"] - data["全网统一出清价格_日前"]
    data["load_dev"] = data["net_load_id"] - data["net_load_da"]
    data["wind_dev"] = data["新能源负荷-风电_日内"] - data["新能源负荷-风电_日前"]
    data["solar_dev"] = data["新能源负荷-光伏_日内"] - data["新能源负荷-光伏_日前"]
    data["renewable_ratio_da"] = safe_divide(data["新能源负荷-总加_日前"], data["省调负荷_日前"])
    data["renewable_ratio_id"] = safe_divide(data["新能源负荷-总加_日内"], data["省调负荷_日内"])
    data["renewable_dev"] = data["wind_dev"] + data["solar_dev"]
    data["hour"] = data["datetime"].dt.floor("h")
    data["month"] = data["datetime"].dt.to_period("M").dt.to_timestamp()
    for name, ts in policy_events.items():
        data[name] = (data["datetime"] >= pd.Timestamp(ts)).astype(float)
    return data


def aggregate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in frame.columns if column not in {"datetime", "hour", "month"}]
    hourly = frame.groupby("hour", as_index=False)[numeric_columns].mean(numeric_only=True)
    hourly["month"] = hourly["hour"].dt.to_period("M").dt.to_timestamp()
    return hourly


def _stats(prefix: str, series: pd.Series) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_std": float(series.std(ddof=0)),
        f"{prefix}_q25": float(series.quantile(0.25)),
        f"{prefix}_q75": float(series.quantile(0.75)),
        f"{prefix}_max": float(series.max()),
    }


def build_monthly_feature_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    months = sorted(hourly["month"].unique())
    rows: list[dict[str, Any]] = []
    previous_by_month = {month: hourly.loc[hourly["month"] == month].copy() for month in months}

    for index, month in enumerate(months):
        if index == 0:
            continue
        prev_month = months[index - 1]
        prev = previous_by_month[prev_month]
        record: dict[str, Any] = {"month": month}
        record.update(_stats("prev_da_price", prev["全网统一出清价格_日前"]))
        record.update(_stats("prev_id_price", prev["全网统一出清价格_日内"]))
        record.update(_stats("prev_spread", prev["price_spread"]))
        record.update(_stats("prev_net_load_da", prev["net_load_da"]))
        record.update(_stats("prev_net_load_id", prev["net_load_id"]))
        record.update(_stats("prev_wind_dev", prev["wind_dev"]))
        record.update(_stats("prev_solar_dev", prev["solar_dev"]))
        record.update(_stats("prev_renewable_ratio_da", prev["renewable_ratio_da"]))
        record.update(_stats("prev_renewable_ratio_id", prev["renewable_ratio_id"]))
        record.update(_stats("prev_tieline_da", prev["联络线总加_日前"]))
        record.update(_stats("prev_hydro_da", prev["水电出力_日前"]))
        record.update(_stats("prev_nonmarket_da", prev["非市场化机组出力_日前"]))
        record["month_sin"] = float(np.sin((month.month - 1) / 12 * 2 * np.pi))
        record["month_cos"] = float(np.cos((month.month - 1) / 12 * 2 * np.pi))
        for policy_column in [col for col in hourly.columns if col.startswith("policy_event_")]:
            record[policy_column] = float(hourly.loc[hourly["month"] == month, policy_column].max())
        rows.append(record)

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_monthly_metadata(frame_15m: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    quarter = frame_15m.copy()
    quarter["energy_da_mwh"] = quarter["net_load_da"] * 0.25
    quarter["energy_id_mwh"] = quarter["net_load_id"] * 0.25

    month_energy = quarter.groupby("month", as_index=False).agg(
        forecast_monthly_net_demand_mwh=("energy_da_mwh", "sum"),
        actual_monthly_net_demand_mwh=("energy_id_mwh", "sum"),
        da_price_mean=("全网统一出清价格_日前", "mean"),
        id_price_mean=("全网统一出清价格_日内", "mean"),
    )
    month_energy = month_energy.sort_values("month").reset_index(drop=True)
    month_energy["lt_price_m"] = month_energy["da_price_mean"].shift(1)
    month_energy["lt_price_source"] = np.where(
        month_energy["lt_price_m"].notna(),
        "estimated_prev_month_da_mean",
        "warmup_unavailable",
    )

    hour_stats = hourly.groupby("month", as_index=False).agg(
        hourly_net_load_da_vol=("net_load_da", lambda x: float(x.std(ddof=0))),
        hourly_spread_vol=("price_spread", lambda x: float(x.std(ddof=0))),
    )
    return month_energy.merge(hour_stats, on="month", how="left")


def prepare_datasets(frame: pd.DataFrame, policy_events: dict[str, str]) -> dict[str, Any]:
    frame_15m = add_derived_features(frame, policy_events)
    hourly = aggregate_hourly(frame_15m)
    monthly_features = build_monthly_feature_frame(hourly)
    monthly_metadata = build_monthly_metadata(frame_15m, hourly)
    return {
        "quarter": frame_15m,
        "hourly": hourly,
        "monthly_features": monthly_features,
        "monthly_metadata": monthly_metadata,
    }
