from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_derived_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    hour_of_day = data["datetime"].dt.hour
    data["net_load_da"] = data["省调负荷_日前"] - data["新能源负荷-总加_日前"]
    data["net_load_id"] = data["省调负荷_日内"] - data["新能源负荷-总加_日内"]
    data["price_spread"] = data["全网统一出清价格_日内"] - data["全网统一出清价格_日前"]
    data["price_spread_abs"] = data["price_spread"].abs()
    data["load_dev"] = data["net_load_id"] - data["net_load_da"]
    data["load_dev_abs"] = data["load_dev"].abs()
    data["load_dev_sign"] = np.sign(data["load_dev"])
    data["wind_dev"] = data["新能源负荷-风电_日内"] - data["新能源负荷-风电_日前"]
    data["solar_dev"] = data["新能源负荷-光伏_日内"] - data["新能源负荷-光伏_日前"]
    data["renewable_ratio_da"] = safe_divide(data["新能源负荷-总加_日前"], data["省调负荷_日前"])
    data["renewable_ratio_id"] = safe_divide(data["新能源负荷-总加_日内"], data["省调负荷_日内"])
    data["renewable_dev"] = data["wind_dev"] + data["solar_dev"]
    data["renewable_dev_abs"] = data["renewable_dev"].abs()
    data["renewable_dev_sign"] = np.sign(data["renewable_dev"])
    data["business_hour_flag"] = hour_of_day.isin(range(8, 21)).astype(float)
    data["peak_hour_flag"] = hour_of_day.isin([8, 9, 10, 11, 18, 19, 20, 21]).astype(float)
    data["valley_hour_flag"] = hour_of_day.isin([0, 1, 2, 3, 4, 5, 6]).astype(float)
    data["business_hour_spread"] = data["price_spread"] * data["business_hour_flag"]
    data["peak_hour_spread"] = data["price_spread"] * data["peak_hour_flag"]
    data["valley_hour_spread"] = data["price_spread"] * data["valley_hour_flag"]
    data["da_price_valid_flag"] = data["全网统一出清价格_日前"].notna().astype(float)
    data["id_price_valid_flag"] = data["全网统一出清价格_日内"].notna().astype(float)
    data["settlement_effective_flag"] = 1.0
    data["hour"] = data["datetime"].dt.floor("h")
    data["week_start"] = data["datetime"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    data["net_load_da_mwh"] = data["net_load_da"] * 0.25
    data["net_load_id_mwh"] = data["net_load_id"] * 0.25

    manifest_rows = [
        {"column": "net_load_da", "source": "派生", "formula": "省调负荷_日前 - 新能源负荷-总加_日前"},
        {"column": "net_load_id", "source": "派生", "formula": "省调负荷_日内 - 新能源负荷-总加_日内"},
        {"column": "price_spread", "source": "派生", "formula": "全网统一出清价格_日内 - 全网统一出清价格_日前"},
        {"column": "price_spread_abs", "source": "派生", "formula": "abs(price_spread)"},
        {"column": "load_dev", "source": "派生", "formula": "net_load_id - net_load_da"},
        {"column": "load_dev_abs", "source": "派生", "formula": "abs(load_dev)"},
        {"column": "load_dev_sign", "source": "派生", "formula": "sign(load_dev)"},
        {"column": "wind_dev", "source": "派生", "formula": "新能源负荷-风电_日内 - 新能源负荷-风电_日前"},
        {"column": "solar_dev", "source": "派生", "formula": "新能源负荷-光伏_日内 - 新能源负荷-光伏_日前"},
        {"column": "renewable_ratio_da", "source": "派生", "formula": "新能源负荷-总加_日前 / 省调负荷_日前"},
        {"column": "renewable_ratio_id", "source": "派生", "formula": "新能源负荷-总加_日内 / 省调负荷_日内"},
        {"column": "renewable_dev", "source": "派生", "formula": "wind_dev + solar_dev"},
        {"column": "renewable_dev_abs", "source": "派生", "formula": "abs(renewable_dev)"},
        {"column": "renewable_dev_sign", "source": "派生", "formula": "sign(renewable_dev)"},
        {"column": "business_hour_flag", "source": "派生", "formula": "hour in [08, 20]"},
        {"column": "peak_hour_flag", "source": "派生", "formula": "hour in {08-11,18-21}"},
        {"column": "valley_hour_flag", "source": "派生", "formula": "hour in {00-06}"},
        {"column": "business_hour_spread", "source": "派生", "formula": "price_spread * business_hour_flag"},
        {"column": "peak_hour_spread", "source": "派生", "formula": "price_spread * peak_hour_flag"},
        {"column": "valley_hour_spread", "source": "派生", "formula": "price_spread * valley_hour_flag"},
        {"column": "da_price_valid_flag", "source": "派生", "formula": "notna(全网统一出清价格_日前)"},
        {"column": "id_price_valid_flag", "source": "派生", "formula": "notna(全网统一出清价格_日内)"},
        {"column": "settlement_effective_flag", "source": "派生", "formula": "1.0"},
        {"column": "net_load_da_mwh", "source": "派生", "formula": "net_load_da * 0.25"},
        {"column": "net_load_id_mwh", "source": "派生", "formula": "net_load_id * 0.25"},
    ]

    return data, pd.DataFrame(manifest_rows)


def aggregate_hourly(frame_15m: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in frame_15m.columns
        if column not in {"datetime", "hour", "week_start"}
    ]
    hourly = frame_15m.groupby("hour", as_index=False)[numeric_columns].mean(numeric_only=True)
    hourly["week_start"] = hourly["hour"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    hourly["week_end"] = hourly["week_start"] + pd.Timedelta(days=6, hours=23)
    hourly["hour_index_in_week"] = (
        ((hourly["hour"] - hourly["week_start"]) / pd.Timedelta(hours=1)).astype(int)
    )
    hourly = hourly.sort_values(["week_start", "hour_index_in_week"]).reset_index(drop=True)
    for column in ["price_spread", "load_dev", "renewable_dev", "price_spread_abs", "renewable_dev_abs"]:
        if column in hourly.columns:
            hourly[f"{column}_lag1"] = hourly.groupby("week_start")[column].shift(1).fillna(0.0)
    return hourly


def _series_stats(series: pd.Series, prefix: str, quantiles: list[float]) -> dict[str, float]:
    values = {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_std": float(series.std(ddof=0)),
        f"{prefix}_max": float(series.max()),
        f"{prefix}_min": float(series.min()),
    }
    for quantile in quantiles:
        values[f"{prefix}_q{int(quantile * 100):02d}"] = float(series.quantile(quantile))
    return values


def build_weekly_features(hourly: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    weeks = sorted(hourly["week_start"].unique())
    previous_by_week = {week: hourly.loc[hourly["week_start"] == week].copy() for week in weeks}
    rows: list[dict[str, Any]] = []

    tracked_columns = {
        "prev_da_price": "全网统一出清价格_日前",
        "prev_id_price": "全网统一出清价格_日内",
        "prev_spread": "price_spread",
        "prev_price_spread_abs": "price_spread_abs",
        "prev_net_load_da": "net_load_da",
        "prev_net_load_id": "net_load_id",
        "prev_load_dev": "load_dev",
        "prev_wind_dev": "wind_dev",
        "prev_solar_dev": "solar_dev",
        "prev_renewable_dev_abs": "renewable_dev_abs",
        "prev_renewable_ratio_da": "renewable_ratio_da",
        "prev_renewable_ratio_id": "renewable_ratio_id",
        "prev_business_hour_spread": "business_hour_spread",
        "prev_peak_hour_spread": "peak_hour_spread",
        "prev_valley_hour_spread": "valley_hour_spread",
        "prev_tieline_da": "联络线总加_日前",
        "prev_hydro_da": "水电出力_日前",
        "prev_nonmarket_da": "非市场化机组出力_日前",
    }

    for index, week in enumerate(weeks):
        if index == 0:
            continue
        previous_week = weeks[index - 1]
        prev_frame = previous_by_week[previous_week]
        record: dict[str, Any] = {"week_start": week}
        for prefix, column in tracked_columns.items():
            record.update(_series_stats(prev_frame[column], prefix, quantiles))
        week_ts = pd.Timestamp(week)
        iso_week = int(week_ts.isocalendar().week)
        record["weekofyear_sin"] = float(np.sin(2 * np.pi * iso_week / 52))
        record["weekofyear_cos"] = float(np.cos(2 * np.pi * iso_week / 52))
        rows.append(record)

    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)
