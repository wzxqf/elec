from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_engineering import add_derived_features, aggregate_hourly, build_weekly_features


def _is_partial_week(frame: pd.DataFrame, datetime_column: str) -> bool:
    start = frame[datetime_column].min()
    end = frame[datetime_column].max()
    expected_points = 7 * 24 * 4
    is_full_span = (start.weekday() == 0 and start.hour == 0 and start.minute == 0) and (
        end.weekday() == 6 and end.hour == 23 and end.minute == 45
    )
    return (len(frame) != expected_points) or (not is_full_span)


def build_weekly_bundle(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    frame_15m, feature_manifest = add_derived_features(frame)
    hourly = aggregate_hourly(frame_15m)

    weekly_meta_rows: list[dict[str, Any]] = []
    for week_start, week_quarter in frame_15m.groupby("week_start"):
        week_hourly = hourly.loc[hourly["week_start"] == week_start].copy()
        da_cost_proxy = week_quarter["net_load_da_mwh"] * week_quarter["全网统一出清价格_日前"]
        weekly_meta_rows.append(
            {
                "week_start": pd.Timestamp(week_start),
                "week_end": pd.Timestamp(week_start) + pd.Timedelta(days=6, hours=23, minutes=45),
                "is_partial_week": bool(_is_partial_week(week_quarter, "datetime")),
                "hour_count": int(len(week_hourly)),
                "quarter_count": int(len(week_quarter)),
                "forecast_weekly_net_demand_mwh": float(week_hourly["net_load_da"].sum()),
                "actual_weekly_net_demand_mwh": float(week_hourly["net_load_id"].sum()),
                "da_price_mean": float(week_hourly["全网统一出清价格_日前"].mean()),
                "da_price_std": float(week_hourly["全网统一出清价格_日前"].std(ddof=0)),
                "id_price_mean": float(week_hourly["全网统一出清价格_日内"].mean()),
                "id_price_std": float(week_hourly["全网统一出清价格_日内"].std(ddof=0)),
                "spread_mean": float(week_hourly["price_spread"].mean()),
                "spread_std": float(week_hourly["price_spread"].std(ddof=0)),
                "load_dev_std": float(week_hourly["load_dev"].std(ddof=0)),
                "renewable_dev_std": float(week_hourly["renewable_dev"].std(ddof=0)),
                "proxy_da_cost_mean": float(da_cost_proxy.mean()),
                "proxy_da_cost_cvar95": float(da_cost_proxy[da_cost_proxy >= da_cost_proxy.quantile(0.95)].mean()),
            }
        )

    weekly_metadata = pd.DataFrame(weekly_meta_rows).sort_values("week_start").reset_index(drop=True)
    weekly_metadata["lt_price_w"] = weekly_metadata["da_price_mean"].shift(1)
    weekly_metadata["lt_price_source"] = np.where(
        weekly_metadata["lt_price_w"].notna(),
        "estimated_prev_week_da_mean",
        config["lt_price"]["warmup_label"],
    )

    weekly_features = build_weekly_features(hourly, list(config["feature_quantiles"]))
    weekly_features = weekly_features.merge(
        weekly_metadata[
            [
                "week_start",
                "is_partial_week",
                "forecast_weekly_net_demand_mwh",
                "actual_weekly_net_demand_mwh",
                "lt_price_w",
                "lt_price_source",
            ]
        ],
        on="week_start",
        how="left",
    )
    weekly_features["lt_price_w"] = weekly_features["lt_price_w"].fillna(weekly_features["prev_da_price_mean"])

    return {
        "quarter": frame_15m,
        "hourly": hourly,
        "weekly_features": weekly_features,
        "weekly_metadata": weekly_metadata,
        "feature_manifest": feature_manifest,
    }
