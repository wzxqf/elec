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


def _resolve_lt_price_proxy_fields(
    week_start: pd.Timestamp,
    lt_price_value: float | None,
    config: dict[str, Any],
) -> tuple[str, float, float]:
    lt_cfg = config.get("lt_price", {})
    if pd.isna(lt_price_value):
        return str(lt_cfg.get("warmup_label", "warmup_unavailable")), 0.0, 0.0
    linked_effective_start = lt_cfg.get("linked_effective_start")
    if linked_effective_start is not None and pd.Timestamp(week_start) >= pd.Timestamp(linked_effective_start):
        return "linked_mix_40da_60id", 0.4, 0.6
    return "prev_week_da_proxy", 1.0, 0.0


def build_weekly_bundle(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    frame_15m, feature_manifest = add_derived_features(frame)
    hourly = aggregate_hourly(frame_15m)
    analysis_cfg = config.get("analysis_v035", {})
    price_spike_threshold = float(analysis_cfg.get("price_spike_zscore_threshold", 2.5))
    extreme_event_threshold = float(analysis_cfg.get("extreme_event_std_threshold", 2.0))
    global_id_price_std = float(hourly["全网统一出清价格_日内"].std(ddof=0)) if not hourly.empty else 0.0
    global_load_dev_std = float(hourly["load_dev"].std(ddof=0)) if not hourly.empty else 0.0
    global_renewable_dev_std = float(hourly["renewable_dev"].std(ddof=0)) if not hourly.empty else 0.0

    weekly_meta_rows: list[dict[str, Any]] = []
    for week_start, week_quarter in frame_15m.groupby("week_start"):
        week_hourly = hourly.loc[hourly["week_start"] == week_start].copy()
        da_cost_proxy = week_quarter["net_load_da_mwh"] * week_quarter["全网统一出清价格_日前"]
        da_prices = week_hourly["全网统一出清价格_日前"]
        id_prices = week_hourly["全网统一出清价格_日内"]
        da_id_cross_corr = 0.0
        if len(da_prices) > 1 and float(da_prices.std(ddof=0)) > 0.0 and float(id_prices.std(ddof=0)) > 0.0:
            da_id_cross_corr = float(da_prices.corr(id_prices))
        week_id_price_std = float(id_prices.std(ddof=0))
        week_load_dev_std = float(week_hourly["load_dev"].std(ddof=0))
        week_renewable_dev_std = float(week_hourly["renewable_dev"].std(ddof=0))
        id_price_z = week_id_price_std / max(global_id_price_std, 1.0e-6)
        load_dev_z = week_load_dev_std / max(global_load_dev_std, 1.0e-6)
        renewable_dev_z = week_renewable_dev_std / max(global_renewable_dev_std, 1.0e-6)
        price_spike_flag = float(id_price_z >= price_spike_threshold)
        extreme_event_flag = float(
            price_spike_flag > 0.0 or load_dev_z >= extreme_event_threshold or renewable_dev_z >= extreme_event_threshold
        )
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
                "da_id_cross_corr_w": da_id_cross_corr,
                "extreme_price_spike_flag_w": price_spike_flag,
                "extreme_event_flag_w": extreme_event_flag,
                "proxy_da_cost_mean": float(da_cost_proxy.mean()),
                "proxy_da_cost_cvar95": float(da_cost_proxy[da_cost_proxy >= da_cost_proxy.quantile(0.95)].mean()),
            }
        )

    weekly_metadata = pd.DataFrame(weekly_meta_rows).sort_values("week_start").reset_index(drop=True)
    weekly_metadata["lt_price_w"] = weekly_metadata["da_price_mean"].shift(1)
    lt_proxy_fields = weekly_metadata.apply(
        lambda row: _resolve_lt_price_proxy_fields(
            week_start=pd.Timestamp(row["week_start"]),
            lt_price_value=row.get("lt_price_w"),
            config=config,
        ),
        axis=1,
        result_type="expand",
    )
    lt_proxy_fields.columns = ["lt_price_source", "lt_price_fixed_ratio", "lt_price_linked_ratio"]
    weekly_metadata[["lt_price_source", "lt_price_fixed_ratio", "lt_price_linked_ratio"]] = lt_proxy_fields

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
                "lt_price_fixed_ratio",
                "lt_price_linked_ratio",
                "da_id_cross_corr_w",
                "extreme_price_spike_flag_w",
                "extreme_event_flag_w",
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
