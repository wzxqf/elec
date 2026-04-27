from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch


CORE_HOURLY_FEATURE_COLUMNS = [
    "net_load_da",
    "net_load_id",
    "price_spread",
    "load_dev",
    "renewable_dev",
]

EXTENDED_HOURLY_FEATURE_COLUMNS = [
    "price_spread_abs",
    "price_spread_lag1",
    "price_spread_abs_lag1",
    "load_dev_abs",
    "load_dev_lag1",
    "load_dev_sign",
    "renewable_dev_abs",
    "renewable_dev_lag1",
    "renewable_dev_abs_lag1",
    "renewable_dev_sign",
    "business_hour_flag",
    "peak_hour_flag",
    "valley_hour_flag",
    "business_hour_spread",
    "peak_hour_spread",
    "valley_hour_spread",
    "da_price_valid_flag",
    "id_price_valid_flag",
    "settlement_effective_flag",
]

HOURLY_FEATURE_COLUMNS = CORE_HOURLY_FEATURE_COLUMNS + EXTENDED_HOURLY_FEATURE_COLUMNS

QUARTER_FEATURE_COLUMNS = ["全网统一出清价格_日前", "全网统一出清价格_日内"]

LAGGED_HOURLY_FEATURE_SOURCES = {
    "price_spread_lag1": "price_spread",
    "price_spread_abs_lag1": "price_spread_abs",
    "load_dev_lag1": "load_dev",
    "renewable_dev_lag1": "renewable_dev",
    "renewable_dev_abs_lag1": "renewable_dev_abs",
}


@dataclass(frozen=True)
class TrainingTensorBundle:
    device: str
    week_index: pd.Index
    weekly_feature_columns: list[str]
    policy_columns: list[str]
    hourly_feature_columns: list[str]
    quarter_feature_columns: list[str]
    weekly_bound_columns: list[str]
    hourly_bound_columns: list[str]
    weekly_bound_reason_codes: list[str]
    weekly_settlement_modes: list[str]
    weekly_feature_tensor: torch.Tensor
    policy_tensor: torch.Tensor
    hourly_tensor: torch.Tensor
    quarter_price_tensor: torch.Tensor
    weekly_bound_tensor: torch.Tensor
    hourly_bound_tensor: torch.Tensor
    forecast_weekly_load: torch.Tensor
    actual_weekly_load: torch.Tensor
    lt_weekly_price: torch.Tensor
    hourly_valid_mask: torch.Tensor
    quarter_valid_mask: torch.Tensor


def _resolve_device(device: str) -> str:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _ensure_hourly_lag_features(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    if frame.empty or "week_start" not in frame.columns:
        return frame
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    sort_columns = ["week_start"]
    if "hour_index" in frame.columns:
        sort_columns.append("hour_index")
    elif "hour_index_in_week" in frame.columns:
        sort_columns.append("hour_index_in_week")
    elif "hour" in frame.columns:
        sort_columns.append("hour")
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    for lag_column, source_column in LAGGED_HOURLY_FEATURE_SOURCES.items():
        if lag_column not in frame.columns and source_column in frame.columns:
            frame[lag_column] = frame.groupby("week_start")[source_column].shift(1).fillna(0.0)
    return frame


def _frame_by_week(frame: pd.DataFrame, week_index: pd.Index, columns: list[str], index_column: str) -> tuple[torch.Tensor, torch.Tensor]:
    if frame.empty:
        empty = torch.zeros((len(week_index), 0, len(columns)), dtype=torch.float32)
        mask = torch.zeros((len(week_index), 0), dtype=torch.bool)
        return empty, mask

    frame = frame.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    if index_column not in frame.columns:
        if index_column == "hour_index":
            fallback = "hour_index_in_week"
        elif index_column == "interval_index":
            fallback = None
        else:
            fallback = None
        if fallback and fallback in frame.columns:
            frame[index_column] = frame[fallback]
        elif index_column == "interval_index":
            frame[index_column] = frame.groupby("week_start").cumcount()
        else:
            raise KeyError(index_column)
    frame[index_column] = pd.to_numeric(frame[index_column], errors="raise").astype(int)
    max_index = int(frame[index_column].max()) + 1
    tensor = torch.zeros((len(week_index), max_index, len(columns)), dtype=torch.float32)
    mask = torch.zeros((len(week_index), max_index), dtype=torch.bool)
    week_lookup = {pd.Timestamp(week): idx for idx, week in enumerate(week_index)}

    for row in frame.itertuples(index=False):
        week_pos = week_lookup[pd.Timestamp(row.week_start)]
        item_pos = int(getattr(row, index_column))
        values = [float(getattr(row, column)) for column in columns]
        tensor[week_pos, item_pos] = torch.tensor(values, dtype=torch.float32)
        mask[week_pos, item_pos] = True
    return tensor, mask


def compile_training_tensor_bundle(bundle: dict[str, Any], device: str = "cpu") -> TrainingTensorBundle:
    resolved_device = _resolve_device(device)
    weekly_features = bundle["weekly_features"].copy()
    weekly_metadata = bundle["weekly_metadata"].copy()
    policy_state_trace = bundle["policy_state_trace"].copy()
    hourly = _ensure_hourly_lag_features(bundle["hourly"].copy())
    quarter = bundle["quarter"].copy()

    hourly_feature_columns = list(CORE_HOURLY_FEATURE_COLUMNS)
    hourly_feature_columns.extend([column for column in EXTENDED_HOURLY_FEATURE_COLUMNS if column in hourly.columns])

    for column in hourly_feature_columns:
        if column not in hourly.columns:
            hourly[column] = 0.0

    weekly_features["week_start"] = pd.to_datetime(weekly_features["week_start"])
    weekly_metadata["week_start"] = pd.to_datetime(weekly_metadata["week_start"])
    policy_state_trace["week_start"] = pd.to_datetime(policy_state_trace["week_start"])

    week_index = pd.Index(sorted(weekly_metadata["week_start"].tolist()))
    agent_columns = list(bundle.get("agent_feature_columns", []))
    policy_columns = [
        column
        for column in policy_state_trace.columns
        if column != "week_start" and pd.api.types.is_numeric_dtype(policy_state_trace[column])
    ]

    weekly_feature_tensor = torch.tensor(
        weekly_features.set_index("week_start").reindex(week_index).fillna(0.0)[agent_columns].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    policy_tensor = torch.tensor(
        policy_state_trace.set_index("week_start").reindex(week_index).fillna(0.0)[policy_columns].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    metadata_frame = weekly_metadata.set_index("week_start").loc[week_index]
    forecast_weekly_load = torch.tensor(
        metadata_frame["forecast_weekly_net_demand_mwh"].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    actual_weekly_load = torch.tensor(
        metadata_frame["actual_weekly_net_demand_mwh"].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    lt_weekly_price = torch.tensor(
        metadata_frame.get("lt_price_w", pd.Series(0.0, index=metadata_frame.index)).fillna(0.0).to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    hourly_tensor, hourly_valid_mask = _frame_by_week(
        hourly,
        week_index,
        columns=hourly_feature_columns,
        index_column="hour_index",
    )
    quarter_price_tensor, quarter_valid_mask = _frame_by_week(
        quarter,
        week_index,
        columns=QUARTER_FEATURE_COLUMNS,
        index_column="interval_index",
    )
    default_weekly_bounds = pd.DataFrame(
        {
            "week_start": week_index,
            "contract_adjustment_ratio_min": [-1.0] * len(week_index),
            "contract_adjustment_ratio_max": [1.0] * len(week_index),
            "exposure_band_ratio_min": [0.0] * len(week_index),
            "exposure_band_ratio_max": [1.0] * len(week_index),
            "max_hourly_hedge_share": [1.0] * len(week_index),
            "max_hourly_ramp_share": [1.0] * len(week_index),
            "non_negative_position_required": [1.0] * len(week_index),
            "feasible_domain_triggered": [0.0] * len(week_index),
            "bound_reason_code": ["default"] * len(week_index),
        }
    )
    weekly_bound_columns = [
        "contract_adjustment_ratio_min",
        "contract_adjustment_ratio_max",
        "exposure_band_ratio_min",
        "exposure_band_ratio_max",
        "max_hourly_hedge_share",
        "max_hourly_ramp_share",
        "non_negative_position_required",
        "feasible_domain_triggered",
    ]
    hourly_bound_columns = ["max_hourly_hedge_share", "max_hourly_ramp_share"]
    feasible_domain = bundle.get("feasible_domain")
    if feasible_domain is not None:
        weekly_bounds = feasible_domain.weekly_bounds.copy()
        weekly_bounds["week_start"] = pd.to_datetime(weekly_bounds["week_start"])
        settlement_semantics = feasible_domain.settlement_semantics.copy()
        settlement_semantics["week_start"] = pd.to_datetime(settlement_semantics["week_start"])
        hourly_bounds_frame = feasible_domain.hourly_bounds.copy()
        hourly_bounds_frame["week_start"] = pd.to_datetime(hourly_bounds_frame["week_start"])
    else:
        weekly_bounds = default_weekly_bounds
        settlement_semantics = pd.DataFrame({"week_start": week_index, "settlement_mode": ["previous_week_da_proxy"] * len(week_index)})
        hourly_rows: list[dict[str, float | pd.Timestamp | int]] = []
        for week in week_index:
            week_hourly = hourly.loc[hourly["week_start"] == week]
            hour_index_column = "hour_index" if "hour_index" in week_hourly.columns else "hour_index_in_week"
            hour_count = int(week_hourly[hour_index_column].max()) + 1 if not week_hourly.empty else 0
            for hour_index in range(hour_count):
                hourly_rows.append(
                    {
                        "week_start": pd.Timestamp(week),
                        "hour_index": hour_index,
                        "max_hourly_hedge_share": 1.0,
                        "max_hourly_ramp_share": 1.0,
                    }
                )
        hourly_bounds_frame = pd.DataFrame(hourly_rows)
    weekly_bound_tensor = torch.tensor(
        weekly_bounds.set_index("week_start").reindex(week_index).fillna(0.0)[weekly_bound_columns].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )
    hourly_bound_tensor, _ = _frame_by_week(
        hourly_bounds_frame,
        week_index,
        columns=hourly_bound_columns,
        index_column="hour_index",
    )
    weekly_bound_reason_codes = (
        weekly_bounds.set_index("week_start").reindex(week_index)["bound_reason_code"].fillna("default").astype(str).tolist()
    )
    weekly_settlement_modes = (
        settlement_semantics.set_index("week_start").reindex(week_index)["settlement_mode"].fillna("previous_week_da_proxy").astype(str).tolist()
    )

    return TrainingTensorBundle(
        device=resolved_device,
        week_index=week_index,
        weekly_feature_columns=agent_columns,
        policy_columns=policy_columns,
        hourly_feature_columns=hourly_feature_columns,
        quarter_feature_columns=QUARTER_FEATURE_COLUMNS,
        weekly_bound_columns=weekly_bound_columns,
        hourly_bound_columns=hourly_bound_columns,
        weekly_bound_reason_codes=weekly_bound_reason_codes,
        weekly_settlement_modes=weekly_settlement_modes,
        weekly_feature_tensor=weekly_feature_tensor.to(resolved_device),
        policy_tensor=policy_tensor.to(resolved_device),
        hourly_tensor=hourly_tensor.to(resolved_device),
        quarter_price_tensor=quarter_price_tensor.to(resolved_device),
        weekly_bound_tensor=weekly_bound_tensor.to(resolved_device),
        hourly_bound_tensor=hourly_bound_tensor.to(resolved_device),
        forecast_weekly_load=forecast_weekly_load.to(resolved_device),
        actual_weekly_load=actual_weekly_load.to(resolved_device),
        lt_weekly_price=lt_weekly_price.to(resolved_device),
        hourly_valid_mask=hourly_valid_mask.to(resolved_device),
        quarter_valid_mask=quarter_valid_mask.to(resolved_device),
    )
