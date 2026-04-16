from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch


@dataclass(frozen=True)
class TrainingTensorBundle:
    device: str
    week_index: pd.Index
    weekly_feature_tensor: torch.Tensor
    policy_tensor: torch.Tensor
    hourly_tensor: torch.Tensor
    quarter_price_tensor: torch.Tensor
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
    hourly = bundle["hourly"].copy()
    quarter = bundle["quarter"].copy()

    for column in ["net_load_da", "net_load_id", "price_spread", "load_dev", "renewable_dev"]:
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
        columns=["net_load_da", "net_load_id", "price_spread", "load_dev", "renewable_dev"],
        index_column="hour_index",
    )
    quarter_price_tensor, quarter_valid_mask = _frame_by_week(
        quarter,
        week_index,
        columns=["全网统一出清价格_日前", "全网统一出清价格_日内"],
        index_column="interval_index",
    )

    return TrainingTensorBundle(
        device=resolved_device,
        week_index=week_index,
        weekly_feature_tensor=weekly_feature_tensor.to(resolved_device),
        policy_tensor=policy_tensor.to(resolved_device),
        hourly_tensor=hourly_tensor.to(resolved_device),
        quarter_price_tensor=quarter_price_tensor.to(resolved_device),
        forecast_weekly_load=forecast_weekly_load.to(resolved_device),
        actual_weekly_load=actual_weekly_load.to(resolved_device),
        lt_weekly_price=lt_weekly_price.to(resolved_device),
        hourly_valid_mask=hourly_valid_mask.to(resolved_device),
        quarter_valid_mask=quarter_valid_mask.to(resolved_device),
    )
