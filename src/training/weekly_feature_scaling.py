from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


WEEKLY_TOTAL_PATTERNS = (
    "weekly_net_demand_mwh",
    "weekly_load",
)

HOURLY_LOAD_PATTERNS = (
    "net_load",
    "load_dev",
    "wind_dev",
    "solar_dev",
    "renewable_dev",
    "tieline",
    "hydro",
    "nonmarket",
)

PRICE_PATTERNS = (
    "price",
    "spread",
    "lt_price",
)

UNIT_SCALE_PATTERNS = (
    "_active",
    "_flag",
    "_improved",
    "_pause",
    "_ratio",
    "_required",
    "_tight",
    "_triggered",
    "is_partial_week",
    "policy_count",
)


def _matches(column: str, patterns: tuple[str, ...]) -> bool:
    lowered = column.lower()
    return any(pattern in lowered for pattern in patterns)


def _keeps_native_scale(column: str) -> bool:
    return _matches(column, UNIT_SCALE_PATTERNS)


def scale_weekly_upper_features(
    weekly_features: torch.Tensor,
    *,
    columns: Sequence[str],
    forecast_weekly_load: torch.Tensor,
    valid_hours: torch.Tensor,
    config: dict[str, Any] | None,
) -> torch.Tensor:
    cfg = config or {}
    if not bool(cfg.get("enabled", False)):
        return weekly_features

    scaled = weekly_features.clone()
    weekly_scale = forecast_weekly_load.clamp_min(1.0).view(-1, 1)
    hourly_scale = (forecast_weekly_load / valid_hours.clamp_min(1.0)).clamp_min(1.0).view(-1, 1)
    price_scale = max(float(cfg.get("price_scale_yuan_per_mwh", 500.0)), 1.0)
    forward_days_scale = max(float(cfg.get("forward_days_scale", 60.0)), 1.0)

    for idx, column in enumerate(columns):
        if _matches(column, WEEKLY_TOTAL_PATTERNS):
            scaled[:, idx] = scaled[:, idx] / weekly_scale[:, 0]
        elif _matches(column, HOURLY_LOAD_PATTERNS):
            scaled[:, idx] = scaled[:, idx] / hourly_scale[:, 0]
        elif _keeps_native_scale(column):
            continue
        elif column.startswith("forward_") and column.endswith("_days"):
            scaled[:, idx] = scaled[:, idx] / forward_days_scale
        elif _matches(column, PRICE_PATTERNS):
            scaled[:, idx] = scaled[:, idx] / price_scale

    clip_abs = float(cfg.get("clip_abs", 5.0))
    if clip_abs > 0:
        scaled = torch.clamp(scaled, -clip_abs, clip_abs)
    return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
