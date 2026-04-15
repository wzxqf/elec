from __future__ import annotations

from typing import Any

import pandas as pd


def _build_frame_lookup(frame: pd.DataFrame | None) -> dict[pd.Timestamp, pd.DataFrame]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "week_start" not in frame.columns:
        return {}
    return {
        pd.Timestamp(week_start): week_frame.copy()
        for week_start, week_frame in frame.groupby("week_start", sort=False)
    }


def _build_series_lookup(frame: pd.DataFrame | None) -> dict[pd.Timestamp, pd.Series]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "week_start" not in frame.columns:
        return {}

    indexed = frame.copy()
    indexed["week_start"] = pd.to_datetime(indexed["week_start"])
    indexed = indexed.drop_duplicates(subset=["week_start"]).set_index("week_start").sort_index()
    return {
        pd.Timestamp(week_start): row.copy()
        for week_start, row in indexed.iterrows()
    }


def prepare_runtime_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    bundle["quarter_by_week"] = _build_frame_lookup(bundle.get("quarter"))
    bundle["hourly_by_week"] = _build_frame_lookup(bundle.get("hourly"))
    bundle["weekly_metadata_by_week"] = _build_series_lookup(bundle.get("weekly_metadata"))
    bundle["weekly_feature_by_week"] = _build_series_lookup(bundle.get("weekly_features"))
    bundle["reward_reference_by_week"] = _build_series_lookup(bundle.get("reward_reference"))
    return bundle
