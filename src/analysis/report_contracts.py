from __future__ import annotations

from typing import Iterable

import pandas as pd


def infer_date_range(frame: pd.DataFrame, candidates: Iterable[str] = ("week_start", "test_start", "scenario_name")) -> str:
    if frame.empty:
        return "n/a"
    for column in candidates:
        if column not in frame.columns:
            continue
        series = pd.to_datetime(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        return f"{series.min():%Y-%m-%d} -> {series.max():%Y-%m-%d}"
    return "n/a"


def build_summary_scope_lines(
    *,
    sample_scope: str,
    week_count: int | float | None,
    aggregation_method: str,
    date_range: str,
) -> list[str]:
    count_text = "n/a" if week_count is None else str(int(week_count))
    return [
        f"- sample_scope: {sample_scope}",
        f"- week_count: {count_text}",
        f"- aggregation_method: {aggregation_method}",
        f"- date_range: {date_range}",
        "",
    ]


def positive_negative_counts(values: pd.Series) -> tuple[int, int]:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return int(numeric.gt(0.0).sum()), int(numeric.lt(0.0).sum())
