from __future__ import annotations

from typing import Any

import pandas as pd


def _format_timestamp_list(values: list[pd.Timestamp], max_items: int = 20) -> list[str]:
    return [value.strftime("%Y-%m-%d %H:%M:%S") for value in values[:max_items]]


def clean_total_data(
    frame: pd.DataFrame,
    sample_start: str,
    sample_end: str,
    expected_freq_minutes: int = 15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    numeric_columns = [column for column in data.columns if column != "datetime"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data = data.sort_values("datetime").reset_index(drop=True)

    duplicated_mask = data.duplicated(subset=["datetime"], keep=False)
    duplicate_timestamps = sorted(data.loc[duplicated_mask, "datetime"].unique().tolist())

    full_range_before = pd.date_range(
        data["datetime"].min(),
        data["datetime"].max(),
        freq=f"{expected_freq_minutes}min",
    )
    missing_before = full_range_before.difference(pd.DatetimeIndex(data["datetime"]))

    aggregated = (
        data.groupby("datetime", as_index=False)[numeric_columns]
        .mean(numeric_only=True)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    missing_after = pd.date_range(
        aggregated["datetime"].min(),
        aggregated["datetime"].max(),
        freq=f"{expected_freq_minutes}min",
    ).difference(pd.DatetimeIndex(aggregated["datetime"]))

    filtered = aggregated[
        (aggregated["datetime"] >= pd.Timestamp(sample_start))
        & (aggregated["datetime"] <= pd.Timestamp(sample_end))
    ].copy()

    numeric_missing = filtered[numeric_columns].isna().sum().sort_values(ascending=False)
    report = {
        "rows_before_cleaning": int(len(data)),
        "rows_after_duplicate_aggregation": int(len(aggregated)),
        "rows_after_sample_filter": int(len(filtered)),
        "start_before": str(data["datetime"].min()),
        "end_before": str(data["datetime"].max()),
        "start_after": str(filtered["datetime"].min()),
        "end_after": str(filtered["datetime"].max()),
        "duplicate_timestamp_count": int(len(duplicate_timestamps)),
        "duplicate_timestamps": _format_timestamp_list(duplicate_timestamps),
        "missing_timestamp_count_before_aggregation": int(len(missing_before)),
        "missing_timestamps_before_aggregation": _format_timestamp_list(list(missing_before)),
        "missing_timestamp_count_after_aggregation": int(len(missing_after)),
        "missing_timestamps_after_aggregation": _format_timestamp_list(list(missing_after)),
        "numeric_missing_counts": {key: int(value) for key, value in numeric_missing.items()},
    }
    return filtered.reset_index(drop=True), report


def build_data_quality_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 数据质量报告",
        "",
        "## 概览",
        "",
        f"- 原始行数: {report['rows_before_cleaning']}",
        f"- 去重聚合后行数: {report['rows_after_duplicate_aggregation']}",
        f"- 主样本期行数: {report['rows_after_sample_filter']}",
        f"- 原始时间范围: {report['start_before']} 至 {report['end_before']}",
        f"- 主样本时间范围: {report['start_after']} 至 {report['end_after']}",
        f"- 重复时间戳个数: {report['duplicate_timestamp_count']}",
        f"- 去重后缺失 15 分钟时点数: {report['missing_timestamp_count_after_aggregation']}",
        "",
        "## 重复时间戳",
        "",
    ]

    if report["duplicate_timestamps"]:
        lines.extend([f"- {value}" for value in report["duplicate_timestamps"]])
    else:
        lines.append("- 无")

    lines.extend(["", "## 缺失值统计", ""])
    for key, value in report["numeric_missing_counts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
