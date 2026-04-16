from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RollingRetrainWindow:
    window_name: str
    train_weeks: list[pd.Timestamp]
    val_weeks: list[pd.Timestamp]
    test_weeks: list[pd.Timestamp]


@dataclass(frozen=True)
class RollingSummary:
    schedule: pd.DataFrame
    metrics: pd.DataFrame
    aggregate: dict[str, float]


def build_rolling_retrain_plan(config: dict[str, Any], available_weeks: list[pd.Timestamp]) -> list[RollingRetrainWindow]:
    rolling_cfg = config.get("rolling_retrain", {})
    if not rolling_cfg.get("enabled", False):
        return []

    mode = str(rolling_cfg.get("mode", "expanding")).lower()
    train_min_weeks = int(rolling_cfg["train_min_weeks"])
    val_weeks = int(rolling_cfg["val_weeks"])
    test_weeks = int(rolling_cfg["test_weeks"])
    step_weeks = int(rolling_cfg.get("step_weeks", 1))
    available = [pd.Timestamp(week) for week in available_weeks]
    windows: list[RollingRetrainWindow] = []
    window_index = 1

    train_end_idx = train_min_weeks - 1
    while train_end_idx + val_weeks + test_weeks < len(available):
        if mode == "rolling":
            train_start_idx = train_end_idx - train_min_weeks + 1
        else:
            train_start_idx = 0
        val_start_idx = train_end_idx + 1
        test_start_idx = val_start_idx + val_weeks
        windows.append(
            RollingRetrainWindow(
                window_name=f"window_{window_index:02d}",
                train_weeks=available[train_start_idx : train_end_idx + 1],
                val_weeks=available[val_start_idx:test_start_idx],
                test_weeks=available[test_start_idx : test_start_idx + test_weeks],
            )
        )
        window_index += 1
        train_end_idx += step_weeks
    return windows


def summarize_rolling_results(results: list[Any]) -> RollingSummary:
    schedule_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for item in results:
        if isinstance(item, dict):
            getter = lambda key, default=None: item.get(key, default)
        else:
            getter = lambda key, default=None: getattr(item, key, default)
        schedule_rows.append(
            {
                "window_name": getter("window_name"),
                "train_start": min(getter("train_weeks")),
                "train_end": max(getter("train_weeks")),
                "val_start": min(getter("val_weeks")),
                "val_end": max(getter("val_weeks")),
                "test_start": min(getter("test_weeks")),
                "test_end": max(getter("test_weeks")),
            }
        )
        metric_rows.append(
                {
                    "window_name": getter("window_name"),
                    "best_score": float(getter("best_score")),
                    "total_procurement_cost": float(getter("total_procurement_cost")),
                    "total_profit": float(getter("total_profit", 0.0)),
                    "cvar99": float(getter("cvar99")),
                }
            )
    schedule = pd.DataFrame(schedule_rows, columns=["window_name", "train_start", "train_end", "val_start", "val_end", "test_start", "test_end"])
    metrics = pd.DataFrame(metric_rows, columns=["window_name", "best_score", "total_procurement_cost", "total_profit", "cvar99"])
    aggregate = {
        "window_count": float(len(results)),
        "mean_total_procurement_cost": float(metrics["total_procurement_cost"].mean()) if not metrics.empty else 0.0,
        "mean_total_profit": float(metrics["total_profit"].mean()) if not metrics.empty else 0.0,
        "mean_cvar99": float(metrics["cvar99"].mean()) if not metrics.empty else 0.0,
    }
    return RollingSummary(schedule=schedule, metrics=metrics, aggregate=aggregate)
