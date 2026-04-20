from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.rolling_pipeline import build_rolling_retrain_plan, summarize_rolling_results


@dataclass
class _WindowResult:
    window_name: str
    train_weeks: list[pd.Timestamp]
    val_weeks: list[pd.Timestamp]
    test_weeks: list[pd.Timestamp]
    best_score: float
    total_procurement_cost: float
    total_profit: float
    cvar99: float


def test_build_rolling_retrain_plan_generates_retrain_windows() -> None:
    available_weeks = pd.to_datetime(
        [
            "2025-12-29",
            "2026-01-05",
            "2026-01-12",
            "2026-01-19",
            "2026-01-26",
            "2026-02-02",
        ]
    ).tolist()
    config = {
        "rolling_retrain": {
            "enabled": True,
            "mode": "expanding",
            "train_min_weeks": 2,
            "val_weeks": 1,
            "test_weeks": 1,
            "step_weeks": 1,
        }
    }

    plan = build_rolling_retrain_plan(config=config, available_weeks=available_weeks)

    assert len(plan) == 3
    assert plan[0].window_name == "window_01"
    assert len(plan[0].train_weeks) == 2
    assert len(plan[0].val_weeks) == 1
    assert len(plan[0].test_weeks) == 1
    assert plan[1].train_weeks[0] == pd.Timestamp("2025-12-29")


def test_summarize_rolling_results_outputs_audit_frames() -> None:
    results = [
        _WindowResult(
            window_name="window_01",
            train_weeks=pd.to_datetime(["2025-12-29", "2026-01-05"]).tolist(),
            val_weeks=pd.to_datetime(["2026-01-12"]).tolist(),
            test_weeks=pd.to_datetime(["2026-01-19"]).tolist(),
            best_score=10.0,
            total_procurement_cost=1000.0,
            total_profit=120.0,
            cvar99=200.0,
        ),
        _WindowResult(
            window_name="window_02",
            train_weeks=pd.to_datetime(["2025-12-29", "2026-01-05", "2026-01-12"]).tolist(),
            val_weeks=pd.to_datetime(["2026-01-19"]).tolist(),
            test_weeks=pd.to_datetime(["2026-01-26"]).tolist(),
            best_score=9.0,
            total_procurement_cost=900.0,
            total_profit=150.0,
            cvar99=180.0,
        ),
    ]

    summary = summarize_rolling_results(results)

    assert list(summary.schedule.columns) == ["window_name", "train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]
    assert list(summary.metrics.columns) == ["window_name", "best_score", "total_procurement_cost", "total_profit", "cvar99"]
    assert summary.aggregate["window_count"] == 2
    assert summary.aggregate["mean_total_procurement_cost"] == 950.0
    assert summary.aggregate["mean_total_profit"] == 135.0
    assert summary.aggregate["mean_cvar99"] == 190.0
