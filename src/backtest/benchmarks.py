from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def get_dynamic_lock_base(feature_row: pd.Series, config: dict[str, Any]) -> float:
    benchmark_cfg = config["benchmarks"]
    return float(
        np.clip(
            benchmark_cfg["dynamic_lock_base"]
            - benchmark_cfg["dynamic_lock_spread_penalty"] * max(float(feature_row["prev_spread_mean"]), 0.0) / 100.0
            + benchmark_cfg["dynamic_lock_renewable_bonus"] * float(feature_row["prev_renewable_ratio_da_mean"])
            + benchmark_cfg["dynamic_lock_policy_bonus"] * float(feature_row.get("renewable_mechanism_active", 0.0))
            - benchmark_cfg["dynamic_lock_linked_price_penalty"] * float(feature_row.get("lt_price_linked_active", 0.0)),
            0.0,
            1.0,
        )
    )


def get_dynamic_lock_base_for_week(
    weekly_features: pd.DataFrame,
    week_start: pd.Timestamp,
    config: dict[str, Any],
    weekly_feature_by_week: dict[pd.Timestamp, pd.Series] | None = None,
) -> float:
    week_start = pd.Timestamp(week_start)
    if isinstance(weekly_feature_by_week, dict) and week_start in weekly_feature_by_week:
        row = weekly_feature_by_week[week_start]
    else:
        row = weekly_features.set_index("week_start").sort_index().loc[week_start]
    return get_dynamic_lock_base(row, config)


def build_benchmark_actions(
    weeks: list[pd.Timestamp],
    weekly_features: pd.DataFrame,
    config: dict[str, Any],
    weekly_feature_by_week: dict[pd.Timestamp, pd.Series] | None = None,
) -> dict[str, dict[pd.Timestamp, dict[str, float | str]]]:
    feature_index = weekly_features.set_index("week_start").sort_index() if weekly_feature_by_week is None else None
    benchmark_cfg = config["benchmarks"]
    actions = {
        "fixed_lock": {},
        "dynamic_lock_only": {},
        "rule_only": {},
    }

    for week in weeks:
        week_start = pd.Timestamp(week)
        row = weekly_feature_by_week[week_start] if weekly_feature_by_week is not None else feature_index.loc[week_start]
        fixed_lock = float(np.clip(benchmark_cfg["fixed_lock_ratio"], 0.0, 1.0))
        dynamic_lock = get_dynamic_lock_base(row, config)
        actions["fixed_lock"][week_start] = {
            "mode": "absolute",
            "target_lock_ratio": fixed_lock,
            "exposure_bandwidth": 0.0,
        }
        actions["dynamic_lock_only"][week_start] = {
            "mode": "residual",
            "delta_lock_ratio_raw": 0.0,
            "exposure_bandwidth": 0.0,
            "lock_ratio_base": dynamic_lock,
        }
        actions["rule_only"][week_start] = {
            "mode": "absolute",
            "target_lock_ratio": 0.0,
            "exposure_bandwidth": float(np.clip(benchmark_cfg["rule_only_bandwidth"], 0.0, 1.0)),
        }

    return actions
