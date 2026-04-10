from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_benchmark_actions(
    weeks: list[pd.Timestamp],
    weekly_features: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[pd.Timestamp, tuple[float, float]]]:
    feature_index = weekly_features.set_index("week_start").sort_index()
    benchmark_cfg = config["benchmarks"]
    actions = {
        "fixed_lock": {},
        "dynamic_lock_only": {},
        "rule_only": {},
    }

    for week in weeks:
        row = feature_index.loc[pd.Timestamp(week)]
        fixed_lock = float(np.clip(benchmark_cfg["fixed_lock_ratio"], 0.0, 1.0))
        dynamic_lock = float(
            np.clip(
                benchmark_cfg["dynamic_lock_base"]
                - benchmark_cfg["dynamic_lock_spread_penalty"] * max(row["prev_spread_mean"], 0.0) / 100.0
                + benchmark_cfg["dynamic_lock_renewable_bonus"] * row["prev_renewable_ratio_da_mean"]
                + benchmark_cfg["dynamic_lock_policy_bonus"] * max(
                    row.get("policy_event_20260101", 0.0),
                    row.get("policy_event_20260201", 0.0),
                ),
                0.0,
                1.0,
            )
        )
        actions["fixed_lock"][pd.Timestamp(week)] = (fixed_lock, 0.0)
        actions["dynamic_lock_only"][pd.Timestamp(week)] = (dynamic_lock, 0.0)
        actions["rule_only"][pd.Timestamp(week)] = (0.0, float(np.clip(benchmark_cfg["rule_only_intensity"], 0.0, 1.0)))

    return actions
