from __future__ import annotations

import numpy as np
import pandas as pd


def build_benchmark_actions(
    months: list[pd.Timestamp],
    monthly_features: pd.DataFrame,
    config: dict,
) -> dict[str, dict[pd.Timestamp, tuple[float, float]]]:
    feature_index = monthly_features.set_index("month")
    benchmark_cfg = config["benchmarks"]
    actions: dict[str, dict[pd.Timestamp, tuple[float, float]]] = {
        "fixed_lock": {},
        "dynamic_lock_only": {},
        "rule_only": {},
    }

    for month in months:
        row = feature_index.loc[month]
        fixed = float(np.clip(benchmark_cfg["fixed_lock_ratio"], 0.0, 1.0))
        dynamic = benchmark_cfg["dynamic_lock_base"]
        dynamic -= benchmark_cfg["dynamic_lock_spread_penalty"] * min(row["prev_spread_std"] / 100.0, 1.0)
        dynamic += benchmark_cfg["dynamic_lock_renewable_bonus"] * min(row["prev_renewable_ratio_da_mean"], 1.0)
        dynamic += benchmark_cfg["dynamic_lock_policy_bonus"] * row.get("policy_event_20260201", 0.0)
        dynamic = float(np.clip(dynamic, 0.05, 0.95))

        actions["fixed_lock"][month] = (fixed, 0.0)
        actions["dynamic_lock_only"][month] = (dynamic, 0.0)
        actions["rule_only"][month] = (0.0, float(np.clip(benchmark_cfg["rule_only_intensity"], 0.0, 1.0)))

    return actions
