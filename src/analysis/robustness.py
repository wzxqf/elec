from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.agents.train_ppo import evaluate_policy
from src.backtest.simulator import simulate_strategy


def run_robustness_analysis(
    context: dict[str, Any],
    model: Any,
    ppo_actions: dict[pd.Timestamp, tuple[float, float]],
) -> pd.DataFrame:
    config = context["config"]
    rows = []

    for shift in config["robustness"]["contract_ratio_shift"]:
        shifted_actions = {
            week: (float(np.clip(action[0] + shift, 0.0, 1.0)), action[1])
            for week, action in ppo_actions.items()
        }
        result = simulate_strategy(context["bundle"], context["split"].test, shifted_actions, config, "ppo_contract_shift")
        rows.append({"experiment": "合约比例扰动", "value": float(shift), **result["metrics"]})

    all_weeks = sorted(set(context["split"].train + context["split"].val + context["split"].test))
    full_actions = evaluate_policy(model, context["bundle"], all_weeks, config, "ppo_all_weeks")["actions"]
    for cutoff in config["robustness"]["policy_cutoffs"]:
        cutoff_ts = pd.Timestamp(cutoff)
        before = [week for week in all_weeks if week < cutoff_ts]
        after = [week for week in all_weeks if week >= cutoff_ts]
        if before:
            result = simulate_strategy(context["bundle"], before, full_actions, config, "ppo_policy_before")
            rows.append({"experiment": "政策边界前样本", "value": cutoff, **result["metrics"]})
        if after:
            result = simulate_strategy(context["bundle"], after, full_actions, config, "ppo_policy_after")
            rows.append({"experiment": "政策边界后样本", "value": cutoff, **result["metrics"]})

    for scale in config["robustness"]["forecast_error_scale"]:
        result = simulate_strategy(
            context["bundle"],
            context["split"].test,
            ppo_actions,
            config,
            "ppo_forecast_error",
            forecast_error_scale=float(scale),
        )
        rows.append({"experiment": "预测误差水平", "value": float(scale), **result["metrics"]})

    return pd.DataFrame(rows)
