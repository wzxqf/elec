from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from src.backtest.simulator import simulate_strategy


def run_sensitivity_analysis(
    context: dict[str, Any],
    ppo_actions: dict[pd.Timestamp, tuple[float, float]],
) -> pd.DataFrame:
    config = context["config"]
    weeks = context["split"].test
    rows = []

    for value in config["sensitivity"]["beta_tail_risk"]:
        config_variant = deepcopy(config)
        config_variant["reward"]["beta_tail_risk"] = float(value)
        result = simulate_strategy(context["bundle"], weeks, ppo_actions, config_variant, "ppo_sensitivity_beta_tail_risk")
        rows.append({"factor": "尾部风险惩罚系数", "value": float(value), **result["metrics"]})

    for value in config["sensitivity"]["market_vol_scale"]:
        result = simulate_strategy(
            context["bundle"],
            weeks,
            ppo_actions,
            config,
            "ppo_sensitivity_market_vol",
            market_vol_scale=float(value),
        )
        rows.append({"factor": "市场波动率强度", "value": float(value), **result["metrics"]})

    for value in config["sensitivity"]["price_cap_multiplier"]:
        result = simulate_strategy(
            context["bundle"],
            weeks,
            ppo_actions,
            config,
            "ppo_sensitivity_price_cap",
            price_cap_multiplier=float(value),
        )
        rows.append({"factor": "价格限值倍数", "value": float(value), **result["metrics"]})

    return pd.DataFrame(rows)
