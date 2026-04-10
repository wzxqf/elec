from __future__ import annotations

from typing import Any

import pandas as pd


def settle_week(
    quarter_frame: pd.DataFrame,
    hourly_rule_trace: pd.DataFrame,
    lt_price_w: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    hourly_to_join = hourly_rule_trace[
        [
            "hour",
            "q_lt_hourly",
            "q_spot",
            "delta_q",
            "a_t",
            "q_base",
            "spot_need",
            "signal_spread",
            "signal_load_dev",
            "signal_renewable_dev",
        ]
    ].copy()
    settlement = quarter_frame.copy()
    settlement["hour"] = settlement["datetime"].dt.floor("h")
    settlement = settlement.merge(hourly_to_join, on="hour", how="left")

    settlement["actual_need_15m"] = settlement["net_load_id_mwh"].clip(lower=0.0)
    settlement["lt_energy_15m"] = settlement["q_lt_hourly"] * 0.25
    settlement["spot_energy_15m"] = settlement["q_spot"] * 0.25
    settlement["scheduled_energy_15m"] = settlement["lt_energy_15m"] + settlement["spot_energy_15m"]
    settlement["imbalance_energy_15m"] = (settlement["actual_need_15m"] - settlement["scheduled_energy_15m"]).abs()

    penalty_multiplier = float(config["cost"]["imbalance_penalty_multiplier"])
    settlement["lt_cost_15m"] = settlement["lt_energy_15m"] * float(lt_price_w)
    settlement["spot_cost_15m"] = settlement["spot_energy_15m"] * settlement["全网统一出清价格_日前"]
    settlement["imbalance_cost_15m"] = (
        settlement["imbalance_energy_15m"] * settlement["全网统一出清价格_日内"] * penalty_multiplier
    )
    settlement["procurement_cost_15m"] = (
        settlement["lt_cost_15m"] + settlement["spot_cost_15m"] + settlement["imbalance_cost_15m"]
    )
    settlement["settlement_note"] = config["reporting"]["settlement_note"]
    return settlement
