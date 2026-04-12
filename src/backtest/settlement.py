from __future__ import annotations

from typing import Any

import pandas as pd


def resolve_settlement_context(
    quarter_frame: pd.DataFrame,
    metadata: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    da_proxy = float(quarter_frame["全网统一出清价格_日前"].mean())
    id_proxy = float(quarter_frame["全网统一出清价格_日内"].mean())
    if float(metadata.get("lt_price_linked_active", 0.0)) >= 0.5:
        fixed_ratio = float(metadata.get("fixed_price_ratio_max", 0.4) or 0.4)
        linked_ratio = float(metadata.get("linked_price_ratio_min", 0.6) or 0.6)
        total_ratio = fixed_ratio + linked_ratio
        if total_ratio <= 0.0:
            fixed_ratio, linked_ratio = 0.4, 0.6
            total_ratio = 1.0
        fixed_ratio /= total_ratio
        linked_ratio /= total_ratio
        lt_price = fixed_ratio * da_proxy + linked_ratio * id_proxy
        regime = "2026-02起40%日前固定价+60%日内联动价代理"
    else:
        fixed_ratio = 1.0
        linked_ratio = 0.0
        lt_price = float(metadata["lt_price_w"]) if pd.notna(metadata.get("lt_price_w")) else da_proxy
        regime = "2026-02前上一自然周日前均价代理"

    mechanism_active = float(metadata.get("renewable_mechanism_active", 0.0))
    return {
        "lt_price_w": float(lt_price),
        "lt_price_fixed_ratio": float(fixed_ratio),
        "lt_price_linked_ratio": float(linked_ratio),
        "lt_price_regime": regime,
        "settlement_note": config["reporting"]["settlement_note"],
        "renewable_mechanism_active": mechanism_active,
        "mechanism_price_floor": float(metadata.get("mechanism_price_floor", 0.0) or 0.0),
        "mechanism_price_ceiling": float(metadata.get("mechanism_price_ceiling", 0.0) or 0.0),
        "mechanism_volume_ratio_max": float(metadata.get("mechanism_volume_ratio_max", 0.0) or 0.0),
        "mechanism_stage_label": str(metadata.get("mechanism_stage_label", "未启用")),
        "ancillary_freq_reserve_tight": float(metadata.get("ancillary_freq_reserve_tight", 0.0)),
        "ancillary_peak_shaving_pause": float(metadata.get("ancillary_peak_shaving_pause", 0.0)),
    }


def settle_week(
    quarter_frame: pd.DataFrame,
    hourly_rule_trace: pd.DataFrame,
    metadata: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    settlement_context = resolve_settlement_context(quarter_frame, metadata, config)
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
            "signal_composite",
            "bandwidth_mwh",
            "lower_band",
            "upper_band",
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
    settlement["lt_cost_15m"] = settlement["lt_energy_15m"] * settlement_context["lt_price_w"]
    settlement["spot_cost_15m"] = settlement["spot_energy_15m"] * settlement["全网统一出清价格_日前"]
    settlement["imbalance_cost_15m"] = (
        settlement["imbalance_energy_15m"] * settlement["全网统一出清价格_日内"] * penalty_multiplier
    )
    settlement["procurement_cost_15m"] = (
        settlement["lt_cost_15m"] + settlement["spot_cost_15m"] + settlement["imbalance_cost_15m"]
    )
    settlement["lt_price_w"] = settlement_context["lt_price_w"]
    settlement["lt_price_fixed_ratio"] = settlement_context["lt_price_fixed_ratio"]
    settlement["lt_price_linked_ratio"] = settlement_context["lt_price_linked_ratio"]
    settlement["lt_price_regime"] = settlement_context["lt_price_regime"]
    settlement["renewable_mechanism_active"] = settlement_context["renewable_mechanism_active"]
    settlement["mechanism_price_floor"] = settlement_context["mechanism_price_floor"]
    settlement["mechanism_price_ceiling"] = settlement_context["mechanism_price_ceiling"]
    settlement["mechanism_volume_ratio_max"] = settlement_context["mechanism_volume_ratio_max"]
    settlement["mechanism_stage_label"] = settlement_context["mechanism_stage_label"]
    settlement["ancillary_freq_reserve_tight"] = settlement_context["ancillary_freq_reserve_tight"]
    settlement["ancillary_peak_shaving_pause"] = settlement_context["ancillary_peak_shaving_pause"]
    settlement["settlement_note"] = config["reporting"]["settlement_note"]
    return settlement
