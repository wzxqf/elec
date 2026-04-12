from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_POLICY_STATE = {
    "lt_settlement_base": 0.0,
    "spot_marginal_exposure": 0.0,
    "lt_spot_coupling_state": 0.0,
    "ancillary_peak_shaving_pause": 0.0,
    "ancillary_freq_reserve_tight": 0.0,
    "ancillary_price_cap_tight": 0.0,
    "renewable_mechanism_active": 0.0,
    "mechanism_price_floor": 0.0,
    "mechanism_price_ceiling": 0.0,
    "mechanism_volume_ratio_max": 0.0,
    "mechanism_exec_term_years": 0.0,
    "mechanism_stage_label": "未启用",
    "lt_price_linked_active": 0.0,
    "fixed_price_ratio_max": 0.0,
    "linked_price_ratio_min": 0.0,
    "policy_count": 0.0,
}


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Timestamp(value)


def build_policy_state_trace(
    weekly_metadata: pd.DataFrame,
    rule_table: pd.DataFrame,
    inventory: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    forward_window_days = int(config["policy_regime"]["pre_switch_window_days"])
    countdown_cap_days = int(config["policy_regime"]["countdown_cap_days"])

    switch_schedule = (
        rule_table.loc[rule_table["effective_start"].notna(), ["effective_start", "rule_type"]]
        .drop_duplicates()
        .sort_values("effective_start")
        .reset_index(drop=True)
    )
    switch_type_map = {value: index + 1 for index, value in enumerate(sorted(switch_schedule["rule_type"].unique()))}

    for _, meta in weekly_metadata.sort_values("week_start").iterrows():
        week_start = pd.Timestamp(meta["week_start"])
        week_end = pd.Timestamp(meta["week_end"])
        state = {"week_start": week_start}
        state.update(DEFAULT_POLICY_STATE)

        active_mask = rule_table["effective_start"].apply(_as_timestamp).fillna(pd.Timestamp.min) <= week_end
        if "effective_end" in rule_table.columns:
            end_series = rule_table["effective_end"].apply(_as_timestamp)
            active_mask &= end_series.isna() | (end_series >= week_start)
        active_rules = rule_table.loc[active_mask].copy()

        source_files = []
        policy_names = []
        for _, rule in active_rules.iterrows():
            state_name = str(rule["state_name"])
            state_value = rule["state_value"]
            if state_name in DEFAULT_POLICY_STATE and isinstance(DEFAULT_POLICY_STATE[state_name], str):
                state[state_name] = str(state_value)
            else:
                try:
                    state[state_name] = float(state_value)
                except Exception:
                    state[state_name] = state_value
            source_files.append(str(rule["source_file"]))
            policy_names.append(str(rule["policy_name"]))

        next_switch = switch_schedule.loc[switch_schedule["effective_start"] > week_start].head(1)
        if next_switch.empty:
            next_days = float(countdown_cap_days)
            next_type = "none"
            next_code = 0
        else:
            next_ts = pd.Timestamp(next_switch.iloc[0]["effective_start"])
            next_days = float(min((next_ts - week_start).days, countdown_cap_days))
            next_type = str(next_switch.iloc[0]["rule_type"])
            next_code = int(switch_type_map[next_type])

        parse_failures = inventory.loc[inventory["parse_status"] == "failed", "source_file"].tolist()
        state["policy_count"] = float(len(active_rules))
        state["policy_sources"] = "|".join(sorted(set(source_files)))
        state["policy_names"] = "|".join(sorted(set(policy_names)))
        state["pre_switch_window"] = float(next_days <= forward_window_days and next_code > 0)
        state["days_to_next_policy_switch"] = next_days
        state["next_policy_switch_type"] = next_type
        state["next_policy_switch_code"] = float(next_code)
        state["failed_policy_files"] = "|".join(parse_failures)
        rows.append(state)

    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)
