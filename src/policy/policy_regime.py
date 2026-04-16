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
    "ancillary_price_boundary_tight": 0.0,
    "renewable_mechanism_active": 0.0,
    "mechanism_price_floor": 0.0,
    "mechanism_price_ceiling": 0.0,
    "mechanism_volume_ratio_max": 0.0,
    "mechanism_exec_term_years": 0.0,
    "mechanism_stage_label": "未启用",
    "lt_price_linked_active": 0.0,
    "fixed_price_ratio_max": 0.0,
    "linked_price_ratio_min": 0.0,
    "lt_linkage_pre_window": 0.0,
    "info_disclosure_active": 0.0,
    "metering_boundary_improved": 0.0,
    "forecast_boundary_improved": 0.0,
    "policy_count": 0.0,
    "policy_source_file_count": 0.0,
    "policy_parse_failure_count": 0.0,
    "forward_price_linkage_days": 60.0,
    "forward_price_linkage_code": 0.0,
    "forward_price_linkage_in_window": 0.0,
    "forward_price_linkage_type": "none",
    "forward_mechanism_execution_days": 60.0,
    "forward_mechanism_execution_code": 0.0,
    "forward_mechanism_execution_in_window": 0.0,
    "forward_mechanism_execution_type": "none",
    "forward_ancillary_coupling_days": 60.0,
    "forward_ancillary_coupling_code": 0.0,
    "forward_ancillary_coupling_in_window": 0.0,
    "forward_ancillary_coupling_type": "none",
    "forward_info_forecast_boundary_days": 60.0,
    "forward_info_forecast_boundary_code": 0.0,
    "forward_info_forecast_boundary_in_window": 0.0,
    "forward_info_forecast_boundary_type": "none",
}

FORWARD_GROUPS = {
    "price_linkage": {
        "prefix": "forward_price_linkage",
        "rule_types": {"lt_price_linkage"},
    },
    "mechanism_execution": {
        "prefix": "forward_mechanism_execution",
        "rule_types": {"renewable_mechanism_execution"},
    },
    "ancillary_coupling": {
        "prefix": "forward_ancillary_coupling",
        "rule_types": {"ancillary_coupling"},
    },
    "info_forecast_boundary": {
        "prefix": "forward_info_forecast_boundary",
        "rule_types": {"info_forecast_boundary"},
    },
}


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return pd.Timestamp(value)


def _build_forward_state(
    week_start: pd.Timestamp,
    rule_table: pd.DataFrame,
    countdown_cap_days: int,
    forward_window_days: int,
    rule_types: set[str],
    prefix: str,
) -> dict[str, Any]:
    subset = rule_table.loc[
        rule_table["rule_type"].isin(rule_types) & rule_table["effective_start"].notna(),
        ["effective_start", "rule_type", "state_name"],
    ].copy()
    subset["effective_start"] = subset["effective_start"].apply(pd.Timestamp)
    subset = subset.loc[subset["effective_start"] > week_start].sort_values("effective_start").drop_duplicates()

    if subset.empty:
        return {
            f"{prefix}_days": float(countdown_cap_days),
            f"{prefix}_code": 0.0,
            f"{prefix}_in_window": 0.0,
            f"{prefix}_type": "none",
        }

    next_row = subset.iloc[0]
    next_days = float(min((pd.Timestamp(next_row["effective_start"]) - week_start).days, countdown_cap_days))
    type_name = str(next_row["rule_type"])
    type_code = float(sorted(rule_types).index(type_name) + 1 if type_name in rule_types else 0)
    return {
        f"{prefix}_days": next_days,
        f"{prefix}_code": type_code,
        f"{prefix}_in_window": float(next_days <= forward_window_days and type_code > 0),
        f"{prefix}_type": type_name,
    }


def build_policy_state_trace(
    weekly_metadata: pd.DataFrame,
    rule_table: pd.DataFrame,
    inventory: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    forward_window_days = int(config["policy_regime"]["pre_switch_window_days"])
    countdown_cap_days = int(config["policy_regime"]["countdown_cap_days"])
    parse_failures = inventory.loc[inventory["parse_status"] == "failed", "source_file"].astype(str).tolist()

    if rule_table.empty:
        empty_rows = []
        for _, meta in weekly_metadata.sort_values("week_start").iterrows():
            state = {"week_start": pd.Timestamp(meta["week_start"])}
            state.update(DEFAULT_POLICY_STATE)
            state["failed_policy_files"] = "|".join(parse_failures)
            empty_rows.append(state)
        return pd.DataFrame(empty_rows).sort_values("week_start").reset_index(drop=True)

    ordered_rules = rule_table.copy().sort_values(["effective_start", "rule_id"], na_position="last").reset_index(drop=True)
    end_series = ordered_rules["effective_end"].apply(_as_timestamp) if "effective_end" in ordered_rules.columns else None

    for _, meta in weekly_metadata.sort_values("week_start").iterrows():
        week_start = pd.Timestamp(meta["week_start"])
        week_end = pd.Timestamp(meta["week_end"])
        state = {"week_start": week_start}
        state.update(DEFAULT_POLICY_STATE)

        active_mask = ordered_rules["effective_start"].apply(_as_timestamp).fillna(pd.Timestamp.min) <= week_start
        if end_series is not None:
            active_mask &= end_series.isna() | (end_series >= week_start)
        active_rules = ordered_rules.loc[active_mask].copy()

        source_files: list[str] = []
        policy_names: list[str] = []
        state_groups: list[str] = []
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
            state_groups.append(str(rule.get("state_group", "")))

        state["policy_count"] = float(len(active_rules))
        state["policy_source_file_count"] = float(len(set(source_files)))
        state["policy_parse_failure_count"] = float(len(parse_failures))
        state["policy_sources"] = "|".join(sorted(set(source_files)))
        state["policy_names"] = "|".join(sorted(set(policy_names)))
        state["active_state_groups"] = "|".join(sorted(group for group in set(state_groups) if group))
        state["failed_policy_files"] = "|".join(parse_failures)

        for group_config in FORWARD_GROUPS.values():
            state.update(
                _build_forward_state(
                    week_start=week_start,
                    rule_table=ordered_rules,
                    countdown_cap_days=countdown_cap_days,
                    forward_window_days=forward_window_days,
                    rule_types=group_config["rule_types"],
                    prefix=group_config["prefix"],
                )
            )

        state["lt_linkage_pre_window"] = float(
            state["forward_price_linkage_in_window"] > 0.5 and state["lt_price_linked_active"] < 0.5
        )
        rows.append(state)

    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)
