from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeasibleDomainBundle:
    weekly_bounds: pd.DataFrame
    hourly_bounds: pd.DataFrame
    settlement_semantics: pd.DataFrame


def _cfg(policy_cfg: dict[str, Any], key: str, default: float | bool) -> float | bool:
    value = policy_cfg.get(key, default)
    if isinstance(default, bool):
        return bool(value)
    return float(value)


def _resolve_policy_config(config: dict[str, Any]) -> dict[str, Any]:
    if "policy_feasible_domain" in config:
        return dict(config["policy_feasible_domain"])
    projection_cfg = dict(config.get("policy_projection", {}))
    return {
        "enabled": True,
        "strict_mode": True,
        "bind_upper_actions": True,
        "bind_lower_actions": True,
        "bind_settlement_mode": True,
        "non_negative_position_required": bool(projection_cfg.get("enforce_non_negative_position", True)),
        "contract_adjustment_ratio_limit": 0.30,
        "contract_adjustment_ratio_limit_linked": 0.12,
        "exposure_band_ratio_floor": 0.05,
        "exposure_band_ratio_cap": 0.25,
        "exposure_band_ratio_cap_ancillary_tight": 0.12,
        "hourly_hedge_share_cap": 1.00,
        "hourly_hedge_share_cap_ancillary_tight": 0.40,
        "hourly_ramp_share_cap": 1.00,
        "hourly_ramp_share_cap_renewable_active": 0.20,
    }


def compile_feasible_domain(
    *,
    config: dict[str, Any],
    weekly_metadata: pd.DataFrame,
    policy_state_trace: pd.DataFrame,
) -> FeasibleDomainBundle:
    policy_cfg = _resolve_policy_config(config)
    weekly = weekly_metadata.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    policy = policy_state_trace.copy()
    policy["week_start"] = pd.to_datetime(policy["week_start"])
    merged = weekly.merge(policy, on="week_start", how="left").fillna(0.0)

    weekly_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    settlement_rows: list[dict[str, Any]] = []
    bind_settlement = bool(_cfg(policy_cfg, "bind_settlement_mode", True))

    for row in merged.itertuples(index=False):
        week_start = pd.Timestamp(row.week_start)
        lt_price_linked = float(getattr(row, "lt_price_linked_active", 0.0)) > 0.5
        ancillary_tight = float(getattr(row, "ancillary_freq_reserve_tight", 0.0)) > 0.5
        renewable_active = float(getattr(row, "renewable_mechanism_active", 0.0)) > 0.5
        hour_count = max(int(getattr(row, "hour_count", 168) or 168), 1)

        contract_limit = float(
            _cfg(
                policy_cfg,
                "contract_adjustment_ratio_limit_linked" if lt_price_linked else "contract_adjustment_ratio_limit",
                0.30,
            )
        )
        exposure_cap = float(
            _cfg(
                policy_cfg,
                "exposure_band_ratio_cap_ancillary_tight" if ancillary_tight else "exposure_band_ratio_cap",
                0.25,
            )
        )
        hourly_share_cap = float(
            _cfg(
                policy_cfg,
                "hourly_hedge_share_cap_ancillary_tight" if ancillary_tight else "hourly_hedge_share_cap",
                1.00,
            )
        )
        hourly_ramp_cap = float(
            _cfg(
                policy_cfg,
                "hourly_ramp_share_cap_renewable_active" if renewable_active else "hourly_ramp_share_cap",
                1.00,
            )
        )

        reason_tokens: list[str] = []
        if lt_price_linked:
            reason_tokens.append("lt_price_linked")
        if ancillary_tight:
            reason_tokens.append("ancillary_tight")
        if renewable_active:
            reason_tokens.append("renewable_active")
        bound_reason_code = "|".join(reason_tokens) if reason_tokens else "default"
        triggered = bool(reason_tokens)

        weekly_rows.append(
            {
                "week_start": week_start,
                "contract_adjustment_ratio_min": -contract_limit,
                "contract_adjustment_ratio_max": contract_limit,
                "exposure_band_ratio_min": float(_cfg(policy_cfg, "exposure_band_ratio_floor", 0.05)),
                "exposure_band_ratio_max": exposure_cap,
                "max_hourly_hedge_share": hourly_share_cap,
                "max_hourly_ramp_share": hourly_ramp_cap,
                "non_negative_position_required": bool(_cfg(policy_cfg, "non_negative_position_required", True)),
                "feasible_domain_triggered": triggered,
                "bound_reason_code": bound_reason_code,
            }
        )
        settlement_rows.append(
            {
                "week_start": week_start,
                "settlement_mode": "linked_40_60" if (bind_settlement and lt_price_linked) else "previous_week_da_proxy",
                "bound_reason_code": bound_reason_code,
            }
        )
        for hour_index in range(hour_count):
            hourly_rows.append(
                {
                    "week_start": week_start,
                    "hour_index": hour_index,
                    "max_hourly_hedge_share": hourly_share_cap,
                    "max_hourly_ramp_share": hourly_ramp_cap,
                    "bound_reason_code": bound_reason_code,
                }
            )

    return FeasibleDomainBundle(
        weekly_bounds=pd.DataFrame(weekly_rows),
        hourly_bounds=pd.DataFrame(hourly_rows),
        settlement_semantics=pd.DataFrame(settlement_rows),
    )
