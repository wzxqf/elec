from __future__ import annotations

import pandas as pd

from src.policy.feasible_domain import compile_feasible_domain


def _config() -> dict:
    return {
        "policy_feasible_domain": {
            "enabled": True,
            "strict_mode": True,
            "bind_upper_actions": True,
            "bind_lower_actions": True,
            "bind_settlement_mode": True,
            "non_negative_position_required": True,
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
    }


def test_compile_feasible_domain_tightens_bounds_by_policy_state() -> None:
    weeks = pd.to_datetime(["2026-01-26", "2026-02-02"])
    weekly_metadata = pd.DataFrame(
        {
            "week_start": weeks,
            "forecast_weekly_net_demand_mwh": [1000.0, 1200.0],
            "hour_count": [168, 168],
        }
    )
    policy_state_trace = pd.DataFrame(
        {
            "week_start": weeks,
            "lt_price_linked_active": [0.0, 1.0],
            "ancillary_freq_reserve_tight": [0.0, 1.0],
            "renewable_mechanism_active": [0.0, 1.0],
        }
    )

    domain = compile_feasible_domain(
        config=_config(),
        weekly_metadata=weekly_metadata,
        policy_state_trace=policy_state_trace,
    )

    weekly_bounds = domain.weekly_bounds.set_index("week_start")
    settlement = domain.settlement_semantics.set_index("week_start")

    assert weekly_bounds.loc[weeks[0], "contract_adjustment_ratio_min"] == -0.30
    assert weekly_bounds.loc[weeks[0], "contract_adjustment_ratio_max"] == 0.30
    assert weekly_bounds.loc[weeks[1], "contract_adjustment_ratio_min"] == -0.12
    assert weekly_bounds.loc[weeks[1], "contract_adjustment_ratio_max"] == 0.12
    assert weekly_bounds.loc[weeks[1], "exposure_band_ratio_max"] == 0.12
    assert weekly_bounds.loc[weeks[1], "max_hourly_hedge_share"] == 0.40
    assert weekly_bounds.loc[weeks[1], "max_hourly_ramp_share"] == 0.20
    assert bool(weekly_bounds.loc[weeks[1], "feasible_domain_triggered"]) is True
    assert "lt_price_linked" in weekly_bounds.loc[weeks[1], "bound_reason_code"]
    assert settlement.loc[weeks[0], "settlement_mode"] == "previous_week_da_proxy"
    assert settlement.loc[weeks[1], "settlement_mode"] == "linked_40_60"

