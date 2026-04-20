from __future__ import annotations

import pandas as pd

from src.analysis.module1 import build_contract_value_weekly, build_risk_factor_manifest


def test_build_contract_value_weekly_emits_required_columns() -> None:
    weekly_results = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2026-01-05"]),
            "strategy": ["validation"],
            "contract_position_mwh": [600.0],
            "forecast_weekly_net_demand_mwh": [1000.0],
            "hedge_error_w": [0.05],
            "cvar99_w": [100.0],
            "lock_ratio_proxy_w": [0.6],
            "curve_match_score_w": [0.8],
            "stability_score_w": [0.7],
            "contract_curve_h1": [0.04],
            "contract_curve_h2": [0.04],
            "da_price_mean": [320.0],
            "id_price_mean": [330.0],
            "spread_std": [5.0],
            "da_id_cross_corr_w": [0.8],
            "extreme_event_flag_w": [0.0],
            "extreme_price_spike_flag_w": [0.0],
        }
    )

    contract_value = build_contract_value_weekly(weekly_results)

    assert {"expected_spot_price_w", "liquidity_premium_w", "contract_value_w", "lock_ratio_proxy_w"} <= set(
        contract_value.columns
    )


def test_build_risk_factor_manifest_emits_five_categories() -> None:
    weekly_results = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2026-01-05"]),
            "da_price_std": [10.0],
            "id_price_std": [12.0],
            "spread_std": [5.0],
            "da_id_cross_corr_w": [0.8],
            "forecast_weekly_net_demand_mwh": [1000.0],
            "actual_weekly_net_demand_mwh": [980.0],
            "renewable_dev_std": [15.0],
            "prev_renewable_ratio_da_mean": [0.2],
            "extreme_event_flag_w": [1.0],
            "extreme_price_spike_flag_w": [1.0],
            "renewable_mechanism_active": [1.0],
        }
    )

    manifest = build_risk_factor_manifest(weekly_results)

    assert manifest["factor_category"].nunique() == 5
