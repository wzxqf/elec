from __future__ import annotations

import pandas as pd

from src.backtest.materialize import materialize_particle_pair
from src.training.tensor_bundle import compile_training_tensor_bundle


def _bundle() -> dict:
    weeks = pd.to_datetime(["2026-01-05", "2026-01-12"])
    return {
        "weekly_features": pd.DataFrame(
            {
                "week_start": weeks,
                "feature_a": [1.0, 2.0],
                "feature_b": [0.5, 1.5],
            }
        ),
        "weekly_metadata": pd.DataFrame(
            {
                "week_start": weeks,
                "forecast_weekly_net_demand_mwh": [1000.0, 1200.0],
                "actual_weekly_net_demand_mwh": [1020.0, 1180.0],
                "lt_price_w": [290.0, 300.0],
                "da_price_mean": [300.0, 320.0],
                "id_price_mean": [310.0, 330.0],
                "da_price_std": [8.0, 9.0],
                "id_price_std": [10.0, 11.0],
                "spread_std": [4.0, 5.0],
                "da_id_cross_corr_w": [0.8, 0.7],
                "renewable_dev_std": [12.0, 14.0],
                "prev_renewable_ratio_da_mean": [0.2, 0.22],
                "extreme_event_flag_w": [0.0, 1.0],
                "extreme_price_spike_flag_w": [0.0, 1.0],
            }
        ),
        "policy_state_trace": pd.DataFrame(
            {
                "week_start": weeks,
                "renewable_mechanism_active": [0.0, 1.0],
                "lt_price_linked_active": [0.0, 1.0],
            }
        ),
        "hourly": pd.DataFrame(
            {
                "week_start": [weeks[0], weeks[0], weeks[1], weeks[1]],
                "hour_index": [0, 1, 0, 1],
                "net_load_da": [100.0, 120.0, 110.0, 130.0],
                "net_load_id": [98.0, 122.0, 112.0, 129.0],
                "price_spread": [10.0, 12.0, 9.0, 8.0],
                "load_dev": [2.0, -2.0, 1.0, -1.0],
                "renewable_dev": [5.0, 4.0, 6.0, 3.0],
            }
        ),
        "quarter": pd.DataFrame(
            {
                "week_start": [weeks[0], weeks[0], weeks[1], weeks[1]],
                "interval_index": [0, 1, 0, 1],
                "net_load_id_mwh": [25.0, 26.0, 31.0, 30.0],
                "全网统一出清价格_日前": [300.0, 302.0, 320.0, 321.0],
                "全网统一出清价格_日内": [310.0, 311.0, 330.0, 331.0],
            }
        ),
        "agent_feature_columns": ["feature_a", "feature_b"],
    }


def test_materialize_particle_pair_emits_v035_analysis_columns() -> None:
    tensor_bundle = compile_training_tensor_bundle(_bundle(), device="cpu")

    result = materialize_particle_pair(
        tensor_bundle=tensor_bundle,
        upper_particle=[0.0] * 12,
        lower_particle=[0.0] * 8,
        strategy_name="hybrid_pso_validation",
    )

    assert "lock_ratio_proxy_w" in result.weekly_results.columns
    assert "curve_match_score_w" in result.weekly_results.columns
    assert "stability_score_w" in result.weekly_results.columns


def test_materialize_particle_pair_handles_full_week_hourly_profile() -> None:
    week = pd.Timestamp("2026-01-05")
    hourly = pd.DataFrame(
        {
            "week_start": [week] * (24 * 7),
            "hour_index": list(range(24 * 7)),
            "net_load_da": [100.0 + float(index % 24) for index in range(24 * 7)],
            "net_load_id": [101.0 + float(index % 24) for index in range(24 * 7)],
            "price_spread": [8.0] * (24 * 7),
            "load_dev": [1.0] * (24 * 7),
            "renewable_dev": [2.0] * (24 * 7),
        }
    )
    quarter = pd.DataFrame(
        {
            "week_start": [week] * (24 * 7 * 4),
            "interval_index": list(range(24 * 7 * 4)),
            "全网统一出清价格_日前": [320.0] * (24 * 7 * 4),
            "全网统一出清价格_日内": [330.0] * (24 * 7 * 4),
        }
    )
    bundle = {
        "weekly_features": pd.DataFrame({"week_start": [week], "feature_a": [1.0], "feature_b": [0.5]}),
        "weekly_metadata": pd.DataFrame(
            {
                "week_start": [week],
                "forecast_weekly_net_demand_mwh": [1000.0],
                "actual_weekly_net_demand_mwh": [980.0],
                "lt_price_w": [300.0],
                "da_price_mean": [320.0],
                "id_price_mean": [330.0],
                "da_price_std": [8.0],
                "id_price_std": [10.0],
                "spread_std": [4.0],
                "da_id_cross_corr_w": [0.8],
                "renewable_dev_std": [12.0],
                "prev_renewable_ratio_da_mean": [0.2],
                "extreme_event_flag_w": [0.0],
                "extreme_price_spike_flag_w": [0.0],
            }
        ),
        "policy_state_trace": pd.DataFrame({"week_start": [week], "renewable_mechanism_active": [0.0]}),
        "hourly": hourly,
        "quarter": quarter,
        "agent_feature_columns": ["feature_a", "feature_b"],
    }

    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")

    result = materialize_particle_pair(
        tensor_bundle=tensor_bundle,
        upper_particle=[0.0] * 12,
        lower_particle=[0.0] * 8,
        strategy_name="hybrid_pso_validation",
    )

    assert result.weekly_results["curve_match_score_w"].notna().all()
