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


def test_materialize_particle_pair_emits_parametric_weekly_outputs() -> None:
    tensor_bundle = compile_training_tensor_bundle(_bundle(), device="cpu")

    result = materialize_particle_pair(
        tensor_bundle=tensor_bundle,
        upper_particle=[0.0] * 12,
        lower_particle=[0.0] * 8,
        strategy_name="hybrid_pso_validation",
    )

    assert "contract_adjustment_mwh_exec" in result.weekly_results.columns
    assert "contract_position_mwh" in result.weekly_results.columns
    assert "profit_w" in result.weekly_results.columns
    assert "reward_w" in result.weekly_results.columns
    assert "cvar99_w" in result.weekly_results.columns
    assert "spot_hedge_mwh" in result.hourly_results.columns
    assert "procurement_cost_15m" in result.settlement_results.columns
