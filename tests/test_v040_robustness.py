from __future__ import annotations

import pandas as pd

from src.analysis.robustness import run_robustness_analysis


def test_run_robustness_analysis_emits_scenario_rows() -> None:
    weekly_results = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2026-01-05", "2026-02-09", "2026-03-02"]),
            "profit_w": [100.0, 120.0, 80.0],
            "procurement_cost_w": [500.0, 520.0, 510.0],
            "cvar99_w": [50.0, 60.0, 55.0],
            "hedge_error_w": [0.05, 0.04, 0.06],
            "contract_position_mwh": [550.0, 560.0, 540.0],
            "forecast_weekly_net_demand_mwh": [1000.0, 1000.0, 1000.0],
        }
    )
    config = {
        "robustness": {
            "contract_ratio_shift": [-0.10, 0.00, 0.10],
            "policy_cutoffs": ["2026-02-01", "2026-03-01"],
            "forecast_error_scale": [0.80, 1.00, 1.20],
        }
    }

    result = run_robustness_analysis(weekly_results=weekly_results, config=config)

    assert not result.empty
    assert {"scenario_group", "scenario_name", "total_profit", "mean_cvar99", "robustness_rank"} <= set(result.columns)
    assert result["scenario_group"].nunique() == 3

