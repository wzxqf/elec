from __future__ import annotations

import pandas as pd
import torch

from src.analysis.robustness import run_robustness_analysis
from src.training.tensor_bundle import TrainingTensorBundle


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


def test_run_robustness_analysis_can_rerun_settlement_scenarios() -> None:
    tensor_bundle = TrainingTensorBundle(
        device="cpu",
        week_index=pd.Index(pd.to_datetime(["2026-02-02"])),
        weekly_feature_columns=["feature_a"],
        policy_columns=[],
        hourly_feature_columns=["net_load_da", "net_load_id", "price_spread", "load_dev", "renewable_dev"],
        quarter_feature_columns=["全网统一出清价格_日前", "全网统一出清价格_日内", "net_load_da_mwh", "net_load_id_mwh"],
        weekly_bound_columns=[
            "contract_adjustment_ratio_min",
            "contract_adjustment_ratio_max",
            "exposure_band_ratio_min",
            "exposure_band_ratio_max",
            "max_hourly_hedge_share",
            "max_hourly_ramp_share",
            "non_negative_position_required",
            "feasible_domain_triggered",
        ],
        hourly_bound_columns=["max_hourly_hedge_share", "max_hourly_ramp_share"],
        weekly_bound_reason_codes=["default"],
        weekly_settlement_modes=["linked_40_60"],
        weekly_feature_tensor=torch.zeros((1, 1), dtype=torch.float32),
        policy_tensor=torch.zeros((1, 0), dtype=torch.float32),
        hourly_tensor=torch.zeros((1, 1, 5), dtype=torch.float32),
        quarter_price_tensor=torch.tensor([[[100.0, 120.0, 0.0, 60.0], [100.0, 140.0, 0.0, 40.0]]], dtype=torch.float32),
        weekly_bound_tensor=torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        hourly_bound_tensor=torch.tensor([[[1.0, 1.0]]], dtype=torch.float32),
        forecast_weekly_load=torch.tensor([100.0], dtype=torch.float32),
        actual_weekly_load=torch.tensor([100.0], dtype=torch.float32),
        lt_weekly_price=torch.tensor([90.0], dtype=torch.float32),
        hourly_valid_mask=torch.tensor([[True]]),
        quarter_valid_mask=torch.tensor([[True, True]]),
    )
    config = {
        "reward": {
            "baseline_position_ratios": [0.50],
            "cvar_alpha": 0.99,
            "lambda_tail": 0.0,
            "lambda_hedge": 0.0,
            "lambda_trade": 0.0,
            "lambda_violate": 0.0,
        },
        "score_kernel": {
            "contract_position_base_ratio": 0.50,
            "exposure_band_base_ratio": 0.00,
            "lt_settlement_weight": 1.00,
            "da_settlement_weight": 0.00,
            "hourly_limit": {"base_multiplier": 0.00, "shrink_multiplier": 0.00},
        },
        "economics": {
            "retail_tariff_yuan_per_mwh": 200.0,
            "imbalance_penalty_multiplier": 1.0,
            "adjustment_cost_yuan_per_mwh": 0.0,
            "friction_cost_yuan_per_mwh": 0.0,
        },
        "robustness": {"forecast_error_scale": [1.20]},
    }

    result = run_robustness_analysis(
        weekly_results=pd.DataFrame({"week_start": pd.to_datetime(["2026-02-02"])}),
        config=config,
        tensor_bundle=tensor_bundle,
        upper_particle=torch.zeros(4, dtype=torch.float32),
        lower_particle=torch.zeros(4, dtype=torch.float32),
    )

    row = result.iloc[0]
    assert row["scenario_group"] == "forecast_error_scale"
    assert row["scenario_settlement_method"] == "rerun_materialize_particle_pair"
    assert row["week_count"] == 1
    assert row["settlement_record_count"] == 2
    assert row["total_profit"] != 0.0
