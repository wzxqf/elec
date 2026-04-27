from __future__ import annotations

import pandas as pd
import torch

from src.analysis.benchmarks import evaluate_benchmark_strategies
from src.training.tensor_bundle import TrainingTensorBundle


def test_evaluate_benchmark_strategies_sanitizes_nan_inputs() -> None:
    tensor_bundle = TrainingTensorBundle(
        device="cpu",
        week_index=pd.Index(pd.to_datetime(["2026-02-02"])),
        weekly_feature_columns=[],
        policy_columns=[],
        hourly_feature_columns=["net_load_da", "net_load_id", "price_spread", "load_dev", "renewable_dev"],
        quarter_feature_columns=["全网统一出清价格_日前", "全网统一出清价格_日内"],
        weekly_bound_columns=[],
        hourly_bound_columns=[],
        weekly_bound_reason_codes=["default"],
        weekly_settlement_modes=["previous_week_da_proxy"],
        weekly_feature_tensor=torch.zeros((1, 0), dtype=torch.float32),
        policy_tensor=torch.zeros((1, 0), dtype=torch.float32),
        hourly_tensor=torch.tensor(
            [[[100.0, 100.0, 10.0, 0.0, 0.0], [100.0, 100.0, float("nan"), 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        quarter_price_tensor=torch.tensor(
            [[[350.0, 360.0], [350.0, float("nan")]]],
            dtype=torch.float32,
        ),
        weekly_bound_tensor=torch.zeros((1, 0), dtype=torch.float32),
        hourly_bound_tensor=torch.zeros((1, 2, 0), dtype=torch.float32),
        forecast_weekly_load=torch.tensor([1000.0], dtype=torch.float32),
        actual_weekly_load=torch.tensor([980.0], dtype=torch.float32),
        lt_weekly_price=torch.tensor([320.0], dtype=torch.float32),
        hourly_valid_mask=torch.tensor([[True, True]]),
        quarter_valid_mask=torch.tensor([[True, True]]),
    )

    result = evaluate_benchmark_strategies({"tensor_bundle": tensor_bundle}, {"economics": {"retail_tariff_yuan_per_mwh": 430.0}})

    assert result["total_profit"].notna().all()
    assert result["total_procurement_cost"].notna().all()
    assert torch.isfinite(torch.tensor(result["total_profit"].tolist(), dtype=torch.float32)).all()
    assert torch.isfinite(torch.tensor(result["total_procurement_cost"].tolist(), dtype=torch.float32)).all()


def test_simple_rolling_hedge_uses_signed_net_energy_for_settlement() -> None:
    tensor_bundle = TrainingTensorBundle(
        device="cpu",
        week_index=pd.Index(pd.to_datetime(["2026-02-02"])),
        weekly_feature_columns=[],
        policy_columns=[],
        hourly_feature_columns=["net_load_da", "net_load_id", "price_spread", "load_dev", "renewable_dev"],
        quarter_feature_columns=["全网统一出清价格_日前", "全网统一出清价格_日内"],
        weekly_bound_columns=[],
        hourly_bound_columns=[],
        weekly_bound_reason_codes=["default"],
        weekly_settlement_modes=["previous_week_da_proxy"],
        weekly_feature_tensor=torch.zeros((1, 0), dtype=torch.float32),
        policy_tensor=torch.zeros((1, 0), dtype=torch.float32),
        hourly_tensor=torch.tensor(
            [[[100.0, 100.0, -20.0, 0.0, 0.0], [100.0, 100.0, -20.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        quarter_price_tensor=torch.tensor(
            [[[350.0, 360.0], [350.0, 360.0]]],
            dtype=torch.float32,
        ),
        weekly_bound_tensor=torch.zeros((1, 0), dtype=torch.float32),
        hourly_bound_tensor=torch.zeros((1, 2, 0), dtype=torch.float32),
        forecast_weekly_load=torch.tensor([1000.0], dtype=torch.float32),
        actual_weekly_load=torch.tensor([500.0], dtype=torch.float32),
        lt_weekly_price=torch.tensor([320.0], dtype=torch.float32),
        hourly_valid_mask=torch.tensor([[True, True]]),
        quarter_valid_mask=torch.tensor([[True, True]]),
    )

    result = evaluate_benchmark_strategies(
        {"tensor_bundle": tensor_bundle},
        {"economics": {"retail_tariff_yuan_per_mwh": 430.0, "friction_cost_yuan_per_mwh": 0.0}},
    )

    simple_profit = float(result.loc[result["strategy_name"] == "simple_rolling_hedge", "total_profit"].iloc[0])
    dynamic_profit = float(result.loc[result["strategy_name"] == "dynamic_lock_only", "total_profit"].iloc[0])
    assert simple_profit > dynamic_profit
