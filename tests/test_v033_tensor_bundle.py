from __future__ import annotations

import pandas as pd
import torch

from src.training.tensor_bundle import compile_training_tensor_bundle


def _build_bundle() -> dict:
    weeks = pd.to_datetime(["2026-01-05", "2026-01-12"])
    weekly_features = pd.DataFrame(
        {
            "week_start": weeks,
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )
    weekly_metadata = pd.DataFrame(
        {
            "week_start": weeks,
            "forecast_weekly_net_demand_mwh": [1000.0, 1200.0],
            "actual_weekly_net_demand_mwh": [980.0, 1190.0],
        }
    )
    policy_state_trace = pd.DataFrame(
        {
            "week_start": weeks,
            "renewable_mechanism_active": [0.0, 1.0],
            "lt_price_linked_active": [0.0, 0.0],
        }
    )
    hourly = pd.DataFrame(
        {
            "week_start": [weeks[0], weeks[0], weeks[1], weeks[1]],
            "hour_index": [0, 1, 0, 1],
            "net_load_da": [100.0, 110.0, 120.0, 130.0],
            "net_load_id": [98.0, 112.0, 122.0, 128.0],
            "price_spread": [10.0, 11.0, 9.0, 8.0],
            "load_dev": [2.0, -1.0, 3.0, -2.0],
            "renewable_dev": [5.0, 6.0, 4.0, 3.0],
        }
    )
    quarter = pd.DataFrame(
        {
            "week_start": [weeks[0], weeks[0], weeks[1], weeks[1]],
            "interval_index": [0, 1, 0, 1],
            "net_load_id_mwh": [24.0, 25.0, 30.0, 31.0],
            "全网统一出清价格_日前": [300.0, 305.0, 320.0, 318.0],
            "全网统一出清价格_日内": [315.0, 310.0, 330.0, 332.0],
        }
    )
    return {
        "weekly_features": weekly_features,
        "weekly_metadata": weekly_metadata,
        "policy_state_trace": policy_state_trace,
        "hourly": hourly,
        "quarter": quarter,
        "agent_feature_columns": ["feature_a", "feature_b"],
    }


def test_compile_training_tensor_bundle_exposes_core_tensors() -> None:
    tensor_bundle = compile_training_tensor_bundle(_build_bundle(), device="cpu")

    assert tensor_bundle.device == "cpu"
    assert tensor_bundle.weekly_feature_tensor.shape == (2, 2)
    assert tensor_bundle.policy_tensor.shape == (2, 2)
    assert tensor_bundle.hourly_tensor.shape == (2, 2, 4)
    assert tensor_bundle.quarter_price_tensor.shape == (2, 2, 2)
    assert tensor_bundle.week_index.tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12")]
    assert isinstance(tensor_bundle.weekly_feature_tensor, torch.Tensor)


def test_compile_training_tensor_bundle_keeps_settlement_targets_and_masks() -> None:
    tensor_bundle = compile_training_tensor_bundle(_build_bundle(), device="cpu")

    assert tensor_bundle.forecast_weekly_load.shape == (2,)
    assert tensor_bundle.actual_weekly_load.shape == (2,)
    assert tensor_bundle.hourly_valid_mask.shape == (2, 2)
    assert tensor_bundle.quarter_valid_mask.shape == (2, 2)
    assert tensor_bundle.hourly_valid_mask.sum().item() == 4
    assert tensor_bundle.quarter_valid_mask.sum().item() == 4
