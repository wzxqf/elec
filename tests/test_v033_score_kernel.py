from __future__ import annotations

import pandas as pd
import torch

from src.training.score_kernel import batch_score_particles
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
                "da_price_mean": [300.0, 320.0],
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


def test_batch_score_particles_returns_cartesian_score_matrix() -> None:
    tensor_bundle = compile_training_tensor_bundle(_bundle(), device="cpu")
    upper_particles = torch.zeros((3, 12), dtype=torch.float32)
    lower_particles = torch.zeros((2, 8), dtype=torch.float32)

    result = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper_particles,
        lower_particles=lower_particles,
        device="cpu",
    )

    assert result.total_score.shape == (3, 2)
    assert result.procurement_cost.shape == (3, 2)
    assert result.profit.shape == (3, 2)
    assert result.reward.shape == (3, 2)
    assert result.cvar99.shape == (3, 2)
    assert torch.isfinite(result.total_score).all()


def test_batch_score_particles_stays_in_tensor_mode() -> None:
    tensor_bundle = compile_training_tensor_bundle(_bundle(), device="cpu")
    upper_particles = torch.randn((2, 12), dtype=torch.float32)
    lower_particles = torch.randn((2, 8), dtype=torch.float32)

    result = batch_score_particles(
        tensor_bundle=tensor_bundle,
        upper_particles=upper_particles,
        lower_particles=lower_particles,
        device="cpu",
    )

    assert isinstance(result.total_score, torch.Tensor)
    assert isinstance(result.contract_adjustment_mwh_raw, torch.Tensor)
    assert isinstance(result.contract_position_mwh, torch.Tensor)
    assert isinstance(result.exposure_band_mwh, torch.Tensor)
    assert isinstance(result.spot_hedge_mwh, torch.Tensor)
    assert result.contract_adjustment_mwh_raw.shape[:2] == (2, 2)
    assert result.spot_hedge_mwh.shape[:3] == (2, 2, 2)
