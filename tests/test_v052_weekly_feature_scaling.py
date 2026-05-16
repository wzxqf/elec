from __future__ import annotations

import torch

from src.training.weekly_feature_scaling import scale_weekly_upper_features


def test_unscaled_weekly_mwh_feature_would_saturate_contract_action() -> None:
    feature_value = torch.tensor([[4_699_226.0]], dtype=torch.float32)
    feature_weight = torch.tensor([[0.07]], dtype=torch.float32)
    pre_tanh = 0.15 * (feature_value @ feature_weight.T)

    assert float(pre_tanh.item()) > 10_000.0
    assert float(torch.tanh(pre_tanh).item()) == 1.0


def test_weekly_mwh_features_are_scaled_by_weekly_or_hourly_load() -> None:
    features = torch.tensor(
        [
            [4_699_226.0, 22_154.34, 392.32, 13.0, 1.0],
        ],
        dtype=torch.float32,
    )
    columns = [
        "forecast_weekly_net_demand_mwh",
        "prev_net_load_da_mean",
        "prev_da_price_mean",
        "forward_price_linkage_days",
        "renewable_mechanism_active",
    ]
    forecast_weekly_load = torch.tensor([4_699_226.0], dtype=torch.float32)
    valid_hours = torch.tensor([168.0], dtype=torch.float32)

    scaled = scale_weekly_upper_features(
        features,
        columns=columns,
        forecast_weekly_load=forecast_weekly_load,
        valid_hours=valid_hours,
        config={
            "enabled": True,
            "price_scale_yuan_per_mwh": 500.0,
            "forward_days_scale": 60.0,
        },
    )

    assert scaled.shape == features.shape
    assert float(scaled[0, 0]) == 1.0
    assert 0.79 < float(scaled[0, 1]) < 0.80
    assert 0.78 < float(scaled[0, 2]) < 0.79
    assert 0.21 < float(scaled[0, 3]) < 0.22
    assert float(scaled[0, 4]) == 1.0


def test_ratio_and_flag_columns_keep_native_scale_even_when_name_contains_price() -> None:
    features = torch.tensor([[0.40, 1.0, 390.0]], dtype=torch.float32)
    columns = [
        "fixed_price_ratio_max",
        "extreme_price_spike_flag",
        "mechanism_price_ceiling",
    ]

    scaled = scale_weekly_upper_features(
        features,
        columns=columns,
        forecast_weekly_load=torch.tensor([4_699_226.0], dtype=torch.float32),
        valid_hours=torch.tensor([168.0], dtype=torch.float32),
        config={"enabled": True, "price_scale_yuan_per_mwh": 500.0},
    )

    assert 0.399 < float(scaled[0, 0]) < 0.401
    assert float(scaled[0, 1]) == 1.0
    assert 0.77 < float(scaled[0, 2]) < 0.79


def test_scaled_weekly_feature_keeps_contract_action_inside_continuous_region() -> None:
    features = torch.tensor([[4_699_226.0]], dtype=torch.float32)
    columns = ["forecast_weekly_net_demand_mwh"]
    scaled = scale_weekly_upper_features(
        features,
        columns=columns,
        forecast_weekly_load=torch.tensor([4_699_226.0], dtype=torch.float32),
        valid_hours=torch.tensor([168.0], dtype=torch.float32),
        config={"enabled": True, "clip_abs": 5.0},
    )
    weight = torch.tensor([[0.07]], dtype=torch.float32)
    pre_tanh = 0.15 * (scaled @ weight.T)
    raw_delta_ratio = 0.30 * torch.tanh(pre_tanh)

    assert 0.0 < float(raw_delta_ratio.item()) < 0.01
