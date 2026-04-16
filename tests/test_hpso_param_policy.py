from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.hpso_param_policy import infer_upper_action, split_theta


def test_split_theta_matches_config_layout() -> None:
    theta = np.arange(64, dtype=float)

    parts = split_theta(theta)

    assert parts.upper.lock_delta_weights.shape == (16,)
    assert parts.upper.bandwidth_weights.shape == (12,)
    assert parts.upper.curve_shape_weights.shape == (8,)
    assert parts.upper.policy_gate_weights.shape == (4,)
    assert parts.lower.spread_response.shape == (6,)
    assert parts.lower.load_forecast_response.shape == (6,)
    assert parts.lower.renewable_response.shape == (4,)
    assert parts.lower.smoothing_response.shape == (4,)
    assert parts.lower.policy_band_shrink.shape == (4,)


def test_infer_upper_action_is_bounded_and_keeps_required_fields() -> None:
    theta = np.zeros(64, dtype=float)
    row = pd.Series(
        {
            "prev_spread_mean": 10.0,
            "prev_da_price_mean": 300.0,
            "prev_id_price_mean": 310.0,
            "prev_load_dev_std": 20.0,
            "prev_renewable_ratio_da_mean": 0.25,
            "renewable_mechanism_active": 1.0,
            "lt_price_linked_active": 1.0,
            "forward_price_linkage_days": 7.0,
            "forward_mechanism_execution_days": 0.0,
        }
    )

    action = infer_upper_action(theta, row, lock_ratio_base=0.55, previous_lock_ratio=0.50)

    assert {"lock_ratio_base", "delta_lock_ratio", "lock_ratio_final", "exposure_bandwidth"}.issubset(action)
    assert 0.0 <= action["exposure_bandwidth"] <= 0.85
    assert 0.10 <= action["lock_ratio_final"] <= 0.85


def test_infer_upper_action_handles_duplicate_feature_names() -> None:
    theta = np.zeros(64, dtype=float)
    row = pd.Series(
        [8.0, 0.0, 1.0],
        index=["prev_spread_mean", "renewable_mechanism_active", "renewable_mechanism_active"],
    )

    action = infer_upper_action(theta, row, lock_ratio_base=0.55, previous_lock_ratio=0.50)

    assert isinstance(action["lock_ratio_final"], float)
