from __future__ import annotations

import torch

from src.backtest.materialize import materialize_particle_pair
from src.model_layout.compiler import compile_parameter_layout
from src.policy.feasible_domain import compile_feasible_domain
from src.training.tensor_bundle import compile_training_tensor_bundle
from tests.test_v051_score_kernel import _bundle, _config


def test_materialize_reports_hourly_spot_activation_columns() -> None:
    bundle = _bundle()
    config = _config()
    config["score_kernel"] = {
        "contract_position_base_ratio": 0.60,
        "exposure_band_base_ratio": 0.00,
        "hourly_signal": {
            "spread_weight": 0.50,
            "load_dev_weight": 0.00,
            "renewable_weight": 0.00,
            "spread_abs_weight": 0.00,
            "renewable_abs_weight": 0.00,
        },
        "hourly_limit": {
            "base_multiplier": 1.00,
            "shrink_multiplier": 0.00,
        },
        "hourly_gate": {
            "enabled": False,
        },
    }
    bundle["hourly"]["price_spread_lag1"] = [200.0, 200.0]
    bundle["feasible_domain"] = compile_feasible_domain(
        config=config,
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=bundle["policy_state_trace"],
    )
    layout = compile_parameter_layout(config=config, bundle=bundle)
    tensor_bundle = compile_training_tensor_bundle(bundle, device="cpu")
    upper = torch.zeros((1, layout.upper.total_dimension), dtype=torch.float32)
    lower = torch.ones((1, layout.lower.total_dimension), dtype=torch.float32)

    result = materialize_particle_pair(
        tensor_bundle=tensor_bundle,
        upper_particle=upper[0].tolist(),
        lower_particle=lower[0].tolist(),
        strategy_name="test_activation",
        config=config,
        compiled_layout=layout,
    )

    row = result.weekly_results.iloc[0]
    assert row["spot_hedge_abs_mwh_w"] > 0.0
    assert row["spot_hedge_nonzero_hours_w"] > 0
    assert row["spot_hedge_limit_mean_mwh_w"] > 0.0
