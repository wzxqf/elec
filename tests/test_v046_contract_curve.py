from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.contract_curve import allocate_weekly_contract_curve, build_base_24h_profile


def test_contract_curve_conserves_weekly_energy() -> None:
    hourly = pd.DataFrame(
        {
            "hour": pd.date_range("2026-03-02", periods=168, freq="h"),
            "hour_index_in_week": list(range(168)),
            "net_load_da": np.linspace(100.0, 200.0, 168),
        }
    )

    profile = build_base_24h_profile(hourly)
    result = allocate_weekly_contract_curve(
        hourly,
        q_lt_target=16800.0,
        base_profile=profile,
        curve_params=np.zeros(8),
    )

    assert len(profile) == 24
    assert len(result) == 168
    assert abs(result["q_lt_hourly"].sum() - 16800.0) < 1e-6
    assert {"contract_profile_base", "contract_profile_final", "curve_normalization_factor"}.issubset(result.columns)
