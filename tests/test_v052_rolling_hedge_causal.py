from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.rolling_hedge import apply_causal_rolling_hedge


def test_causal_rolling_hedge_does_not_require_realized_intraday_columns() -> None:
    hourly = pd.DataFrame(
        {
            "hour": pd.date_range("2026-03-02", periods=24, freq="h"),
            "net_load_da": np.full(24, 100.0),
            "全网统一出清价格_日前": np.full(24, 300.0),
            "price_spread_lag1": np.zeros(24),
            "load_dev_lag1": np.zeros(24),
            "renewable_dev_lag1": np.zeros(24),
        }
    )
    q_lt = pd.Series(np.full(24, 50.0))

    trace, audit = apply_causal_rolling_hedge(
        hourly_frame=hourly,
        q_lt_hourly=q_lt,
        theta=np.zeros(64),
        exposure_bandwidth=0.5,
        policy_state=pd.Series({"ancillary_freq_reserve_tight": 0.0}),
        config={"policy": {"lower": {"non_negative_spot": True, "smooth_limit_mwh": 420.0}}},
    )

    assert "q_spot" in trace.columns
    assert bool(audit.loc[audit["field"] == "net_load_id", "used_for_decision"].iloc[0]) is False
    assert bool(audit.loc[audit["field"] == "全网统一出清价格_日内", "used_for_decision"].iloc[0]) is False
