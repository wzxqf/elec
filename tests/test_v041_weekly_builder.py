from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.weekly_builder import build_weekly_bundle


def _build_frame() -> pd.DataFrame:
    dt = pd.date_range("2026-01-26 00:00:00", periods=14 * 24 * 4, freq="15min")
    index = np.arange(len(dt), dtype=float)
    da_price = 320.0 + 5.0 * np.sin(index / 32.0)
    id_price = da_price + 8.0 * np.cos(index / 24.0)
    return pd.DataFrame(
        {
            "datetime": dt,
            "省调负荷_日前": 1000.0 + 20.0 * np.sin(index / 48.0),
            "省调负荷_日内": 1005.0 + 22.0 * np.sin(index / 48.0 + 0.1),
            "新能源负荷-总加_日前": 180.0 + 10.0 * np.cos(index / 36.0),
            "新能源负荷-总加_日内": 182.0 + 12.0 * np.cos(index / 36.0 + 0.2),
            "新能源负荷-风电_日前": 100.0 + 4.0 * np.cos(index / 28.0),
            "新能源负荷-风电_日内": 101.0 + 5.0 * np.cos(index / 28.0 + 0.1),
            "新能源负荷-光伏_日前": 80.0 + 3.0 * np.sin(index / 30.0),
            "新能源负荷-光伏_日内": 81.0 + 4.0 * np.sin(index / 30.0 + 0.15),
            "联络线总加_日前": 50.0,
            "水电出力_日前": 120.0,
            "非市场化机组出力_日前": 60.0,
            "全网统一出清价格_日前": da_price,
            "全网统一出清价格_日内": id_price,
        }
    )


def test_build_weekly_bundle_marks_post_2026_02_lt_price_proxy_switch() -> None:
    bundle = build_weekly_bundle(
        _build_frame(),
        {
            "feature_quantiles": [0.25, 0.5, 0.75],
            "lt_price": {
                "warmup_label": "warmup_unavailable",
                "linked_effective_start": "2026-02-01 00:00:00",
            },
            "analysis_v035": {"price_spike_zscore_threshold": 2.5, "extreme_event_std_threshold": 2.0},
        },
    )

    weekly_metadata = bundle["weekly_metadata"].set_index("week_start")
    linked_week = pd.Timestamp("2026-02-02 00:00:00")

    assert weekly_metadata.loc[linked_week, "lt_price_source"] == "linked_mix_40da_60id"
    assert weekly_metadata.loc[linked_week, "lt_price_fixed_ratio"] == 0.4
    assert weekly_metadata.loc[linked_week, "lt_price_linked_ratio"] == 0.6
