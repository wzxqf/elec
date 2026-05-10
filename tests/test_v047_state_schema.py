from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.weekly_builder import build_weekly_bundle
from src.scripts.common import _build_feature_manifest
from src.scripts.common import prepare_project_context


def _build_frame() -> pd.DataFrame:
    dt = pd.date_range("2025-11-03 00:00:00", periods=14 * 24 * 4, freq="15min")
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


def test_build_weekly_bundle_emits_state_columns() -> None:
    bundle = build_weekly_bundle(
        _build_frame(),
        {
            "feature_quantiles": [0.25, 0.5, 0.75],
            "lt_price": {"warmup_label": "warmup_unavailable"},
            "analysis_v035": {"price_spike_zscore_threshold": 2.5, "extreme_event_std_threshold": 2.0},
        },
    )

    assert "business_hour_flag" in bundle["hourly"].columns
    assert "price_spread_abs" in bundle["hourly"].columns
    assert "renewable_dev_abs" in bundle["hourly"].columns
    assert "prev_business_hour_spread_mean" in bundle["weekly_features"].columns
    assert "prev_price_spread_abs_mean" in bundle["weekly_features"].columns
    assert "extreme_event_flag_w" in bundle["weekly_features"].columns


def test_prepare_project_context_exports_tensor_bundle_audit() -> None:
    context = prepare_project_context(".", logger_name="test_prepare_context_state_schema")

    assert (context["output_paths"]["reports"] / "tensor_bundle_audit.md").exists()
    assert (context["output_paths"]["reports"] / "state_schema_snapshot.md").exists()


def test_agent_feature_manifest_excludes_current_week_realized_fields() -> None:
    weekly_features = pd.DataFrame(
        {
            "week_start": [pd.Timestamp("2026-02-02")],
            "prev_spread_mean": [10.0],
            "forecast_weekly_net_demand_mwh": [1000.0],
            "actual_weekly_net_demand_mwh": [990.0],
            "da_id_cross_corr_w": [0.7],
            "extreme_price_spike_flag_w": [1.0],
            "extreme_event_flag_w": [1.0],
        }
    )
    policy_trace = pd.DataFrame(
        {
            "week_start": [pd.Timestamp("2026-02-02")],
            "renewable_mechanism_active": [1.0],
        }
    )
    feature_manifest, agent_columns = _build_feature_manifest(
        weekly_features=weekly_features,
        base_manifest=pd.DataFrame(),
        policy_trace=policy_trace,
        config={
            "feature_selection": {
                "feature_include_for_agent": [],
                "feature_exclude_for_agent": [],
                "feature_keep_for_report_only": [],
            },
            "parameter_compiler": {"upper": {"blocks": {"policy_feature_weights": {}}}},
        },
    )

    blocked = {
        "actual_weekly_net_demand_mwh",
        "da_id_cross_corr_w",
        "extreme_price_spike_flag_w",
        "extreme_event_flag_w",
    }
    assert "prev_spread_mean" in agent_columns
    assert "forecast_weekly_net_demand_mwh" in agent_columns
    assert blocked.isdisjoint(agent_columns)
    blocked_rows = feature_manifest.loc[feature_manifest["column"].isin(blocked)]
    assert blocked_rows["selected_for_agent"].eq(False).all()
    assert blocked_rows["report_only"].eq(True).all()

