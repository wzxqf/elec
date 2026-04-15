from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.backtest.benchmarks import build_benchmark_actions
from src.backtest.runtime_cache import prepare_runtime_bundle
from src.backtest.simulator import _get_week_frames, simulate_strategy
from src.config.load_config import load_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_test_bundle() -> dict:
    week_starts = [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-09")]

    hourly_rows: list[dict] = []
    quarter_rows: list[dict] = []
    weekly_metadata_rows: list[dict] = []
    weekly_feature_rows: list[dict] = []
    reward_reference_rows: list[dict] = []

    for week_index, week_start in enumerate(week_starts, start=1):
        weekly_forecast = 0.0
        weekly_actual = 0.0
        da_prices = []
        id_prices = []
        for hour_offset in range(2):
            hour = week_start + pd.Timedelta(hours=hour_offset)
            da_price = 320.0 + week_index * 10.0 + hour_offset
            id_price = 330.0 + week_index * 10.0 + hour_offset
            net_load_da = 100.0 + week_index * 10.0 + hour_offset * 5.0
            net_load_id = net_load_da + 6.0
            weekly_forecast += net_load_da
            weekly_actual += net_load_id
            da_prices.append(da_price)
            id_prices.append(id_price)

            hourly_rows.append(
                {
                    "week_start": week_start,
                    "hour": hour,
                    "net_load_da": net_load_da,
                    "net_load_id": net_load_id,
                    "price_spread": id_price - da_price,
                    "load_dev": net_load_id - net_load_da,
                    "renewable_dev": -4.0 - hour_offset,
                    "全网统一出清价格_日前": da_price,
                    "全网统一出清价格_日内": id_price,
                }
            )

            for quarter_offset in range(4):
                quarter_rows.append(
                    {
                        "week_start": week_start,
                        "datetime": hour + pd.Timedelta(minutes=15 * quarter_offset),
                        "net_load_da": net_load_da,
                        "net_load_da_mwh": net_load_da * 0.25,
                        "net_load_id_mwh": net_load_id * 0.25,
                        "全网统一出清价格_日前": da_price,
                        "全网统一出清价格_日内": id_price,
                    }
                )

        weekly_metadata_rows.append(
            {
                "week_start": week_start,
                "is_partial_week": True,
                "forecast_weekly_net_demand_mwh": weekly_forecast,
                "actual_weekly_net_demand_mwh": weekly_actual,
                "da_price_mean": sum(da_prices) / len(da_prices),
                "lt_price_w": 300.0 + week_index,
                "lt_price_linked_active": 0.0,
                "fixed_price_ratio_max": 0.4,
                "linked_price_ratio_min": 0.6,
                "renewable_mechanism_active": 0.0,
                "mechanism_price_floor": 0.0,
                "mechanism_price_ceiling": 0.0,
                "mechanism_volume_ratio_max": 0.0,
                "mechanism_stage_label": "未启用",
                "ancillary_freq_reserve_tight": 0.0,
                "ancillary_peak_shaving_pause": 0.0,
            }
        )
        weekly_feature_rows.append(
            {
                "week_start": week_start,
                "prev_spread_mean": 8.0 + week_index,
                "prev_renewable_ratio_da_mean": 0.2 + 0.05 * week_index,
                "renewable_mechanism_active": 0.0,
                "lt_price_linked_active": 0.0,
            }
        )
        reward_reference_rows.append(
            {
                "week_start": week_start,
                "baseline_cost_w": 100000.0 + week_index * 1000.0,
                "baseline_risk_w": 5000.0 + week_index * 100.0,
            }
        )

    return {
        "quarter": pd.DataFrame(quarter_rows),
        "hourly": pd.DataFrame(hourly_rows),
        "weekly_metadata": pd.DataFrame(weekly_metadata_rows),
        "weekly_features": pd.DataFrame(weekly_feature_rows),
        "reward_reference": pd.DataFrame(reward_reference_rows),
        "reward_robust_stats": {"delta_cost": {"median": 0.0, "iqr": 1.0}},
    }


class BacktestRuntimeCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_runtime_config(PROJECT_ROOT)

    def test_prepare_runtime_bundle_preserves_week_slices(self) -> None:
        bundle = _make_test_bundle()
        cached_bundle = prepare_runtime_bundle(bundle.copy())
        week_start = pd.Timestamp("2026-03-09")

        legacy_quarter, legacy_hourly, legacy_metadata = _get_week_frames(bundle, week_start)
        cached_quarter, cached_hourly, cached_metadata = _get_week_frames(cached_bundle, week_start)

        self.assertEqual(set(cached_bundle["quarter_by_week"].keys()), set(bundle["weekly_metadata"]["week_start"]))
        self.assertEqual(set(cached_bundle["hourly_by_week"].keys()), set(bundle["weekly_metadata"]["week_start"]))
        self.assertEqual(set(cached_bundle["weekly_feature_by_week"].keys()), set(bundle["weekly_features"]["week_start"]))
        self.assertEqual(set(cached_bundle["reward_reference_by_week"].keys()), set(bundle["reward_reference"]["week_start"]))
        pd.testing.assert_frame_equal(legacy_quarter, cached_quarter)
        pd.testing.assert_frame_equal(legacy_hourly, cached_hourly)
        pd.testing.assert_series_equal(legacy_metadata, cached_metadata)

    def test_build_benchmark_actions_cache_lookup_matches_legacy(self) -> None:
        bundle = _make_test_bundle()
        cached_bundle = prepare_runtime_bundle(bundle.copy())
        weeks = sorted(pd.to_datetime(bundle["weekly_features"]["week_start"]).tolist())

        legacy_actions = build_benchmark_actions(weeks, bundle["weekly_features"], self.config)
        cached_actions = build_benchmark_actions(
            weeks,
            bundle["weekly_features"],
            self.config,
            weekly_feature_by_week=cached_bundle["weekly_feature_by_week"],
        )

        self.assertEqual(legacy_actions, cached_actions)

    def test_scaled_simulation_refreshes_runtime_cache(self) -> None:
        bundle = prepare_runtime_bundle(_make_test_bundle())
        weeks = [pd.Timestamp("2026-03-02")]
        actions = {
            weeks[0]: {
                "mode": "absolute",
                "target_lock_ratio": 0.5,
                "exposure_bandwidth": 0.0,
            }
        }

        baseline = simulate_strategy(bundle, weeks, actions, self.config, "baseline")
        scaled = simulate_strategy(bundle, weeks, actions, self.config, "scaled", forecast_error_scale=2.0)

        baseline_week = baseline["weekly_results"].iloc[0]
        scaled_week = scaled["weekly_results"].iloc[0]
        self.assertAlmostEqual(
            scaled_week["forecast_weekly_net_demand_mwh"],
            baseline_week["forecast_weekly_net_demand_mwh"] * 2.0,
        )
        self.assertAlmostEqual(
            scaled_week["q_lt_target_w"],
            baseline_week["q_lt_target_w"] * 2.0,
        )


if __name__ == "__main__":
    unittest.main()
