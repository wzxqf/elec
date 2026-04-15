from __future__ import annotations

import unittest

import pandas as pd
import torch

from src.agents.hpso import (
    HPSOSettings,
    HybridParticleSwarmOptimizer,
    _build_hourly_trace_from_delta,
    simulate_hpso_strategy,
)
from src.backtest.runtime_cache import prepare_runtime_bundle
from src.config.load_config import load_runtime_config


def _make_hpso_test_bundle() -> dict:
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
        for hour_offset in range(3):
            hour = week_start + pd.Timedelta(hours=hour_offset)
            da_price = 310.0 + week_index * 12.0 + hour_offset * 3.0
            id_price = da_price + 5.0 + hour_offset
            net_load_da = 90.0 + week_index * 8.0 + hour_offset * 6.0
            net_load_id = net_load_da + 4.0 + hour_offset
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
                    "renewable_dev": -3.0 - hour_offset,
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
                "id_price_mean": sum(id_prices) / len(id_prices),
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
                "prev_spread_mean": 7.0 + week_index,
                "prev_renewable_ratio_da_mean": 0.25,
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


class HPSOStrategyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_runtime_config(".")
        cls.config["hpso"] = {
            **cls.config["hpso"],
            "upper": {**cls.config["hpso"]["upper"], "particles": 8, "iterations": 5},
            "lower": {**cls.config["hpso"]["lower"], "particles": 8, "iterations": 5},
            "allow_cpu": True,
        }

    def test_hpso_strategy_is_reproducible_and_feasible(self) -> None:
        bundle = prepare_runtime_bundle(_make_hpso_test_bundle())
        weeks = [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-09")]

        first = simulate_hpso_strategy(bundle, weeks, self.config, "hpso")
        second = simulate_hpso_strategy(bundle, weeks, self.config, "hpso")

        pd.testing.assert_frame_equal(first["weekly_results"], second["weekly_results"])
        pd.testing.assert_frame_equal(first["hourly_results"], second["hourly_results"])

        weekly = first["weekly_results"].sort_values("week_start").reset_index(drop=True)
        self.assertTrue(weekly["exposure_bandwidth"].between(0.0, self.config["hpso"]["upper"]["exposure_bandwidth_max"]).all())

        hourly = first["hourly_results"]
        self.assertTrue((hourly["q_spot"] == hourly["q_spot_raw"]).all())
        self.assertEqual(int(weekly["bound_clip_count"].sum()), 0)
        self.assertEqual(int(weekly["smooth_clip_count"].sum()), 0)
        self.assertEqual(int(weekly["non_negative_clip_count"].sum()), 0)
        self.assertEqual(len(first["settlement_results"]), 24)

    def test_backprop_refinement_improves_differentiable_objective(self) -> None:
        settings = HPSOSettings(
            particles=1,
            iterations=1,
            inertia_weight=0.0,
            cognitive_factor=0.0,
            social_factor=0.0,
            initial_temperature=0.0,
            cooling_rate=1.0,
            perturbation_scale=0.0,
            stagnation_window=10,
            seed=7,
            device="cpu",
            allow_cpu=True,
            backprop_steps=20,
            backprop_learning_rate=0.25,
            backprop_clip_norm=10.0,
        )

        def objective(positions: torch.Tensor) -> torch.Tensor:
            return ((positions - 0.25) ** 2).sum(dim=1)

        optimizer = HybridParticleSwarmOptimizer(
            lower=torch.tensor([0.0], dtype=torch.float64),
            upper=torch.tensor([1.0], dtype=torch.float64),
            settings=settings,
            objective=objective,
        )

        best, best_score, convergence = optimizer.optimize()

        self.assertLess(abs(float(best[0].item()) - 0.25), 1e-3)
        self.assertLess(best_score, 1e-6)
        self.assertIn("bp_steps", convergence.columns)
        self.assertGreater(int(convergence["bp_steps"].sum()), 0)
        self.assertGreater(int(convergence["bp_improved_particles"].sum()), 0)
        self.assertTrue((convergence["bp_grad_norm_mean"] >= 0.0).all())

    def test_upper_contract_adjustment_is_not_hard_limited(self) -> None:
        bundle = prepare_runtime_bundle(_make_hpso_test_bundle())
        weeks = [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-09")]

        result = simulate_hpso_strategy(bundle, weeks, self.config, "hpso")
        weekly = result["weekly_results"].sort_values("week_start").reset_index(drop=True)

        self.assertIn("delta_lock_ratio_raw", result["upper_actions"].columns)
        pd.testing.assert_series_equal(
            weekly["delta_lock_ratio_raw"],
            weekly["delta_lock_ratio"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            weekly["lock_ratio_base"] + weekly["delta_lock_ratio_raw"],
            weekly["lock_ratio_final"],
            check_names=False,
        )

    def test_hourly_spot_contract_adjustment_is_not_hard_limited(self) -> None:
        hourly = pd.DataFrame(
            {
                "net_load_da": [100.0, 100.0],
                "net_load_id": [80.0, 140.0],
                "price_spread": [100.0, -100.0],
                "load_dev": [0.0, 0.0],
                "renewable_dev": [0.0, 0.0],
            }
        )
        raw_delta = torch.tensor([-150.0, 250.0], dtype=torch.float64).numpy()

        trace, stats = _build_hourly_trace_from_delta(
            hourly_frame=hourly,
            q_lt_hourly=pd.Series([0.0, 0.0]),
            exposure_bandwidth=0.0,
            policy_state=pd.Series(dtype=float),
            rules_config=self.config["rules"],
            raw_delta=raw_delta,
        )

        expected = trace["q_base"] + raw_delta
        pd.testing.assert_series_equal(trace["q_spot"], expected, check_names=False)
        self.assertEqual(stats["bound_clip_count"], 0)
        self.assertEqual(stats["smooth_clip_count"], 0)
        self.assertEqual(stats["non_negative_clip_count"], 0)


if __name__ == "__main__":
    unittest.main()
