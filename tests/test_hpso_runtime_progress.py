from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from src.agents.hpso import HPSOSettings, HybridParticleSwarmOptimizer
from src.agents.hpso_param_policy import HPSOParamPolicyModel, train_hpso_param_policy


def test_optimizer_invokes_progress_callback_for_each_iteration() -> None:
    events: list[dict[str, float | int | bool]] = []
    settings = HPSOSettings(
        particles=2,
        iterations=3,
        inertia_weight=0.0,
        cognitive_factor=0.0,
        social_factor=0.0,
        initial_temperature=1.0,
        cooling_rate=0.9,
        perturbation_scale=0.0,
        stagnation_window=10,
        seed=7,
        device="cpu",
        allow_cpu=True,
    )

    def objective(positions: torch.Tensor) -> torch.Tensor:
        return ((positions - 0.25) ** 2).sum(dim=1)

    optimizer = HybridParticleSwarmOptimizer(
        lower=torch.tensor([0.0], dtype=torch.float64),
        upper=torch.tensor([1.0], dtype=torch.float64),
        settings=settings,
        objective=objective,
        progress_callback=events.append,
    )

    optimizer.optimize()

    assert [event["iteration"] for event in events] == [0, 1, 2, 3]
    assert all(event["total_iterations"] == 3 for event in events)


def test_train_hpso_param_policy_reports_runtime_profile(
    monkeypatch,
) -> None:
    config = {
        "version": "v0.32",
        "seed": 42,
        "device": "cpu",
        "env": {"reward_temperature": 2.2},
        "constraints": {"delta_h_max": 0.12},
        "rules": {"gamma_max": 420.0},
        "cost": {
            "cvar_alpha": 0.95,
            "risk_vol_weight": 0.5,
            "risk_cvar_weight": 0.5,
            "transaction_linear": 1.2,
            "transaction_quadratic": 0.015,
            "lt_adjust_cost": 0.6,
        },
        "hpso": {
            "device": "cpu",
            "allow_cpu": True,
            "seed": 42,
            "swarm": {
                "particles": 4,
                "iterations": 2,
                "inertia_weight": 0.7,
                "cognitive_factor": 1.45,
                "social_factor": 1.45,
                "initial_temperature": 1.0,
                "cooling_rate": 0.96,
                "perturbation_scale": 0.05,
                "stagnation_window": 10,
                "backprop_steps": 0,
                "backprop_learning_rate": 0.0,
                "backprop_clip_norm": 5.0,
            },
            "parallel": {
                "worker_count": 3,
            },
            "objective_weights": {
                "procurement_cost": 1.0,
                "dynamic_baseline_gap": 1.15,
                "tail_risk": 0.55,
                "risk": 0.5,
                "transaction": 1.0,
                "hourly_smooth": 0.08,
                "hedge_error": 0.18,
                "action_smooth": 0.12,
            },
        },
    }
    base_dir = Path(".cache") / "tests" / f"hpso-runtime-{uuid4().hex}"
    output_paths = {
        "models": base_dir / "models",
        "metrics": base_dir / "metrics",
    }
    output_paths["models"].mkdir(parents=True)
    output_paths["metrics"].mkdir(parents=True)

    class FakeOptimizer:
        def __init__(self, lower, upper, settings, objective, bounded_mask=None, progress_callback=None):
            self.settings = settings
            self.objective = objective
            self.progress_callback = progress_callback

        def optimize(self):
            if self.progress_callback is not None:
                self.progress_callback({"iteration": 0, "total_iterations": 2, "best_score": 1.0, "temperature": 1.0, "stagnated": False})
                self.progress_callback({"iteration": 1, "total_iterations": 2, "best_score": 0.8, "temperature": 0.96, "stagnated": False})
                self.progress_callback({"iteration": 2, "total_iterations": 2, "best_score": 0.7, "temperature": 0.92, "stagnated": False})
            return torch.zeros(64, dtype=torch.float64), 0.7, pd.DataFrame({"iteration": [0, 1, 2], "best_score": [1.0, 0.8, 0.7]})

    def fake_simulate_param_policy_strategy(bundle, weeks, theta, config, strategy_name="hpso_param_policy"):
        weekly = pd.DataFrame(
            {
                "week_start": list(weeks),
                "hpso_param_objective_w": [10.0 for _ in weeks],
                "reward": [0.1 for _ in weeks],
            }
        )
        return {
            "weekly_results": weekly,
            "hourly_results": pd.DataFrame(),
            "settlement_results": pd.DataFrame(),
            "contract_curve": pd.DataFrame(),
            "information_boundary_audit": pd.DataFrame(),
            "metrics": {"total_procurement_cost": 10.0},
            "actions": {},
            "theta": np.asarray(theta, dtype=float),
        }

    monkeypatch.setattr("src.agents.hpso.HybridParticleSwarmOptimizer", FakeOptimizer)
    monkeypatch.setattr("src.agents.hpso_param_policy.simulate_param_policy_strategy", fake_simulate_param_policy_strategy)

    result = train_hpso_param_policy(
        bundle={},
        train_weeks=[pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-12")],
        config=config,
        output_paths=output_paths,
        persist_artifacts=False,
    )

    assert isinstance(result["model"], HPSOParamPolicyModel)
    assert result["runtime_profile"]["optimizer_device"] == "cpu"
    assert result["runtime_profile"]["rollout_compute_device"] == "cpu"
    assert result["runtime_profile"]["parallel_workers"] == 3
