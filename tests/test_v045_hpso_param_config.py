from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_runtime_config
from src.utils.versioning import load_project_version


CURRENT_PROJECT_VERSION = load_project_version(Path.cwd())


def test_config_uses_hybrid_pso() -> None:
    config = load_runtime_config(Path.cwd())

    assert config["version"] == CURRENT_PROJECT_VERSION
    assert config["training"]["algorithm"] == "HYBRID_PSO_V040"
    assert config["reward"]["cvar_alpha"] == 0.99
    assert "parameter_compiler" in config
    assert "blocks" in config["parameter_compiler"]["upper"]
    assert "hourly_feature_groups" in config["parameter_compiler"]["lower"]
    assert "policy_projection" in config
    assert "policy_feasible_domain" in config
    assert "upper_strategy" in config
    assert "lower_strategy" in config
    assert "economics" in config


def test_sections_are_normalized() -> None:
    config = load_runtime_config(Path.cwd())

    assert config["rolling_retrain"]["enabled"] is True
    assert config["policy_deep"]["llm_candidate_parser"]["enabled"] is False
    assert config["upper_strategy"]["contract_curve_hours"] == 24
    assert config["reward"]["baseline_strategy"] == "dynamic_lock_only"
    assert config["parameter_compiler"]["upper"]["blocks"]["contract_curve_latent"]["size"] >= 8


def test_runtime_sections_include_hybrid_optimizer_and_score_kernel() -> None:
    config = load_runtime_config(Path.cwd())

    optimizer = config["hybrid_pso"]["optimizer"]
    score_kernel = config["score_kernel"]

    assert optimizer["init_scale"] == 0.10
    assert optimizer["inertia"] == 0.65
    assert optimizer["cognitive"] == 1.35
    assert optimizer["social"] == 1.35
    assert optimizer["position_clip_abs"] == 1.0

    assert score_kernel["contract_position_base_ratio"] == 0.60
    assert score_kernel["exposure_band_base_ratio"] == 0.20
    assert score_kernel["lt_settlement_weight"] == 0.60
    assert score_kernel["da_settlement_weight"] == 0.40
