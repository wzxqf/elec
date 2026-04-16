from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_runtime_config


def test_v033_config_uses_hybrid_pso() -> None:
    config = load_runtime_config(Path.cwd())

    assert config["version"] == "v0.36"
    assert config["training"]["algorithm"] == "HYBRID_PSO_V036"
    assert config["reward"]["cvar_alpha"] == 0.99
    assert "parameter_compiler" in config
    assert "blocks" in config["parameter_compiler"]["upper"]
    assert "hourly_feature_groups" in config["parameter_compiler"]["lower"]
    assert "policy_projection" in config
    assert "upper_strategy" in config
    assert "lower_strategy" in config
    assert "economics" in config


def test_v033_sections_are_normalized() -> None:
    config = load_runtime_config(Path.cwd())

    assert config["rolling_retrain"]["enabled"] is True
    assert config["policy_deep"]["llm_candidate_parser"]["enabled"] is False
    assert config["upper_strategy"]["contract_curve_hours"] == 24
    assert config["reward"]["baseline_strategy"] == "dynamic_lock_only"
    assert config["parameter_compiler"]["upper"]["blocks"]["contract_curve_latent"]["size"] >= 8
