from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_runtime_config


def test_v033_config_uses_hybrid_pso() -> None:
    config = load_runtime_config(Path.cwd())

    assert config["version"] == "v0.33"
    assert config["training"]["algorithm"] == "HYBRID_PSO_V033"
    assert config["reward"]["cvar_alpha"] == 0.99
    assert config["hybrid_pso"]["upper"]["dimension"] >= 10
    assert config["hybrid_pso"]["lower"]["dimension"] >= 8
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
