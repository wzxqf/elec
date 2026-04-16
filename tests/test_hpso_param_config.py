from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_runtime_config


def test_v032_config_uses_hpso_param_policy() -> None:
    config = load_runtime_config(Path.cwd())

    assert config["version"] == "v0.32"
    assert config["training"]["algorithm"] == "HPSO_PARAM_POLICY"
    assert config["hpso"]["parameter_dimension"] == 64
    assert "n_steps" not in config["training"]
    assert "gae_lambda" not in config["training"]


def test_hpso_param_sections_are_normalized() -> None:
    config = load_runtime_config(Path.cwd())

    assert config["hpso"]["theta_layout"]["upper_total"] == 40
    assert config["hpso"]["theta_layout"]["lower_total"] == 24
    assert config["policy"]["contract_curve"]["periods"] == 24
