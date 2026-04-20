from __future__ import annotations

from pathlib import Path

import pytest

from src.config.load_config import load_runtime_config
from src.scripts.common import prepare_project_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config_path", "version", "upper_particles", "upper_dim", "lower_dim"),
    [
        ("configs/experiments/v0.45_param_opt_balanced.yaml", "v0.45-param-opt-balanced", 192, 186, 48),
        ("configs/experiments/v0.45_param_opt_explore.yaml", "v0.45-param-opt-explore", 224, 192, 64),
    ],
)
def test_runtime_config_loads_experiment_overrides(
    config_path: str,
    version: str,
    upper_particles: int,
    upper_dim: int,
    lower_dim: int,
) -> None:
    config = load_runtime_config(PROJECT_ROOT, config_path)

    assert config["version"] == version
    assert config["hybrid_pso"]["upper"]["particles"] == upper_particles
    assert config["hybrid_pso"]["upper"]["dimension"] == upper_dim
    assert config["hybrid_pso"]["lower"]["dimension"] == lower_dim


def test_prepare_project_context_accepts_explicit_config_path() -> None:
    config_path = "configs/experiments/v0.45_param_opt_balanced.yaml"
    context = prepare_project_context(PROJECT_ROOT, logger_name="test_param_opt_balanced", config_path=config_path)

    assert context["config"]["version"] == "v0.45-param-opt-balanced"
    assert context["run_metadata"]["config_path"] == "configs/experiments/v0.45_param_opt_balanced.yaml"
    assert context["run_metadata"]["output_root"] == "outputs/v0.45-param-opt-balanced"

