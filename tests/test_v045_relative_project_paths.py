from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config.load_config import load_runtime_config
from src.scripts.common import prepare_project_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _has_absolute_joined_path(value: str) -> bool:
    for item in str(value).split("|"):
        if item and Path(item).is_absolute():
            return True
    return False


def test_runtime_config_resolves_project_relative_paths() -> None:
    config = load_runtime_config(PROJECT_ROOT)

    assert config["raw_experiment_config"]["project"]["project_root"] == "."
    assert Path(config["project_root"]) == PROJECT_ROOT
    assert Path(config["policy_directory"]) == PROJECT_ROOT / "政策环境"
    assert [Path(path) for path in config["data_candidates"]] == [
        PROJECT_ROOT / "total.csv",
        PROJECT_ROOT / "data" / "total.csv",
    ]
    assert Path(config["outputs"]["root"]) == PROJECT_ROOT / "outputs"


def test_prepare_project_context_exports_project_relative_paths() -> None:
    context = prepare_project_context(PROJECT_ROOT, logger_name="test_relative_project_paths")

    version = context["config"]["version"]
    release_manifest = json.loads((context["output_paths"]["root"] / "release_manifest.json").read_text(encoding="utf-8"))
    inventory = pd.read_csv(context["output_paths"]["metrics"] / "policy_file_inventory.csv")
    rule_table = pd.read_csv(context["output_paths"]["metrics"] / "policy_rule_table.csv")
    state_trace = pd.read_csv(context["output_paths"]["metrics"] / "policy_state_trace.csv").fillna("")

    assert context["run_metadata"]["config_path"] == "experiment_config.yaml"
    assert context["run_metadata"]["output_root"] == f"outputs/{version}"
    assert inventory["source_file"].astype(str).map(lambda value: not Path(value).is_absolute()).all()
    assert rule_table["source_file"].astype(str).map(lambda value: not Path(value).is_absolute()).all()
    assert not state_trace["policy_sources"].astype(str).map(_has_absolute_joined_path).any()
    assert release_manifest["compiled_layout_path"].startswith("reports/")
