from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_runtime_config
from src.utils.versioning import (
    build_test_filename,
    build_test_prefix,
    build_version_report_filename,
    load_project_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_project_version_reads_root_config() -> None:
    assert load_project_version(PROJECT_ROOT) == load_runtime_config(PROJECT_ROOT)["version"]


def test_build_test_prefix_normalizes_dotted_version() -> None:
    assert build_test_prefix("v0.45") == "test_v045_"


def test_build_test_filename_uses_current_project_version() -> None:
    project_version = load_project_version(PROJECT_ROOT)
    filename = build_test_filename("hpso_param_config", project_version)
    assert (PROJECT_ROOT / "tests" / filename).exists()


def test_build_version_report_filename_uses_project_version() -> None:
    project_version = load_project_version(PROJECT_ROOT)
    assert build_version_report_filename(project_version) == f"{project_version}报告.md"

