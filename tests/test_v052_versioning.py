from __future__ import annotations

from pathlib import Path

from src.config.load_config import load_runtime_config
from src.utils.versioning import (
    build_ai_structured_report_filename,
    build_human_report_filename,
    build_test_filename,
    build_test_prefix,
    load_project_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_project_version_reads_root_config() -> None:
    assert load_project_version(PROJECT_ROOT) == load_runtime_config(PROJECT_ROOT)["version"]


def test_build_test_prefix_normalizes_dotted_version() -> None:
    assert build_test_prefix("v0.52") == "test_v052_"


def test_build_test_filename_uses_current_project_version() -> None:
    project_version = load_project_version(PROJECT_ROOT)
    filename = build_test_filename("hpso_param_config", project_version)
    assert (PROJECT_ROOT / "tests" / filename).exists()


def test_build_report_filenames_use_project_version() -> None:
    project_version = load_project_version(PROJECT_ROOT)
    assert build_human_report_filename(project_version) == f"{project_version}_human_report.md"
    assert build_ai_structured_report_filename(project_version) == f"{project_version}_ai_structured_report.json"
