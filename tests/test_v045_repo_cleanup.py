from __future__ import annotations

from pathlib import Path

from src.utils.versioning import build_test_prefix, load_project_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
CURRENT_TEST_PREFIX = build_test_prefix(load_project_version(PROJECT_ROOT))
DOC_SUFFIXES = {".md", ".txt", ".rst"}


def test_legacy_mainline_modules_are_removed() -> None:
    legacy_paths = [
        PROJECT_ROOT / "src" / "agents" / "hpso.py",
        PROJECT_ROOT / "src" / "backtest" / "simulator.py",
        PROJECT_ROOT / "src" / "backtest" / "benchmarks.py",
        PROJECT_ROOT / "src" / "rules" / "hourly_hedge.py",
        PROJECT_ROOT / "src" / "analysis" / "sensitivity.py",
        PROJECT_ROOT / "tests" / "test_hpso_strategy.py",
        PROJECT_ROOT / "tests" / "test_backtest_runtime_cache.py",
        PROJECT_ROOT / "tests" / "test_v033_hybrid_pso.py",
        PROJECT_ROOT / "tests" / "test_v033_materialize.py",
        PROJECT_ROOT / "tests" / "test_v033_rolling_pipeline.py",
        PROJECT_ROOT / "tests" / "test_v033_score_kernel.py",
        PROJECT_ROOT / "tests" / "test_v033_tensor_bundle.py",
        PROJECT_ROOT / "tests" / "test_v035_excess_return_analysis.py",
        PROJECT_ROOT / "tests" / "test_v035_materialize.py",
        PROJECT_ROOT / "tests" / "test_v035_module1_analysis.py",
        PROJECT_ROOT / "tests" / "test_v035_script_outputs.py",
        PROJECT_ROOT / "tests" / "test_v035_weekly_builder.py",
        PROJECT_ROOT / "tests" / "test_v036_hybrid_pso.py",
        PROJECT_ROOT / "tests" / "test_v036_parameter_compiler.py",
        PROJECT_ROOT / "tests" / "test_v036_prepare_context.py",
        PROJECT_ROOT / "tests" / "test_v036_score_kernel.py",
    ]

    existing = [str(path.relative_to(PROJECT_ROOT)) for path in legacy_paths if path.exists()]
    assert existing == []


def test_active_tests_use_current_version_prefix() -> None:
    invalid = sorted(path.name for path in TESTS_ROOT.glob("test_*.py") if not path.name.startswith(CURRENT_TEST_PREFIX))
    assert invalid == []


def test_tests_directory_does_not_contain_doc_files() -> None:
    docs = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in TESTS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in DOC_SUFFIXES
    )
    assert docs == []
