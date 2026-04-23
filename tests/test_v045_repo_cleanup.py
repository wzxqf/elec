from __future__ import annotations

from pathlib import Path
import re

from src.utils.versioning import build_test_prefix, load_project_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
CURRENT_TEST_PREFIX = build_test_prefix(load_project_version(PROJECT_ROOT))
DOC_SUFFIXES = {".md", ".txt", ".rst"}
ARCHIVE_ROOT = PROJECT_ROOT / "已归档"
ACTIVE_DOCS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "CHANGELOG.md",
    PROJECT_ROOT / "v0.45.md",
    PROJECT_ROOT / "docs" / "ARCHITECTURE.md",
    PROJECT_ROOT / "docs" / "CONSTRAINTS.md",
    PROJECT_ROOT / "docs" / "STATE_SCHEMA.md",
    PROJECT_ROOT / "docs" / "agents.md",
    PROJECT_ROOT / "docs" / "v0.45_architecture_implementation.md",
    PROJECT_ROOT / "configs" / "experiments" / "README.md",
]
LEGACY_DOC_PATTERN = re.compile(r"\bppo\b|\bdrl\b|stable-baselines3|gymnasium|sb3\b|gym\b|强化学习", re.IGNORECASE)


def test_legacy_mainline_modules_are_removed() -> None:
    legacy_paths = [
        PROJECT_ROOT / "configs" / "ppo.yaml",
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
        PROJECT_ROOT / "src" / "envs",
        PROJECT_ROOT / "docs" / "V040_SPEC.md",
        PROJECT_ROOT / "docs" / "superpowers",
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


def test_param_opt_experiment_configs_are_archived() -> None:
    archived_configs = ARCHIVE_ROOT / "configs" / "experiments"

    assert (archived_configs / "v0.45_param_opt_balanced.yaml").exists()
    assert (archived_configs / "v0.45_param_opt_explore.yaml").exists()
    assert not (PROJECT_ROOT / "configs" / "experiments" / "v0.45_param_opt_balanced.yaml").exists()
    assert not (PROJECT_ROOT / "configs" / "experiments" / "v0.45_param_opt_explore.yaml").exists()


def test_obsolete_outputs_are_archived() -> None:
    archived_outputs = ARCHIVE_ROOT / "outputs"

    assert (archived_outputs / "v0.44").exists()
    assert (archived_outputs / "v0.44-param-opt-balanced").exists()
    assert (archived_outputs / "v0.44-param-opt-explore").exists()
    assert (archived_outputs / "v0.45-param-opt-explore").exists()
    assert not (PROJECT_ROOT / "outputs" / "v0.44").exists()
    assert not (PROJECT_ROOT / "outputs" / "v0.44-param-opt-balanced").exists()
    assert not (PROJECT_ROOT / "outputs" / "v0.44-param-opt-explore").exists()
    assert not (PROJECT_ROOT / "outputs" / "v0.45-param-opt-explore").exists()


def test_experiment_override_test_is_archived() -> None:
    archived_test = ARCHIVE_ROOT / "tests" / "v0.45" / "test_v045_experiment_config_overrides.py"

    assert archived_test.exists()
    assert not (TESTS_ROOT / "test_v045_experiment_config_overrides.py").exists()


def test_tests_directory_does_not_contain_pycache() -> None:
    assert not (TESTS_ROOT / "__pycache__").exists()


def test_experiment_readme_marks_v045_param_opt_as_archived() -> None:
    readme = (PROJECT_ROOT / "configs" / "experiments" / "README.md").read_text(encoding="utf-8")

    assert "根目录 `experiment_config.yaml` 为唯一正式入口" in readme
    assert "已归档/configs/experiments/" in readme
    assert "v0.45_param_opt_balanced.yaml" in readme
    assert "v0.45_param_opt_explore.yaml" in readme


def test_requirements_do_not_keep_legacy_training_stack() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()

    assert "gymnasium" not in requirements
    assert "stable-baselines3" not in requirements
    assert "tensorboard" not in requirements


def test_current_architecture_doc_exists() -> None:
    assert (PROJECT_ROOT / "docs" / "v0.45_architecture_implementation.md").exists()


def test_legacy_docs_are_archived() -> None:
    archived_docs = ARCHIVE_ROOT / "docs"

    assert (archived_docs / "V040_SPEC.md").exists()
    assert (archived_docs / "superpowers").exists()
    assert not (PROJECT_ROOT / "docs" / "V040_SPEC.md").exists()
    assert not (PROJECT_ROOT / "docs" / "superpowers").exists()


def test_current_output_dir_does_not_keep_transitional_version_labels() -> None:
    output_root = PROJECT_ROOT / "outputs" / load_project_version(PROJECT_ROOT)
    if not output_root.exists():
        return

    bad_paths = [
        str(path.relative_to(PROJECT_ROOT))
        for path in output_root.rglob("*")
        if "param-opt-balanced" in path.name
    ]
    assert bad_paths == []

    metadata_files = [
        output_root / "artifact_index.md",
        output_root / "release_manifest.json",
        output_root / "run_manifest.json",
        output_root / "reports" / "train_config_snapshot.yaml",
    ]
    offenders = []
    for path in metadata_files:
        if path.exists() and "param-opt-balanced" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_active_docs_do_not_describe_legacy_training_stack() -> None:
    offenders = []
    for path in ACTIVE_DOCS:
        text = path.read_text(encoding="utf-8")
        if LEGACY_DOC_PATTERN.search(text):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
