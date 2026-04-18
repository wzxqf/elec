from __future__ import annotations

import importlib
from pathlib import Path


def test_build_pytest_args_routes_cache_tmp_and_results_to_single_directory() -> None:
    module = importlib.import_module("src.scripts.run_pytest")

    args = module.build_pytest_args(["tests/test_hpso_param_config.py", "-q"])

    assert f"cache_dir={(module.PYTEST_ROOT / 'cache').as_posix()}" in args
    assert f"--basetemp={(module.PYTEST_ROOT / 'tmp').as_posix()}" in args
    assert f"--junitxml={(module.PYTEST_ROOT / 'results' / 'junit.xml').as_posix()}" in args


def test_run_pytest_cleans_artifacts_on_success(monkeypatch) -> None:
    module = importlib.import_module("src.scripts.run_pytest")
    pytest_root = Path(".cache") / "tests" / "pytest-success"
    cleanup_calls: list[Path] = []

    monkeypatch.setattr(module, "PYTEST_ROOT", pytest_root)
    monkeypatch.setattr(module.pytest, "main", lambda args: 0)
    monkeypatch.setattr(module, "cleanup_pytest_artifacts", lambda root: cleanup_calls.append(root) or True)

    code = module.run_pytest(["tests/test_hpso_param_config.py", "-q"])

    assert code == 0
    assert cleanup_calls == [pytest_root]


def test_run_pytest_keeps_artifacts_on_failure(monkeypatch) -> None:
    module = importlib.import_module("src.scripts.run_pytest")
    pytest_root = Path(".cache") / "tests" / "pytest-failure"

    monkeypatch.setattr(module, "PYTEST_ROOT", pytest_root)
    monkeypatch.setattr(module.pytest, "main", lambda args: 1)

    code = module.run_pytest(["tests/test_hpso_param_config.py", "-q"])

    assert code == 1
    assert pytest_root.exists()
