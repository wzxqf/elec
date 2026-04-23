from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace
from pathlib import Path

from src.utils.versioning import build_test_filename, load_project_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROJECT_VERSION = load_project_version(PROJECT_ROOT)
CURRENT_CONFIG_TEST = str(Path("tests") / build_test_filename("hpso_param_config", CURRENT_PROJECT_VERSION))


def test_build_pytest_args_routes_cache_tmp_and_results_to_single_directory() -> None:
    module = importlib.import_module("src.scripts.run_pytest")

    args = module.build_pytest_args([CURRENT_CONFIG_TEST, "-q"])

    assert "cache_dir=" not in " ".join(args)
    assert "--assert=plain" in args
    assert "-p" in args
    assert "no:cacheprovider" in args
    assert f"--basetemp={(module.PYTEST_ROOT / 'tmp').as_posix()}" in args
    assert f"--junitxml={(module.PYTEST_ROOT / 'results' / 'junit.xml').as_posix()}" in args


def test_run_pytest_cleans_artifacts_on_success(monkeypatch) -> None:
    module = importlib.import_module("src.scripts.run_pytest")
    pytest_root = Path(".cache") / "tests" / "pytest-success"
    cleanup_calls: list[Path] = []
    fallback_calls: list[Path] = []

    monkeypatch.setattr(module, "PYTEST_ROOT", pytest_root)
    monkeypatch.setattr(module, "invoke_pytest", lambda args: 0)
    monkeypatch.setattr(module, "cleanup_pytest_artifacts", lambda root: cleanup_calls.append(root) or True)
    monkeypatch.setattr(module, "cleanup_pytest_artifacts_fallback", lambda root: fallback_calls.append(root) or True)

    code = module.run_pytest([CURRENT_CONFIG_TEST, "-q"])

    assert code == 0
    assert cleanup_calls == [pytest_root]
    assert fallback_calls == []


def test_run_pytest_uses_fallback_cleanup_when_primary_cleanup_fails(monkeypatch) -> None:
    module = importlib.import_module("src.scripts.run_pytest")
    pytest_root = Path(".cache") / "tests" / "pytest-fallback"
    fallback_calls: list[Path] = []

    monkeypatch.setattr(module, "PYTEST_ROOT", pytest_root)
    monkeypatch.setattr(module, "invoke_pytest", lambda args: 0)
    monkeypatch.setattr(module, "cleanup_pytest_artifacts", lambda root: False)
    monkeypatch.setattr(module, "cleanup_pytest_artifacts_fallback", lambda root: fallback_calls.append(root) or True)

    code = module.run_pytest([CURRENT_CONFIG_TEST, "-q"])

    assert code == 0
    assert fallback_calls == [pytest_root]


def test_run_pytest_keeps_artifacts_on_failure(monkeypatch) -> None:
    module = importlib.import_module("src.scripts.run_pytest")
    pytest_root = Path(".cache") / "tests" / "pytest-failure"

    monkeypatch.setattr(module, "PYTEST_ROOT", pytest_root)
    monkeypatch.setattr(module, "invoke_pytest", lambda args: 1)

    code = module.run_pytest([CURRENT_CONFIG_TEST, "-q"])

    assert code == 1
    assert pytest_root.exists()


def test_invoke_pytest_uses_subprocess_with_pytest_module_and_no_bytecode_env(monkeypatch) -> None:
    module = importlib.import_module("src.scripts.run_pytest")
    captured: dict[str, object] = {}

    def fake_run(command, check, env):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    code = module.invoke_pytest(["tests/test_v045_hpso_param_config.py", "-q"])

    assert code == 0
    assert captured["command"] == [sys.executable, "-m", "pytest", "tests/test_v045_hpso_param_config.py", "-q"]
    assert captured["check"] is False
    assert isinstance(captured["env"], dict)
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"]["PYTEST_ADDOPTS"] == os.environ.get("PYTEST_ADDOPTS", "")


def test_run_pytest_cleans_test_bytecode_before_and_after_execution(monkeypatch) -> None:
    module = importlib.import_module("src.scripts.run_pytest")
    pytest_root = Path(".cache") / "tests" / "pytest-clean-bytecode"
    cleanup_calls: list[Path] = []

    monkeypatch.setattr(module, "PYTEST_ROOT", pytest_root)
    monkeypatch.setattr(module, "invoke_pytest", lambda args: 0)
    monkeypatch.setattr(module, "cleanup_pytest_artifacts", lambda root: True)
    monkeypatch.setattr(module, "cleanup_test_bytecode", lambda root: cleanup_calls.append(root))

    code = module.run_pytest([CURRENT_CONFIG_TEST, "-q"])

    assert code == 0
    assert cleanup_calls == [module.TESTS_ROOT, module.TESTS_ROOT]

