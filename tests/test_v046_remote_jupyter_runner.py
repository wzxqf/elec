from __future__ import annotations

import zipfile
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from src.scripts import run_remote_jupyter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manual_temp_root() -> Path:
    root = Path(".cache") / "remote_jupyter_unit" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root


def test_source_archive_excludes_local_outputs_caches_git_and_secret_files() -> None:
    temp_root = _manual_temp_root()
    try:
        project_root = temp_root / "project"
        project_root.mkdir()
        (project_root / "run_all.py").write_text("print('run')\n", encoding="utf-8")
        (project_root / "experiment_config.yaml").write_text("project:\n  version: v0.46\n", encoding="utf-8")
        (project_root / "src").mkdir()
        (project_root / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (project_root / "tests").mkdir()
        (project_root / "tests" / "test_v046_example.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        (project_root / "outputs" / "v0.46").mkdir(parents=True)
        (project_root / "outputs" / "v0.46" / "local.csv").write_text("local\n", encoding="utf-8")
        (project_root / "已归档" / "outputs" / "v0.44-param-opt-explore" / "logs").mkdir(parents=True)
        (project_root / ".git").mkdir()
        (project_root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        (project_root / ".cache").mkdir()
        (project_root / ".cache" / "pytest.log").write_text("cache\n", encoding="utf-8")
        (project_root / ".env").write_text("ELEC_JUPYTER_PASSWORD=secret\n", encoding="utf-8")

        archive_path = temp_root / "bundle.zip"

        manifest = run_remote_jupyter.build_source_archive(project_root, archive_path, include_outputs=False)

        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())

        assert "run_all.py" in names
        assert "experiment_config.yaml" in names
        assert "src/module.py" in names
        assert "tests/test_v046_example.py" in names
        assert "outputs/v0.46/local.csv" not in names
        assert "已归档/outputs/v0.44-param-opt-explore/" in names
        assert "已归档/outputs/v0.44-param-opt-explore/logs/" in names
        assert ".git/config" not in names
        assert ".cache/pytest.log" not in names
        assert ".env" not in names
        assert manifest["file_count"] == 4
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_remote_execution_code_runs_pipeline_then_pytest_and_emits_result_marker() -> None:
    spec = run_remote_jupyter.RemoteRunSpec(
        run_id="v046-remote-test",
        version="v0.46",
        archive_remote_path="elec_remote_runs/v0.46/v046-remote-test/source.zip",
        remote_project_dir="elec_remote_runs/v0.46/v046-remote-test/project",
        artifact_remote_path="elec_remote_runs/v0.46/v046-remote-test/artifacts.zip",
        pytest_args=["tests/test_v046_repo_cleanup.py", "-q"],
        run_pipeline=True,
        run_tests=True,
    )

    code = run_remote_jupyter.build_remote_execution_code(spec)

    assert "__ELEC_REMOTE_RESULT__" in code
    assert '"run_all.py"' in code
    assert '"src.scripts.run_pytest"' in code
    assert '"tests/test_v046_repo_cleanup.py"' in code
    assert '"-q"' in code
    assert "CONFIG = json.loads(" in code
    compile(code, "<remote_execution>", "exec")


def test_default_download_dir_is_versioned_remote_jupyter_subdirectory() -> None:
    project_root = Path("project-root")

    download_dir = run_remote_jupyter.default_download_dir(project_root, "v0.46", "20260426T120000")

    assert download_dir == project_root / "outputs" / "v0.46" / "remote_jupyter" / "20260426T120000"


def test_sync_pulled_outputs_replaces_local_formal_outputs_and_keeps_remote_history() -> None:
    temp_root = _manual_temp_root()
    try:
        project_root = temp_root / "project"
        target = project_root / "outputs" / "v0.46"
        target.mkdir(parents=True)
        (target / "metrics").mkdir()
        (target / "metrics" / "old.csv").write_text("old\n", encoding="utf-8")
        (target / "stale_manifest.json").write_text("{}\n", encoding="utf-8")
        remote_history = target / "remote_jupyter" / "run-1"
        remote_history.mkdir(parents=True)
        (remote_history / "keep.txt").write_text("history\n", encoding="utf-8")

        pulled_output = remote_history / "outputs" / "v0.46"
        (pulled_output / "metrics").mkdir(parents=True)
        (pulled_output / "reports").mkdir()
        (pulled_output / "metrics" / "new.csv").write_text("new\n", encoding="utf-8")
        (pulled_output / "reports" / "summary.md").write_text("# remote\n", encoding="utf-8")
        (pulled_output / "artifact_index.md").write_text("remote index\n", encoding="utf-8")

        manifest = run_remote_jupyter.sync_pulled_outputs_to_local(remote_history, project_root, "v0.46")

        assert not (target / "metrics" / "old.csv").exists()
        assert not (target / "stale_manifest.json").exists()
        assert (target / "metrics" / "new.csv").read_text(encoding="utf-8") == "new\n"
        assert (target / "reports" / "summary.md").read_text(encoding="utf-8") == "# remote\n"
        assert (target / "artifact_index.md").read_text(encoding="utf-8") == "remote index\n"
        assert (remote_history / "keep.txt").read_text(encoding="utf-8") == "history\n"
        assert manifest["synced_files"] == 3
        assert manifest["preserved_names"] == ["remote_jupyter"]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_connection_config_reads_secret_from_env_without_leaking_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEC_JUPYTER_PASSWORD", "secret-password")
    monkeypatch.delenv("ELEC_JUPYTER_TOKEN", raising=False)
    args = SimpleNamespace(
        url=None,
        kernel=None,
        password_env="ELEC_JUPYTER_PASSWORD",
        token_env="ELEC_JUPYTER_TOKEN",
        timeout=30,
    )

    config = run_remote_jupyter.build_connection_config(args)

    assert config.url == "http://10.26.27.72:9007/"
    assert config.kernel == "python3"
    assert config.password == "secret-password"
    assert "secret-password" not in repr(config)


def test_connection_config_requires_password_or_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEC_JUPYTER_PASSWORD", raising=False)
    monkeypatch.delenv("ELEC_JUPYTER_TOKEN", raising=False)
    args = SimpleNamespace(
        url="http://10.26.27.72:9007/",
        kernel="torch311",
        password_env="ELEC_JUPYTER_PASSWORD",
        token_env="ELEC_JUPYTER_TOKEN",
        timeout=30,
    )

    with pytest.raises(RuntimeError, match="ELEC_JUPYTER_PASSWORD"):
        run_remote_jupyter.build_connection_config(args)


def test_default_http_timeout_allows_large_archive_uploads() -> None:
    args = run_remote_jupyter.parse_args(["--dry-run"])

    assert run_remote_jupyter.DEFAULT_HTTP_TIMEOUT_SECONDS == 300
    assert args.http_timeout == 300


def test_local_output_sync_is_enabled_by_default_and_can_be_disabled() -> None:
    default_args = run_remote_jupyter.parse_args(["--dry-run"])
    disabled_args = run_remote_jupyter.parse_args(["--dry-run", "--no-sync-local-output"])

    assert default_args.no_sync_local_output is False
    assert disabled_args.no_sync_local_output is True


def test_extract_kernel_names_from_jupyter_kernelspec_payload() -> None:
    payload = {
        "kernelspecs": {
            "python3": {"name": "python3"},
            "torch311": {"name": "torch311"},
        }
    }

    assert run_remote_jupyter.extract_kernel_names(payload) == ["python3", "torch311"]


def test_build_probe_code_embeds_remote_env_without_format_errors() -> None:
    code = run_remote_jupyter.build_probe_code("torch311", "/opt/conda/envs/torch311/bin/python")

    assert '"remote_env": "torch311"' in code
    assert '"/opt/conda/envs/torch311/bin/python"' in code
    assert "__ELEC_JUPYTER_PROBE__" in code
    assert "resolve_remote_python_command" in code
    assert "CONFIG = json.loads(" in code
    compile(code, "<probe>", "exec")


def test_remote_python_resolver_includes_research_miniforge_env_path() -> None:
    code = run_remote_jupyter.build_remote_python_resolver_code()

    assert '"/research/miniforge3/envs/{env_name}"' in code


def test_parse_probe_result_reads_probe_marker() -> None:
    output = 'noise\n__ELEC_JUPYTER_PROBE__{"remote_env": "torch311", "remote_python_probe_returncode": 0}\n'

    result = run_remote_jupyter.parse_probe_result(output)

    assert result["remote_env"] == "torch311"
    assert result["remote_python_probe_returncode"] == 0


def test_gitignore_keeps_local_jupyter_credentials_untracked() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".env" in gitignore


def test_root_remote_jupyter_script_delegates_to_module_dry_run() -> None:
    script_path = PROJECT_ROOT / "run_remote_jupyter.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--dry-run", "--run-id", "root-wrapper-test"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Run id: root-wrapper-test" in result.stdout
    assert "Remote root: elec_remote_runs/v0.46/root-wrapper-test" in result.stdout
