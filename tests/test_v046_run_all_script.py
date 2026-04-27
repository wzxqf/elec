from __future__ import annotations

import os
import subprocess
import shutil
import sys
import unittest
from unittest import mock
from pathlib import Path

import run_all
from src.utils.versioning import parse_project_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "run_all.sh"
PYTHON_SCRIPT_PATH = PROJECT_ROOT / "run_all.py"
POWERSHELL_SCRIPT_PATH = PROJECT_ROOT / "run_all.ps1"
BAT_SCRIPT_PATH = PROJECT_ROOT / "run_all.bat"


def _expected_version() -> str:
    return parse_project_version(PROJECT_ROOT / "experiment_config.yaml")


class RunAllScriptTest(unittest.TestCase):
    def test_python_entrypoint_exists_and_windows_wrappers_are_removed(self) -> None:
        self.assertTrue(PYTHON_SCRIPT_PATH.exists(), "run_all.py should exist at project root")
        self.assertFalse(POWERSHELL_SCRIPT_PATH.exists(), "run_all.ps1 should be removed")
        self.assertFalse(BAT_SCRIPT_PATH.exists(), "run_all.bat should be removed")

    def test_python_dry_run_reports_pipeline_command(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PYTHON_SCRIPT_PATH), "--dry-run"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(f"Config path: {(PROJECT_ROOT / 'experiment_config.yaml').resolve()}", result.stdout)
        self.assertIn(f"Experiment version: {_expected_version()}", result.stdout)
        self.assertIn(f"Output dir: {(PROJECT_ROOT / 'outputs' / _expected_version()).resolve()}", result.stdout)
        self.assertIn(" -m src.scripts.run_pipeline", result.stdout)

    def test_shell_wrapper_delegates_to_python_entrypoint(self) -> None:
        self.assertTrue(SCRIPT_PATH.exists(), "run_all.sh should exist at project root")
        if shutil.which("bash") is None:
            self.skipTest("bash is not available in this environment")

        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "--dry-run"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(f"Experiment version: {_expected_version()}", result.stdout)
        self.assertIn(" -m src.scripts.run_pipeline", result.stdout)

    def test_python_resolver_supports_remote_linux_miniforge_env(self) -> None:
        expected = Path("/research/miniforge3/envs/torch311/bin/python")

        def fake_is_file(path: Path) -> bool:
            normalized = str(path).replace("\\", "/")
            return normalized.endswith("/research/miniforge3/envs/torch311/bin/python")

        with (
            mock.patch.object(run_all.sys, "executable", "/usr/bin/python"),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(run_all.Path, "home", return_value=Path("/home/jovyan")),
            mock.patch.object(run_all.Path, "is_file", fake_is_file),
        ):
            self.assertEqual(run_all.resolve_python_executable(), expected)


if __name__ == "__main__":
    unittest.main()

