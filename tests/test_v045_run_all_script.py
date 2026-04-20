from __future__ import annotations

import subprocess
import shutil
import sys
import unittest
from pathlib import Path

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
        self.assertIn(f"Experiment version: {_expected_version()}", result.stdout)
        self.assertIn(" -m src.scripts.run_pipeline", result.stdout)

    def test_python_dry_run_accepts_explicit_experiment_config(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "experiments" / "v0.45_param_opt_balanced.yaml"
        result = subprocess.run(
            [sys.executable, str(PYTHON_SCRIPT_PATH), "--dry-run", "--config", str(config_path.relative_to(PROJECT_ROOT))],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(f"Config path: {config_path.resolve()}", result.stdout)
        self.assertIn("Experiment version: v0.45-param-opt-balanced", result.stdout)
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


if __name__ == "__main__":
    unittest.main()

