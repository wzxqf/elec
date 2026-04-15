from __future__ import annotations

import subprocess
import shutil
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "run_all.sh"
POWERSHELL_SCRIPT_PATH = PROJECT_ROOT / "run_all.ps1"
BAT_SCRIPT_PATH = PROJECT_ROOT / "run_all.bat"


class RunAllScriptTest(unittest.TestCase):
    def test_dry_run_uses_fixed_torch311_pipeline_command(self) -> None:
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
        self.assertIn("mamba run -n torch311 python -m src.scripts.run_pipeline", result.stdout)

    def test_windows_one_click_scripts_exist(self) -> None:
        self.assertTrue(
            POWERSHELL_SCRIPT_PATH.exists(),
            "run_all.ps1 should exist at project root for Windows PowerShell",
        )
        self.assertTrue(BAT_SCRIPT_PATH.exists(), "run_all.bat should exist at project root")

        bat_text = BAT_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("run_all.ps1", bat_text)
        self.assertIn("-ExecutionPolicy Bypass", bat_text)

    def test_powershell_dry_run_uses_torch311_python_directly(self) -> None:
        self.assertTrue(
            POWERSHELL_SCRIPT_PATH.exists(),
            "run_all.ps1 should exist at project root for Windows PowerShell",
        )
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is None:
            self.skipTest("PowerShell is not available in this environment")

        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(POWERSHELL_SCRIPT_PATH),
                "-DryRun",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(" -m src.scripts.run_pipeline", result.stdout)
        self.assertNotIn("mamba run", result.stdout)


if __name__ == "__main__":
    unittest.main()
