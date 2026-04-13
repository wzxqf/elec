from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "run_all.sh"


class RunAllScriptTest(unittest.TestCase):
    def test_dry_run_uses_fixed_elec_env_pipeline_command(self) -> None:
        self.assertTrue(SCRIPT_PATH.exists(), "run_all.sh should exist at project root")

        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "--dry-run"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("mamba run -n elec_env python -m src.scripts.run_pipeline", result.stdout)


if __name__ == "__main__":
    unittest.main()
