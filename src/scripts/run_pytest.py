from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PYTEST_ROOT = Path(".cache") / "tests" / "pytest"
TESTS_ROOT = Path("tests")


def build_pytest_args(extra_args: list[str]) -> list[str]:
    return [
        "-p",
        "no:cacheprovider",
        "--assert=plain",
        f"--basetemp={(PYTEST_ROOT / 'tmp').as_posix()}",
        f"--junitxml={(PYTEST_ROOT / 'results' / 'junit.xml').as_posix()}",
        *extra_args,
    ]


def cleanup_pytest_artifacts(root: Path = PYTEST_ROOT) -> bool:
    for _ in range(5):
        shutil.rmtree(root, ignore_errors=True)
        if not root.exists():
            return True
        time.sleep(0.05)
    return not root.exists()


def cleanup_pytest_artifacts_fallback(root: Path = PYTEST_ROOT) -> bool:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        return False
    command = [
        shell,
        "-NoProfile",
        "-Command",
        (
            f"Start-Sleep -Milliseconds 200; "
            f"if (Test-Path -LiteralPath '{root}') "
            f"{{ Remove-Item -LiteralPath '{root}' -Recurse -Force }}"
        ),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    return True


def invoke_pytest(args: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_ADDOPTS"] = env.get("PYTEST_ADDOPTS", "")
    completed = subprocess.run([sys.executable, "-m", "pytest", *args], check=False, env=env)
    return int(completed.returncode)


def cleanup_test_bytecode(root: Path = TESTS_ROOT) -> None:
    if not root.exists():
        return
    for pycache_dir in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(pycache_dir, ignore_errors=True)


def run_pytest(extra_args: list[str]) -> int:
    cleanup_test_bytecode(TESTS_ROOT)
    PYTEST_ROOT.mkdir(parents=True, exist_ok=True)
    (PYTEST_ROOT / "results").mkdir(parents=True, exist_ok=True)
    code = invoke_pytest(build_pytest_args(extra_args))
    cleanup_test_bytecode(TESTS_ROOT)
    if code == 0:
        cleaned = cleanup_pytest_artifacts(PYTEST_ROOT)
        if not cleaned:
            cleanup_pytest_artifacts_fallback(PYTEST_ROOT)
    return code


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return run_pytest(args)


if __name__ == "__main__":
    raise SystemExit(main())
