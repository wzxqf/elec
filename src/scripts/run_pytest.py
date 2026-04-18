from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


PYTEST_ROOT = Path(".cache") / "tests" / "pytest"


def build_pytest_args(extra_args: list[str]) -> list[str]:
    return [
        "-o",
        f"cache_dir={(PYTEST_ROOT / 'cache').as_posix()}",
        f"--basetemp={(PYTEST_ROOT / 'tmp').as_posix()}",
        f"--junitxml={(PYTEST_ROOT / 'results' / 'junit.xml').as_posix()}",
        *extra_args,
    ]


def cleanup_pytest_artifacts(root: Path = PYTEST_ROOT) -> bool:
    shutil.rmtree(root, ignore_errors=True)
    return not root.exists()


def run_pytest(extra_args: list[str]) -> int:
    PYTEST_ROOT.mkdir(parents=True, exist_ok=True)
    (PYTEST_ROOT / "results").mkdir(parents=True, exist_ok=True)
    code = int(pytest.main(build_pytest_args(extra_args)))
    if code == 0:
        cleanup_pytest_artifacts(PYTEST_ROOT)
    return code


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return run_pytest(args)


if __name__ == "__main__":
    raise SystemExit(main())
