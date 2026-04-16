from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_mainline_modules_are_removed() -> None:
    legacy_paths = [
        PROJECT_ROOT / "src" / "agents" / "hpso.py",
        PROJECT_ROOT / "src" / "backtest" / "simulator.py",
        PROJECT_ROOT / "src" / "backtest" / "benchmarks.py",
        PROJECT_ROOT / "src" / "rules" / "hourly_hedge.py",
        PROJECT_ROOT / "src" / "analysis" / "sensitivity.py",
        PROJECT_ROOT / "tests" / "test_hpso_strategy.py",
        PROJECT_ROOT / "tests" / "test_backtest_runtime_cache.py",
    ]

    existing = [str(path.relative_to(PROJECT_ROOT)) for path in legacy_paths if path.exists()]
    assert existing == []
