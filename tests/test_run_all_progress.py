from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import run_all


def test_read_status_snapshot_returns_defaults_for_missing_file() -> None:
    status_path = Path(".cache") / "tests" / f"missing-status-{uuid4().hex}.json"
    snapshot = run_all.read_status_snapshot(status_path)

    assert snapshot["stage"] == "等待启动"
    assert snapshot["phase_name"] == "初始化"
    assert snapshot["phase_progress"] == 0.0
    assert snapshot["total_progress"] == 0.0


def test_read_status_snapshot_loads_json_payload() -> None:
    status_path = Path(".cache") / "tests" / f"runtime-status-{uuid4().hex}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "stage": "训练",
                "phase_name": "HPSO 参数训练",
                "phase_progress": 0.375,
                "total_progress": 0.125,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshot = run_all.read_status_snapshot(status_path)

    assert snapshot["stage"] == "训练"
    assert snapshot["phase_name"] == "HPSO 参数训练"
    assert snapshot["phase_progress"] == 0.375
    assert snapshot["total_progress"] == 0.125


def test_format_progress_line_includes_stage_progress_and_resources() -> None:
    line = run_all.format_progress_line(
        {
            "stage": "训练",
            "phase_name": "HPSO 参数训练",
            "phase_progress": 0.5,
            "total_progress": 0.25,
            "message": "iter 40/80",
        },
        {
            "process_cpu_percent": 88.4,
            "process_memory_gb": 3.2,
            "gpu_util_percent": 67.0,
            "gpu_memory_used_gb": 5.5,
            "gpu_memory_total_gb": 16.0,
        },
        elapsed_seconds=125.0,
        width=220,
    )

    assert "训练" in line
    assert "HPSO 参数训练" in line
    assert "总进度 25.0%" in line
    assert "阶段 50.0%" in line
    assert "CPU 88.4%" in line
    assert "GPU 67.0%" in line
    assert "显存 5.5/16.0GB" in line
    assert "iter 40/80" in line
    assert "\n" not in line
