from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_STATUS = {
    "stage": "等待启动",
    "phase_name": "初始化",
    "phase_progress": 0.0,
    "total_progress": 0.0,
    "message": "",
}


def clamp_progress(value: Any) -> float:
    try:
        progress = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(progress, 0.0), 1.0)


def interpolate_progress(start: float, end: float, phase_progress: Any) -> float:
    progress = clamp_progress(phase_progress)
    return float(start) + (float(end) - float(start)) * progress


def build_training_phase_name(algorithm: Any) -> str:
    label = str(algorithm or "").strip()
    if not label:
        return "训练"
    return f"{label} 训练"


def build_training_progress_message(progress: dict[str, Any]) -> str:
    parts: list[str] = []
    iteration = progress.get("iteration")
    iterations = progress.get("iterations")
    if iteration is not None and iterations is not None:
        try:
            parts.append(f"迭代 {int(iteration)}/{int(iterations)}")
        except (TypeError, ValueError):
            pass
    for label, key in [("最优", "best_score"), ("均值", "mean_score")]:
        value = progress.get(key)
        if value is None:
            continue
        try:
            parts.append(f"{label} {float(value):.4f}")
        except (TypeError, ValueError):
            continue
    return " | ".join(parts) if parts else "训练中"


def read_runtime_status(path: str | Path) -> dict[str, Any]:
    status_path = Path(path)
    if not status_path.exists():
        return dict(DEFAULT_RUNTIME_STATUS)
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_RUNTIME_STATUS)
    return {**DEFAULT_RUNTIME_STATUS, **payload}


def write_runtime_status(path: str | Path, payload: dict[str, Any]) -> None:
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {**DEFAULT_RUNTIME_STATUS, **payload}
    temp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    serialized = json.dumps(merged, ensure_ascii=False, indent=2)
    temp_path.write_text(serialized, encoding="utf-8")
    try:
        temp_path.replace(status_path)
    except PermissionError:
        status_path.write_text(serialized, encoding="utf-8")
        if temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except PermissionError:
                pass


class RuntimeStatusTracker:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def update(self, **payload: Any) -> None:
        current = read_runtime_status(self.path)
        current.update(payload)
        write_runtime_status(self.path, current)
