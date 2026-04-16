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
    temp_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(status_path)


class RuntimeStatusTracker:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def update(self, **payload: Any) -> None:
        current = read_runtime_status(self.path)
        current.update(payload)
        write_runtime_status(self.path, current)
