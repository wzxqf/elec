from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_directories(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(payload: dict[str, Any], path: str | Path) -> None:
    import yaml

    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    def default(obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Unsupported type for JSON serialization: {type(obj)!r}")

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=default)


def save_markdown(text: str, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(text)


def _resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(project_root) / resolved
    return resolved.resolve()


def resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    outputs_config = config["outputs"]
    version = str(config.get("version") or config.get("project", {}).get("version") or "unversioned")
    project_root = config.get("project_root", Path.cwd())

    if "root" in outputs_config:
        version_root = _resolve_project_path(outputs_config["root"], project_root) / version
        outputs = {
            "root": version_root,
            "logs": version_root / "logs",
            "models": version_root / "models",
            "metrics": version_root / "metrics",
            "figures": version_root / "figures",
            "reports": version_root / "reports",
        }
    else:
        outputs = {key: _resolve_project_path(value, project_root) for key, value in outputs_config.items()}

    ensure_directories(outputs.values())
    return outputs


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged
