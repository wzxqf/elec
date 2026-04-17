from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.model_layout.schema import CompiledParameterLayout


def stable_hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parameter_layout_audit_markdown(layout: CompiledParameterLayout) -> str:
    lines = [
        "# 参数布局审计",
        "",
        "## 上层参数块",
        "",
    ]
    for block in layout.upper.blocks:
        lines.extend(
            [
                f"- `{block.name}`",
                f"  起止索引: [{block.slice_start}, {block.slice_end})",
                f"  维度: {block.size}",
                f"  字段: {', '.join(block.columns) if block.columns else '(latent)'}",
            ]
        )
    lines.extend(["", "## 下层参数块", ""])
    for block in layout.lower.blocks:
        lines.extend(
            [
                f"- `{block.name}`",
                f"  起止索引: [{block.slice_start}, {block.slice_end})",
                f"  维度: {block.size}",
                f"  字段: {', '.join(block.columns) if block.columns else '(none)'}",
            ]
        )
    forward_consumption = {
        column: block.name
        for block in layout.upper.blocks
        for column in block.columns
        if str(column).startswith("forward_") and str(column).endswith("_days")
    }
    lines.extend(["", "## 前瞻制度状态消费位置", ""])
    if forward_consumption:
        for column, block_name in sorted(forward_consumption.items()):
            lines.append(f"- `{column}` -> `{block_name}`")
    else:
        lines.append("- 无前瞻制度状态接入参数布局。")
    return "\n".join(lines) + "\n"


def build_feasible_domain_summary(domain: Any) -> str:
    weekly = domain.weekly_bounds if hasattr(domain, "weekly_bounds") else pd.DataFrame()
    settlement = domain.settlement_semantics if hasattr(domain, "settlement_semantics") else pd.DataFrame()
    if weekly.empty:
        policy_tightening_week_count = 0
        default_bound_week_count = 0
    else:
        reason_codes = weekly.get("bound_reason_code", pd.Series("default", index=weekly.index)).fillna("default").astype(str)
        policy_tightening_week_count = int(reason_codes.ne("default").sum())
        default_bound_week_count = int(reason_codes.eq("default").sum())
    projection_clip_mean = 0.0
    projection_clip_max = 0.0
    settlement_modes = settlement.get("settlement_mode", pd.Series(dtype="object")).dropna().astype(str).unique().tolist() if not settlement.empty else []
    return "\n".join(
        [
            "# 可行域摘要",
            "",
            "- summary_scope: static_policy_bound_inventory",
            f"- 可行域周数: {len(weekly)}",
            f"- policy_tightening_week_count: {policy_tightening_week_count}",
            f"- default_bound_week_count: {default_bound_week_count}",
            f"- projection_clip_mean: {projection_clip_mean:.4f}",
            f"- projection_clip_max: {projection_clip_max:.4f}",
            f"- 结算模式: {', '.join(settlement_modes) if settlement_modes else '无'}",
            "- runtime_projection_reference: reports/constraint_activation_report.md, metrics/ablation_metrics.csv",
            "- note: 本摘要仅统计编译期静态边界，不统计运行时投影/裁剪触发次数。",
            "",
        ]
    )


def build_run_metadata(config: dict[str, Any], output_root: Path, compiled_layout_payload: dict[str, Any], constraints: pd.DataFrame) -> dict[str, Any]:
    config_path = Path(str(config["config_path"]))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    compiled_layout_hash = stable_hash_payload(compiled_layout_payload)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    experiment_id = f"{config['version']}-{config_hash[:8]}"
    enabled_constraints = []
    if not constraints.empty and "constraint_id" in constraints.columns:
        enabled_constraints = [str(value) for value in constraints["constraint_id"].astype(str).tolist()]
    return {
        "version": config["version"],
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "compiled_layout_hash": compiled_layout_hash,
        "run_timestamp": timestamp,
        "device": str(config["training"]["device"]),
        "data_range": {
            "sample_start": str(config["sample_start"]),
            "sample_end": str(config["sample_end"]),
        },
        "config_path": str(config_path),
        "output_root": str(output_root),
        "enabled_constraints": enabled_constraints,
    }


def build_release_manifest(run_metadata: dict[str, Any], config: dict[str, Any], key_outputs: dict[str, str]) -> dict[str, Any]:
    return {
        "version": run_metadata["version"],
        "experiment_id": run_metadata["experiment_id"],
        "config_hash": run_metadata["config_hash"],
        "compiled_layout_hash": run_metadata["compiled_layout_hash"],
        "run_timestamp": run_metadata["run_timestamp"],
        "device": run_metadata["device"],
        "data_range": run_metadata["data_range"],
        "seed": int(config["seed"]),
        "split": {
            "train_start_week": str(config["split"]["train_start_week"]),
            "train_end_week": str(config["split"]["train_end_week"]),
            "val_start_week": str(config["split"]["val_start_week"]),
            "val_end_week": str(config["split"]["val_end_week"]),
            "test_start_week": str(config["split"]["test_start_week"]),
            "test_end_week": str(config["split"]["test_end_week"]),
        },
        "compiled_layout_path": key_outputs.get("compiled_layout_path", ""),
        "enabled_constraints": run_metadata["enabled_constraints"],
        "key_outputs": key_outputs,
    }


def build_run_manifest(run_metadata: dict[str, Any], config: dict[str, Any], key_outputs: dict[str, str]) -> dict[str, Any]:
    return {
        "version": run_metadata["version"],
        "experiment_id": run_metadata["experiment_id"],
        "config_hash": run_metadata["config_hash"],
        "run_timestamp": run_metadata["run_timestamp"],
        "device": run_metadata["device"],
        "data_range": run_metadata["data_range"],
        "algorithm": str(config["training"]["algorithm"]),
        "seed": int(config["seed"]),
        "key_outputs": key_outputs,
    }


def build_artifact_index_markdown(run_metadata: dict[str, Any], key_outputs: dict[str, str]) -> str:
    lines = [
        "# Artifact Index",
        "",
        f"- version: {run_metadata['version']}",
        f"- experiment_id: {run_metadata['experiment_id']}",
        f"- config_hash: {run_metadata['config_hash']}",
        "",
    ]
    for label, path in sorted(key_outputs.items()):
        lines.append(f"- {label}: {path}")
    lines.append("")
    return "\n".join(lines)


def relativize_path(path: str | Path, output_root: str | Path) -> str:
    path_obj = Path(path)
    root_obj = Path(output_root)
    try:
        return path_obj.relative_to(root_obj).as_posix()
    except ValueError:
        return path_obj.as_posix()


def prepend_report_header(body: str, run_metadata: dict[str, Any], device: str | None = None) -> str:
    header = [
        "> version: {}".format(run_metadata["version"]),
        "> experiment_id: {}".format(run_metadata["experiment_id"]),
        "> config_hash: {}".format(run_metadata["config_hash"]),
        "> run_timestamp: {}".format(run_metadata["run_timestamp"]),
        "> device: {}".format(device or run_metadata["device"]),
        "> data_range: {} -> {}".format(run_metadata["data_range"]["sample_start"], run_metadata["data_range"]["sample_end"]),
        "",
    ]
    return "\n".join(header + [body.strip(), ""])


def fallback_run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": str(config.get("version", config.get("project", {}).get("version", "unknown"))),
        "experiment_id": "unknown",
        "config_hash": "unknown",
        "run_timestamp": "unknown",
        "device": str(config.get("training", {}).get("device", "cpu")),
        "data_range": {
            "sample_start": str(config.get("sample_start", "")),
            "sample_end": str(config.get("sample_end", "")),
        },
    }


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Unsupported type for JSON serialization: {type(obj)!r}")
