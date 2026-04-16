from __future__ import annotations

from typing import Any

import pandas as pd

from src.model_layout.schema import CompiledParameterLayout, LayerLayout, ParameterBlockSpec


def _numeric_policy_columns(policy_state_trace: pd.DataFrame) -> list[str]:
    return [
        column
        for column in policy_state_trace.columns
        if column != "week_start" and pd.api.types.is_numeric_dtype(policy_state_trace[column])
    ]


def _resolve_upper_block_columns(
    block_config: dict[str, Any],
    source_registry: dict[str, list[str]],
) -> list[str]:
    source_name = str(block_config.get("source", "")).strip()
    if source_name:
        source_name = {
            "weekly_feature_source": "agent_feature_columns",
            "policy_feature_source": "policy_state_numeric_columns",
        }.get(source_name, source_name)
        if source_name not in source_registry:
            raise KeyError(f"Unknown parameter_compiler upper source: {source_name}")
        columns = list(source_registry[source_name])
    else:
        columns = []

    include = block_config.get("include")
    if include is not None:
        include_columns = [str(column) for column in include]
        missing = [column for column in include_columns if column not in columns]
        if missing:
            raise KeyError(", ".join(missing))
        columns = include_columns

    explicit_size = block_config.get("size")
    if explicit_size is not None:
        size = int(explicit_size)
        if size <= 0:
            raise ValueError("parameter_compiler upper block size must be positive")
        if columns:
            return columns
        return [f"{source_name or 'latent'}_{index}" for index in range(size)]

    if not columns:
        raise ValueError("parameter_compiler upper block must declare source/include or positive size")
    return columns


def compile_parameter_layout(config: dict[str, Any], bundle: dict[str, Any]) -> CompiledParameterLayout:
    compiler_cfg = config["parameter_compiler"]
    upper_cfg = compiler_cfg["upper"]
    lower_cfg = compiler_cfg["lower"]

    agent_feature_columns = [str(column) for column in bundle.get("agent_feature_columns", [])]
    if not agent_feature_columns:
        raise ValueError("agent_feature_columns is empty, cannot compile upper parameter layout")
    policy_columns = _numeric_policy_columns(bundle["policy_state_trace"])

    source_registry = {
        str(upper_cfg.get("weekly_feature_source", "agent_feature_columns")): agent_feature_columns,
        str(upper_cfg.get("policy_feature_source", "policy_state_numeric_columns")): policy_columns,
    }

    upper_blocks: list[ParameterBlockSpec] = []
    cursor = 0
    for block_name, block_config in upper_cfg["blocks"].items():
        columns = _resolve_upper_block_columns(block_config, source_registry)
        next_cursor = cursor + len(columns)
        upper_blocks.append(
            ParameterBlockSpec(
                name=str(block_name),
                columns=columns,
                slice_start=cursor,
                slice_end=next_cursor,
            )
        )
        cursor = next_cursor

    lower_blocks: list[ParameterBlockSpec] = []
    lower_cursor = 0
    for block_name, group_config in lower_cfg["hourly_feature_groups"].items():
        response_size = int(group_config["response_size"])
        if response_size <= 0:
            raise ValueError("parameter_compiler lower response_size must be positive")
        next_cursor = lower_cursor + response_size
        lower_blocks.append(
            ParameterBlockSpec(
                name=str(block_name),
                columns=[str(column) for column in group_config.get("columns", [])],
                slice_start=lower_cursor,
                slice_end=next_cursor,
            )
        )
        lower_cursor = next_cursor

    if not upper_blocks or cursor <= 0:
        raise ValueError("compiled upper parameter layout is empty")
    if not lower_blocks or lower_cursor <= 0:
        raise ValueError("compiled lower parameter layout is empty")

    return CompiledParameterLayout(
        upper=LayerLayout(total_dimension=cursor, blocks=upper_blocks),
        lower=LayerLayout(total_dimension=lower_cursor, blocks=lower_blocks),
        feature_sources={
            "agent_feature_columns": agent_feature_columns,
            "policy_state_numeric_columns": policy_columns,
        },
    )
