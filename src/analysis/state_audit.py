from __future__ import annotations

from typing import Any

from src.training.tensor_bundle import TrainingTensorBundle


def build_state_schema_markdown(bundle: dict[str, Any], tensor_bundle: TrainingTensorBundle) -> str:
    lines = [
        "# STATE_SCHEMA",
        "",
        "## Weekly",
        "",
    ]
    for column in tensor_bundle.weekly_feature_columns:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Policy", ""])
    for column in tensor_bundle.policy_columns:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Hourly", ""])
    for column in tensor_bundle.hourly_feature_columns:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Quarter", ""])
    for column in tensor_bundle.quarter_feature_columns:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Bounds", ""])
    for column in tensor_bundle.weekly_bound_columns:
        lines.append(f"- weekly `{column}`")
    for column in tensor_bundle.hourly_bound_columns:
        lines.append(f"- hourly `{column}`")
    lines.append("")
    return "\n".join(lines)


def build_tensor_bundle_audit_markdown(bundle: dict[str, Any], tensor_bundle: TrainingTensorBundle) -> str:
    return "\n".join(
        [
            "# Tensor Bundle Audit",
            "",
            f"- weekly_feature_shape: {tuple(tensor_bundle.weekly_feature_tensor.shape)}",
            f"- policy_shape: {tuple(tensor_bundle.policy_tensor.shape)}",
            f"- hourly_shape: {tuple(tensor_bundle.hourly_tensor.shape)}",
            f"- quarter_shape: {tuple(tensor_bundle.quarter_price_tensor.shape)}",
            f"- weekly_bound_shape: {tuple(tensor_bundle.weekly_bound_tensor.shape)}",
            f"- hourly_bound_shape: {tuple(tensor_bundle.hourly_bound_tensor.shape)}",
            "",
            "## Columns",
            "",
            f"- weekly_feature_columns: {', '.join(tensor_bundle.weekly_feature_columns)}",
            f"- policy_columns: {', '.join(tensor_bundle.policy_columns)}",
            f"- hourly_feature_columns: {', '.join(tensor_bundle.hourly_feature_columns)}",
            f"- quarter_feature_columns: {', '.join(tensor_bundle.quarter_feature_columns)}",
            f"- weekly_bound_columns: {', '.join(tensor_bundle.weekly_bound_columns)}",
            f"- hourly_bound_columns: {', '.join(tensor_bundle.hourly_bound_columns)}",
            "",
        ]
    )

