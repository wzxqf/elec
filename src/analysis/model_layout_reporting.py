from __future__ import annotations

from src.model_layout.schema import CompiledParameterLayout


def build_parameter_layout_payload(layout: CompiledParameterLayout) -> dict:
    def _layer_payload(layer) -> dict:
        return {
            "total_dimension": layer.total_dimension,
            "blocks": [
                {
                    "name": block.name,
                    "columns": block.columns,
                    "slice_start": block.slice_start,
                    "slice_end": block.slice_end,
                    "size": block.size,
                }
                for block in layer.blocks
            ],
        }

    return {
        "upper": _layer_payload(layout.upper),
        "lower": _layer_payload(layout.lower),
        "feature_sources": layout.feature_sources,
    }


def build_parameter_layout_markdown(layout: CompiledParameterLayout) -> str:
    lines = [
        "# 参数布局摘要",
        "",
        f"- 上层总维度: {layout.upper.total_dimension}",
        f"- 下层总维度: {layout.lower.total_dimension}",
        "",
        "## 上层参数块",
        "",
    ]
    for block in layout.upper.blocks:
        lines.extend(
            [
                f"- `{block.name}`: slice=({block.slice_start}, {block.slice_end}), size={block.size}",
                f"  columns: {', '.join(block.columns) if block.columns else '(latent)'}",
            ]
        )
    lines.extend(["", "## 下层参数块", ""])
    for block in layout.lower.blocks:
        lines.extend(
            [
                f"- `{block.name}`: slice=({block.slice_start}, {block.slice_end}), size={block.size}",
                f"  columns: {', '.join(block.columns) if block.columns else '(none)'}",
            ]
        )
    return "\n".join(lines) + "\n"
