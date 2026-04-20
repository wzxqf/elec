from __future__ import annotations

from pathlib import Path

from src.scripts.common import prepare_project_context


def test_prepare_project_context_compiles_and_exports_parameter_layout() -> None:
    context = prepare_project_context(Path.cwd(), logger_name="test_prepare_context")

    assert "compiled_parameter_layout" in context["bundle"]
    layout = context["bundle"]["compiled_parameter_layout"]
    assert layout.upper.total_dimension >= 128
    assert layout.lower.total_dimension >= 32
    assert (context["output_paths"]["reports"] / "compiled_parameter_layout.json").exists()
    assert (context["output_paths"]["reports"] / "parameter_layout_summary.md").exists()
