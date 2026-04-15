from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.scripts.common import prepare_project_context
from src.utils.io import save_markdown


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    printable = frame.copy().fillna("")
    headers = [str(column) for column in printable.columns]
    rows = [[str(value) for value in row] for row in printable.astype(object).values.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run_diagnostics(context: dict[str, Any]) -> dict[str, Any]:
    bundle = context["bundle"]
    split = context["split"]
    weekly_metadata = bundle["weekly_metadata"].copy().sort_values("week_start")
    weekly_features = bundle["weekly_features"].copy().sort_values("week_start")

    lines = [
        "# 诊断报告",
        "",
        f"- 15分钟记录数: {len(bundle['quarter'])}",
        f"- 小时记录数: {len(bundle['hourly'])}",
        f"- 周度记录数: {len(weekly_metadata)}",
        f"- 部分周数量: {int(weekly_metadata['is_partial_week'].sum())}",
        f"- 训练周数: {len(split.train)}",
        f"- 验证周数: {len(split.val)}",
        f"- 回测周数: {len(split.test)}",
        "",
        "## 周度元数据样本",
        "",
        _frame_to_markdown(weekly_metadata.head(10)),
        "",
        "## 周度特征样本",
        "",
        _frame_to_markdown(weekly_features.head(5)),
        "",
    ]
    save_markdown("\n".join(lines), context["output_paths"]["reports"] / "diagnostics_report.md")
    return {
        "weekly_metadata": weekly_metadata,
        "weekly_features": weekly_features,
    }


def main() -> dict[str, Any]:
    context = prepare_project_context(Path.cwd(), logger_name="pipeline")
    return run_diagnostics(context)


if __name__ == "__main__":
    main()
