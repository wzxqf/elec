from __future__ import annotations

import pandas as pd


def build_constraint_activation_report_markdown(weekly_results: pd.DataFrame) -> str:
    frame = weekly_results.copy()
    triggered = frame.get("feasible_domain_triggered_w", pd.Series(dtype="float64")).astype(float)
    gaps = frame.get("feasible_domain_clip_gap_w", pd.Series(dtype="float64")).astype(float)
    reason_counts = (
        frame.get("bound_reason_code", pd.Series(dtype="object"))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    lines = [
        "# 约束激活报告",
        "",
        f"- 触发窗口数: {int(triggered.gt(0.0).sum())}",
        f"- 平均裁剪幅度: {float(gaps.mean() if not gaps.empty else 0.0):.4f}",
        f"- 最大裁剪幅度: {float(gaps.max() if not gaps.empty else 0.0):.4f}",
        "",
        "## 原因码分布",
        "",
    ]
    for key, value in reason_counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)

