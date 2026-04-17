from __future__ import annotations

import pandas as pd

from src.analysis.report_contracts import build_summary_scope_lines, infer_date_range


def build_constraint_activation_report_markdown(
    weekly_results: pd.DataFrame,
    *,
    sample_scope: str = "rolling_backtest_windows",
    week_count: int | None = None,
    aggregation_method: str = "window_level_projection_audit",
    date_range: str | None = None,
) -> str:
    frame = weekly_results.copy()
    triggered = frame.get("feasible_domain_triggered_w", pd.Series(dtype="float64")).astype(float)
    gaps = frame.get("feasible_domain_clip_gap_w", pd.Series(dtype="float64")).astype(float)
    projection_rule = frame.get("projection_rule_name", pd.Series(dtype="object")).fillna("").astype(str)
    reason_counts = (
        projection_rule.where(projection_rule.str.len() > 0, frame.get("bound_reason_code", pd.Series(dtype="object")).fillna("unknown").astype(str))
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )
    policy_tightening_trigger_count = int((triggered.gt(0.0) & projection_rule.str.startswith("policy_")).sum())
    default_projection_trigger_count = int(triggered.gt(0.0).sum()) - policy_tightening_trigger_count
    date_range = date_range or infer_date_range(frame)
    week_count = week_count if week_count is not None else int(len(frame))
    lines = [
        "# 约束激活报告",
        "",
    ]
    lines.extend(
        build_summary_scope_lines(
            sample_scope=sample_scope,
            week_count=week_count,
            aggregation_method=aggregation_method,
            date_range=date_range,
        )
    )
    lines.extend(
        [
            f"- policy_tightening_trigger_count: {policy_tightening_trigger_count}",
            f"- default_projection_trigger_count: {default_projection_trigger_count}",
            f"- projection_clip_mean: {float(gaps.mean() if not gaps.empty else 0.0):.4f}",
            f"- projection_clip_max: {float(gaps.max() if not gaps.empty else 0.0):.4f}",
            f"- total_trigger_count: {int(triggered.gt(0.0).sum())}",
            "",
            "## 原因码分布",
            "",
        ]
    )
    for key, value in reason_counts.items():
        lines.append(f"- {key}: {value}")
    detail_frame = frame.loc[
        triggered.gt(0.0),
        [
            column
            for column in [
                "week_start",
                "projection_target_field",
                "projection_rule_name",
                "projection_before",
                "projection_after",
            ]
            if column in frame.columns
        ],
    ].copy()
    if not detail_frame.empty:
        lines.extend(["", "## 触发明细", ""])
        for row in detail_frame.itertuples(index=False):
            week_text = str(getattr(row, "week_start", "n/a"))
            target_text = str(getattr(row, "projection_target_field", "n/a"))
            rule_text = str(getattr(row, "projection_rule_name", "n/a"))
            before_text = float(getattr(row, "projection_before", 0.0))
            after_text = float(getattr(row, "projection_after", 0.0))
            lines.append(
                f"- week={week_text}, projection_target_field={target_text}, projection_rule_name={rule_text}, projection_before={before_text:.4f}, projection_after={after_text:.4f}"
            )
    lines.append("")
    return "\n".join(lines)

