from __future__ import annotations

import pandas as pd


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    printable = frame.copy().fillna("")
    headers = [str(column) for column in printable.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in printable.astype(object).values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_policy_rule_summary_markdown(
    inventory: pd.DataFrame,
    rule_table: pd.DataFrame,
    failures: pd.DataFrame,
) -> str:
    lines = [
        "# 政策规则汇总",
        "",
        "## 文件清单",
        "",
        _frame_to_markdown(inventory[["file_name", "suffix", "publish_time", "effective_start_guess", "parse_status", "parse_note"]]),
        "",
        "## 结构化规则表",
        "",
        _frame_to_markdown(
            rule_table[
                [
                    "rule_id",
                    "policy_name",
                    "effective_start",
                    "rule_type",
                    "state_name",
                    "state_value",
                    "source_file",
                ]
            ]
        ),
        "",
        "## 解析失败文件",
        "",
        _frame_to_markdown(failures if not failures.empty else pd.DataFrame(columns=["source_file", "error"])),
        "",
    ]
    return "\n".join(lines)
