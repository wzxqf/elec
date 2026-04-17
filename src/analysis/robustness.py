from __future__ import annotations

import pandas as pd

from src.analysis.report_contracts import build_summary_scope_lines, infer_date_range


def run_robustness_analysis(*, weekly_results: pd.DataFrame, config: dict) -> pd.DataFrame:
    robustness_cfg = config.get("robustness", {})
    frame = weekly_results.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    rows: list[dict[str, object]] = []

    for shift in robustness_cfg.get("contract_ratio_shift", []):
        shifted_profit = frame["profit_w"] - frame["procurement_cost_w"] * abs(float(shift)) * 0.05
        shifted_cvar = frame["cvar99_w"] * (1.0 + abs(float(shift)) * 0.5)
        rows.append(
            {
                "scenario_group": "contract_ratio_shift",
                "scenario_name": f"contract_ratio_shift_{float(shift):+.2f}",
                "scenario_value": float(shift),
                "total_profit": float(shifted_profit.sum()),
                "mean_cvar99": float(shifted_cvar.mean()),
            }
        )

    for scale in robustness_cfg.get("forecast_error_scale", []):
        scaled_profit = frame["profit_w"] - frame["procurement_cost_w"] * abs(float(scale) - 1.0) * 0.10
        scaled_cvar = frame["cvar99_w"] * float(scale)
        rows.append(
            {
                "scenario_group": "forecast_error_scale",
                "scenario_name": f"forecast_error_scale_{float(scale):.2f}",
                "scenario_value": float(scale),
                "total_profit": float(scaled_profit.sum()),
                "mean_cvar99": float(scaled_cvar.mean()),
            }
        )

    for cutoff in robustness_cfg.get("policy_cutoffs", []):
        cutoff_ts = pd.Timestamp(cutoff)
        subset = frame.loc[frame["week_start"] >= cutoff_ts]
        rows.append(
            {
                "scenario_group": "policy_cutoff",
                "scenario_name": f"policy_cutoff_{cutoff_ts.date()}",
                "scenario_value": str(cutoff_ts.date()),
                "total_profit": float(subset["profit_w"].sum()) if not subset.empty else 0.0,
                "mean_cvar99": float(subset["cvar99_w"].mean()) if not subset.empty else 0.0,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["robustness_rank"] = (
        result["total_profit"].rank(method="dense", ascending=False) + result["mean_cvar99"].rank(method="dense", ascending=True)
    )
    result = result.sort_values(["robustness_rank", "scenario_name"]).reset_index(drop=True)
    return result


def build_robustness_summary_markdown(
    robustness_metrics: pd.DataFrame,
    *,
    sample_scope: str = "rolling_backtest_windows",
    week_count: int | None = None,
    aggregation_method: str = "scenario_sweep_over_rolling_results",
    date_range: str | None = None,
) -> str:
    if robustness_metrics.empty:
        lines = ["# 稳健性总结", "", *build_summary_scope_lines(sample_scope=sample_scope, week_count=week_count, aggregation_method=aggregation_method, date_range=date_range or "n/a"), "- 无可用情景。", ""]
        return "\n".join(lines)

    date_range = date_range or infer_date_range(robustness_metrics)
    week_count = week_count if week_count is not None else int(len(robustness_metrics))
    lines = [
        "# 稳健性总结",
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
    for row in robustness_metrics.itertuples(index=False):
        lines.append(
            f"- {row.scenario_name}: group={row.scenario_group}, total_profit={float(row.total_profit):.2f}, mean_cvar99={float(row.mean_cvar99):.2f}, robustness_rank={float(row.robustness_rank):.2f}"
        )
    lines.append("")
    return "\n".join(lines)
