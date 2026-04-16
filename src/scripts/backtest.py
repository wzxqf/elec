from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.excess_return import build_policy_risk_adjusted_metrics, summarize_rolling_excess_return
from src.analysis.module1 import enrich_weekly_results
from src.agents.hybrid_pso import HybridPSOModel, train_hybrid_pso_model
from src.backtest.materialize import materialize_particle_pair
from src.backtest.rolling_pipeline import summarize_rolling_results
from src.scripts.common import prepare_project_context, subset_bundle_for_weeks
from src.utils.io import save_markdown
from src.utils.logger import configure_logging


def _build_backtest_summary(summary: dict[str, Any], rolling_excess_return_metrics: pd.DataFrame | None = None) -> str:
    rolling_excess_return_metrics = rolling_excess_return_metrics if rolling_excess_return_metrics is not None else pd.DataFrame()
    avg_sharpe = float(
        rolling_excess_return_metrics.get("window_policy_risk_adjusted_sharpe", pd.Series(dtype="float64")).mean() or 0.0
    )
    persistent_windows = int(
        rolling_excess_return_metrics.get("active_excess_return_persistent", pd.Series(dtype="bool")).astype(bool).sum()
    )
    return "\n".join(
        [
            "# 回测摘要",
            "",
            f"- 滚动窗口数: {summary['aggregate']['window_count']:.0f}",
            f"- 平均测试采购成本: {summary['aggregate']['mean_total_procurement_cost']:.2f}",
            f"- 平均测试经济收益: {summary['aggregate'].get('mean_total_profit', 0.0):.2f}",
            f"- 平均测试 CVaR99: {summary['aggregate']['mean_cvar99']:.2f}",
            f"- 窗口政策风险调整后夏普均值: {avg_sharpe:.4f}",
            f"- 持续跑赢窗口数: {persistent_windows}",
            "",
        ]
    )


def run_backtest(context: dict[str, Any], model=None) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="backtest")
    logger.info("开始执行 v0.33 回测模块。")
    window_records: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    backtest_frames: list[pd.DataFrame] = []
    policy_metric_frames: list[pd.DataFrame] = []
    epsilon = float(context["config"].get("analysis_v035", {}).get("sharpe_epsilon", 1.0e-6))

    for window in context["rolling_plan"]:
        train_bundle = subset_bundle_for_weeks(context["bundle"], window.train_weeks)
        val_bundle = subset_bundle_for_weeks(context["bundle"], window.val_weeks)
        test_bundle = subset_bundle_for_weeks(context["bundle"], window.test_weeks)
        training = train_hybrid_pso_model(train_bundle["tensor_bundle"], context["config"])
        window_model: HybridPSOModel = training.model
        val_result = materialize_particle_pair(
            tensor_bundle=val_bundle["tensor_bundle"],
            upper_particle=window_model.upper_best,
            lower_particle=window_model.lower_best,
            strategy_name=f"{window.window_name}_validation",
            config=context["config"],
        )
        test_result = materialize_particle_pair(
            tensor_bundle=test_bundle["tensor_bundle"],
            upper_particle=window_model.upper_best,
            lower_particle=window_model.lower_best,
            strategy_name=f"{window.window_name}_test",
            config=context["config"],
        )
        window_records.append(
            {
                "window_name": window.window_name,
                "train_weeks": window.train_weeks,
                "val_weeks": window.val_weeks,
                "test_weeks": window.test_weeks,
                "best_score": window_model.best_score,
                "total_procurement_cost": test_result.metrics["total_procurement_cost"],
                "total_profit": test_result.metrics["total_profit"],
                "cvar99": test_result.metrics["cvar99"],
            }
        )
        parameter_rows.append(
            {
                "window_name": window.window_name,
                "best_score": window_model.best_score,
                "upper_best": list(window_model.upper_best),
                "lower_best": list(window_model.lower_best),
            }
        )
        validation_frames.append(pd.DataFrame([{"window_name": window.window_name, **val_result.metrics}]))
        backtest_frames.append(pd.DataFrame([{"window_name": window.window_name, **test_result.metrics}]))
        analysis_input = enrich_weekly_results(
            test_result.weekly_results,
            weekly_metadata=test_bundle["weekly_metadata"],
            weekly_features=test_bundle["weekly_features"],
            policy_state_trace=test_bundle["policy_state_trace"],
        )
        policy_metrics = build_policy_risk_adjusted_metrics(analysis_input, epsilon=epsilon)
        policy_metrics["window_name"] = window.window_name
        policy_metric_frames.append(policy_metrics)

    summary = summarize_rolling_results(window_records)
    rolling_excess_return_metrics = summarize_rolling_excess_return(
        pd.concat(policy_metric_frames, ignore_index=True) if policy_metric_frames else pd.DataFrame(),
        epsilon=epsilon,
    )
    summary.schedule.to_csv(context["output_paths"]["metrics"] / "rolling_window_schedule.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(context["output_paths"]["metrics"] / "rolling_parameter_snapshots.csv", index=False)
    pd.concat(validation_frames, ignore_index=True).to_csv(context["output_paths"]["metrics"] / "rolling_validation_metrics.csv", index=False)
    pd.concat(backtest_frames, ignore_index=True).to_csv(context["output_paths"]["metrics"] / "rolling_backtest_metrics.csv", index=False)
    rolling_excess_return_metrics.to_csv(context["output_paths"]["metrics"] / "rolling_excess_return_metrics.csv", index=False)
    summary.metrics.to_csv(context["output_paths"]["metrics"] / "backtest_metrics.csv", index=False)
    save_markdown(
        _build_backtest_summary({"aggregate": summary.aggregate}, rolling_excess_return_metrics),
        context["output_paths"]["reports"] / "backtest_summary.md",
    )
    save_markdown(
        "\n".join(
            [
                "# 滚动回测审计摘要",
                "",
                f"- 窗口数: {summary.aggregate['window_count']:.0f}",
                f"- 调度文件: {context['output_paths']['metrics'] / 'rolling_window_schedule.csv'}",
                f"- 参数快照: {context['output_paths']['metrics'] / 'rolling_parameter_snapshots.csv'}",
                f"- 验证指标: {context['output_paths']['metrics'] / 'rolling_validation_metrics.csv'}",
                f"- 回测指标: {context['output_paths']['metrics'] / 'rolling_backtest_metrics.csv'}",
                f"- 超额收益验证: {context['output_paths']['metrics'] / 'rolling_excess_return_metrics.csv'}",
                "",
            ]
        ),
        context["output_paths"]["reports"] / "rolling_audit_summary.md",
    )
    logger.info("回测模块执行完成。")
    return {
        "rolling_summary": summary,
        "validation_metrics": pd.concat(validation_frames, ignore_index=True),
        "backtest_metrics": pd.concat(backtest_frames, ignore_index=True),
        "rolling_excess_return_metrics": rolling_excess_return_metrics,
    }


def main() -> dict[str, Any]:
    context = prepare_project_context(Path.cwd(), logger_name="backtest")
    return run_backtest(context)


if __name__ == "__main__":
    main()
