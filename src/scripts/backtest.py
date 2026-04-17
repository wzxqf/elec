from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.ablations import build_ablation_summary_markdown, evaluate_ablation_variants
from src.analysis.benchmarks import build_benchmark_summary_markdown, evaluate_benchmark_strategies
from src.analysis.constraint_reporting import build_constraint_activation_report_markdown
from src.analysis.excess_return import build_policy_risk_adjusted_metrics, summarize_rolling_excess_return
from src.analysis.module1 import enrich_weekly_results
from src.analysis.report_contracts import build_summary_scope_lines, infer_date_range, positive_negative_counts
from src.analysis.robustness import build_robustness_summary_markdown, run_robustness_analysis
from src.agents.hybrid_pso import HybridPSOModel, train_hybrid_pso_model
from src.backtest.materialize import materialize_particle_pair
from src.backtest.rolling_pipeline import summarize_rolling_results
from src.scripts.common import prepare_project_context, subset_bundle_for_weeks
from src.utils.experiment_manifest import fallback_run_metadata, prepend_report_header, relativize_path
from src.utils.io import save_markdown
from src.utils.logger import configure_logging


def _build_backtest_summary(
    summary: dict[str, Any],
    backtest_metrics: pd.DataFrame,
    rolling_excess_return_metrics: pd.DataFrame | None = None,
    rolling_weekly_results: pd.DataFrame | None = None,
) -> str:
    rolling_excess_return_metrics = rolling_excess_return_metrics if rolling_excess_return_metrics is not None else pd.DataFrame()
    rolling_weekly_results = rolling_weekly_results if rolling_weekly_results is not None else pd.DataFrame()
    avg_sharpe = float(
        rolling_excess_return_metrics.get("window_policy_risk_adjusted_sharpe", pd.Series(dtype="float64")).mean() or 0.0
    )
    persistent_windows = int(
        rolling_excess_return_metrics.get("active_excess_return_persistent", pd.Series(dtype="bool")).astype(bool).sum()
    )
    positive_window_count, negative_window_count = positive_negative_counts(backtest_metrics.get("total_profit", pd.Series(dtype="float64")))
    date_range = infer_date_range(rolling_weekly_results)
    return "\n".join(
        [
            "# 回测摘要",
            "",
            *build_summary_scope_lines(
                sample_scope="rolling_backtest_windows",
                week_count=int(summary["aggregate"]["window_count"]),
                aggregation_method="window_level_mean_metrics",
                date_range=date_range,
            ),
            f"- 滚动窗口数: {summary['aggregate']['window_count']:.0f}",
            f"- 平均测试采购成本: {summary['aggregate']['mean_total_procurement_cost']:.2f}",
            f"- 平均测试经济收益: {summary['aggregate'].get('mean_total_profit', 0.0):.2f}",
            f"- 平均测试 CVaR99: {summary['aggregate']['mean_cvar99']:.2f}",
            f"- 窗口政策风险调整后夏普均值: {avg_sharpe:.4f}",
            f"- 持续跑赢窗口数: {persistent_windows}",
            f"- positive_window_count: {positive_window_count}",
            f"- negative_window_count: {negative_window_count}",
            f"- min_window_profit: {float(backtest_metrics.get('total_profit', pd.Series(dtype='float64')).min() if not backtest_metrics.empty else 0.0):.2f}",
            f"- max_window_profit: {float(backtest_metrics.get('total_profit', pd.Series(dtype='float64')).max() if not backtest_metrics.empty else 0.0):.2f}",
            "",
        ]
    )


def _aggregate_benchmark_metrics(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    aggregation = {
        "total_procurement_cost": "sum",
        "total_profit": "sum",
        "mean_profit": "mean",
        "weekly_profit_volatility": "mean",
        "cvar99": "mean",
        "max_drawdown": "mean",
    }
    aggregation = {column: rule for column, rule in aggregation.items() if column in combined.columns}
    result = combined.groupby("strategy_name", as_index=False).agg(aggregation)
    if "total_profit" in result.columns:
        baseline_profit = float(result.loc[result["strategy_name"] == "dynamic_lock_only", "total_profit"].iloc[0])
        result["profit_delta_vs_dynamic_lock_only"] = result["total_profit"] - baseline_profit
    return result.sort_values("strategy_name").reset_index(drop=True)


def _aggregate_ablation_metrics(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    aggregation = {
        "total_procurement_cost": "sum",
        "total_profit": "sum",
        "weekly_cost_volatility": "mean",
        "cvar99": "mean",
        "hedge_error": "mean",
        "avg_adjustment_mwh": "mean",
        "mean_reward": "mean",
        "max_drawdown": "mean",
        "feasible_domain_trigger_weeks": "sum",
    }
    aggregation = {column: rule for column, rule in aggregation.items() if column in combined.columns}
    result = combined.groupby("variant_name", as_index=False).agg(aggregation)
    return result.sort_values("variant_name").reset_index(drop=True)


def run_backtest(context: dict[str, Any], model=None) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="backtest")
    logger.info("开始执行 %s 回测模块。", context["config"]["version"])
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    window_records: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    backtest_frames: list[pd.DataFrame] = []
    policy_metric_frames: list[pd.DataFrame] = []
    weekly_result_frames: list[pd.DataFrame] = []
    hourly_result_frames: list[pd.DataFrame] = []
    settlement_result_frames: list[pd.DataFrame] = []
    ablation_frames: list[pd.DataFrame] = []
    epsilon = float(context["config"].get("analysis_v035", {}).get("sharpe_epsilon", 1.0e-6))

    for window in context["rolling_plan"]:
        train_bundle = subset_bundle_for_weeks(context["bundle"], window.train_weeks)
        val_bundle = subset_bundle_for_weeks(context["bundle"], window.val_weeks)
        test_bundle = subset_bundle_for_weeks(context["bundle"], window.test_weeks)
        training = train_hybrid_pso_model(
            train_bundle["tensor_bundle"],
            context["config"],
            compiled_layout=train_bundle.get("compiled_parameter_layout"),
        )
        window_model: HybridPSOModel = training.model
        val_result = materialize_particle_pair(
            tensor_bundle=val_bundle["tensor_bundle"],
            upper_particle=window_model.upper_best,
            lower_particle=window_model.lower_best,
            strategy_name=f"{window.window_name}_validation",
            config=context["config"],
            compiled_layout=val_bundle.get("compiled_parameter_layout"),
        )
        test_result = materialize_particle_pair(
            tensor_bundle=test_bundle["tensor_bundle"],
            upper_particle=window_model.upper_best,
            lower_particle=window_model.lower_best,
            strategy_name=f"{window.window_name}_test",
            config=context["config"],
            compiled_layout=test_bundle.get("compiled_parameter_layout"),
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
        weekly_result_frames.append(test_result.weekly_results.assign(window_name=window.window_name))
        hourly_result_frames.append(test_result.hourly_results.assign(window_name=window.window_name))
        settlement_result_frames.append(test_result.settlement_results.assign(window_name=window.window_name))
        policy_metrics = build_policy_risk_adjusted_metrics(analysis_input, epsilon=epsilon)
        policy_metrics["window_name"] = window.window_name
        policy_metric_frames.append(policy_metrics)
        ablation_metrics = evaluate_ablation_variants(test_bundle, window_model, context["config"])
        ablation_metrics["window_name"] = window.window_name
        ablation_frames.append(ablation_metrics)

    summary = summarize_rolling_results(window_records)
    rolling_backtest_metrics = pd.concat(backtest_frames, ignore_index=True) if backtest_frames else pd.DataFrame()
    rolling_weekly_results = pd.concat(weekly_result_frames, ignore_index=True) if weekly_result_frames else pd.DataFrame()
    rolling_hourly_results = pd.concat(hourly_result_frames, ignore_index=True) if hourly_result_frames else pd.DataFrame()
    rolling_settlement_results = pd.concat(settlement_result_frames, ignore_index=True) if settlement_result_frames else pd.DataFrame()
    holdout_weeks = list(context["split"].val) + list(context["split"].test)
    holdout_bundle = subset_bundle_for_weeks(context["bundle"], holdout_weeks)
    benchmark_metrics = evaluate_benchmark_strategies(holdout_bundle, context["config"])
    ablation_metrics = _aggregate_ablation_metrics(ablation_frames)
    robustness_metrics = run_robustness_analysis(weekly_results=rolling_weekly_results, config=context["config"])
    rolling_excess_return_metrics = summarize_rolling_excess_return(
        pd.concat(policy_metric_frames, ignore_index=True) if policy_metric_frames else pd.DataFrame(),
        epsilon=epsilon,
    )
    summary.schedule.to_csv(context["output_paths"]["metrics"] / "rolling_window_schedule.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(context["output_paths"]["metrics"] / "rolling_parameter_snapshots.csv", index=False)
    pd.concat(validation_frames, ignore_index=True).to_csv(context["output_paths"]["metrics"] / "rolling_validation_metrics.csv", index=False)
    rolling_backtest_metrics.to_csv(context["output_paths"]["metrics"] / "rolling_backtest_metrics.csv", index=False)
    rolling_weekly_results.to_csv(context["output_paths"]["metrics"] / "rolling_weekly_results.csv", index=False)
    rolling_hourly_results.to_csv(context["output_paths"]["metrics"] / "rolling_hourly_results.csv", index=False)
    rolling_settlement_results.to_csv(context["output_paths"]["metrics"] / "rolling_settlement_results.csv", index=False)
    rolling_excess_return_metrics.to_csv(context["output_paths"]["metrics"] / "rolling_excess_return_metrics.csv", index=False)
    summary.metrics.to_csv(context["output_paths"]["metrics"] / "backtest_metrics.csv", index=False)
    benchmark_metrics.to_csv(context["output_paths"]["metrics"] / "benchmark_comparison.csv", index=False)
    ablation_metrics.to_csv(context["output_paths"]["metrics"] / "ablation_metrics.csv", index=False)
    robustness_metrics.to_csv(context["output_paths"]["metrics"] / "robustness_metrics.csv", index=False)
    run_metadata = context.get("run_metadata", fallback_run_metadata(context["config"]))
    save_markdown(
        prepend_report_header(
            _build_backtest_summary(
                {"aggregate": summary.aggregate},
                rolling_backtest_metrics,
                rolling_excess_return_metrics,
                rolling_weekly_results,
            ),
            run_metadata,
        ),
        context["output_paths"]["reports"] / "backtest_summary.md",
    )
    holdout_date_range = infer_date_range(holdout_bundle["weekly_metadata"])
    save_markdown(
        prepend_report_header(
            build_benchmark_summary_markdown(
                benchmark_metrics,
                sample_scope="holdout_validation_test",
                week_count=len(holdout_weeks),
                aggregation_method="holdout_week_sum_and_mean",
                date_range=holdout_date_range,
            ),
            run_metadata,
        ),
        context["output_paths"]["reports"] / "benchmark_summary.md",
    )
    save_markdown(
        prepend_report_header(
            build_ablation_summary_markdown(
                ablation_metrics,
                sample_scope="rolling_backtest_windows",
                week_count=int(summary.aggregate["window_count"]),
                aggregation_method="aggregate_over_rolling_windows",
                date_range=infer_date_range(rolling_weekly_results),
            ),
            run_metadata,
        ),
        context["output_paths"]["reports"] / "ablation_summary.md",
    )
    save_markdown(
        prepend_report_header(
            build_robustness_summary_markdown(
                robustness_metrics,
                sample_scope="rolling_backtest_windows",
                week_count=int(summary.aggregate["window_count"]),
                aggregation_method="scenario_sweep_over_rolling_results",
                date_range=infer_date_range(rolling_weekly_results),
            ),
            run_metadata,
        ),
        context["output_paths"]["reports"] / "robustness_summary.md",
    )
    save_markdown(
        prepend_report_header(
            build_constraint_activation_report_markdown(
                rolling_weekly_results,
                sample_scope="rolling_backtest_windows",
                week_count=int(summary.aggregate["window_count"]),
                aggregation_method="window_level_projection_audit",
                date_range=infer_date_range(rolling_weekly_results),
            ),
            run_metadata,
        ),
        context["output_paths"]["reports"] / "constraint_activation_report.md",
    )
    save_markdown(
        prepend_report_header(
            "\n".join(
                [
                    "# 滚动回测审计摘要",
                    "",
                    f"- 窗口数: {summary.aggregate['window_count']:.0f}",
                    f"- 调度文件: {relativize_path(context['output_paths']['metrics'] / 'rolling_window_schedule.csv', output_root)}",
                    f"- 参数快照: {relativize_path(context['output_paths']['metrics'] / 'rolling_parameter_snapshots.csv', output_root)}",
                    f"- 验证指标: {relativize_path(context['output_paths']['metrics'] / 'rolling_validation_metrics.csv', output_root)}",
                    f"- 回测指标: {relativize_path(context['output_paths']['metrics'] / 'rolling_backtest_metrics.csv', output_root)}",
                    f"- 周度回测明细: {relativize_path(context['output_paths']['metrics'] / 'rolling_weekly_results.csv', output_root)}",
                    f"- 超额收益验证: {relativize_path(context['output_paths']['metrics'] / 'rolling_excess_return_metrics.csv', output_root)}",
                    f"- 基准比较: {relativize_path(context['output_paths']['metrics'] / 'benchmark_comparison.csv', output_root)}",
                    f"- 消融结果: {relativize_path(context['output_paths']['metrics'] / 'ablation_metrics.csv', output_root)}",
                    f"- 稳健性结果: {relativize_path(context['output_paths']['metrics'] / 'robustness_metrics.csv', output_root)}",
                    f"- 约束激活报告: {relativize_path(context['output_paths']['reports'] / 'constraint_activation_report.md', output_root)}",
                    "",
                ]
            ),
            run_metadata,
        ),
        context["output_paths"]["reports"] / "rolling_audit_summary.md",
    )
    logger.info("回测模块执行完成。")
    return {
        "rolling_summary": summary,
        "validation_metrics": pd.concat(validation_frames, ignore_index=True),
        "backtest_metrics": rolling_backtest_metrics,
        "rolling_weekly_results": rolling_weekly_results,
        "rolling_hourly_results": rolling_hourly_results,
        "rolling_settlement_results": rolling_settlement_results,
        "rolling_excess_return_metrics": rolling_excess_return_metrics,
        "benchmark_metrics": benchmark_metrics,
        "ablation_metrics": ablation_metrics,
        "robustness_metrics": robustness_metrics,
    }


def main() -> dict[str, Any]:
    context = prepare_project_context(Path.cwd(), logger_name="backtest")
    return run_backtest(context)


if __name__ == "__main__":
    main()
