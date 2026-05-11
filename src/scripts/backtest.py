from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.ablations import evaluate_ablation_variants
from src.analysis.benchmarks import evaluate_benchmark_strategies
from src.analysis.excess_return import build_policy_risk_adjusted_metrics, summarize_rolling_excess_return
from src.analysis.module1 import enrich_weekly_results
from src.analysis.robustness import aggregate_robustness_scenario_rows, run_robustness_analysis
from src.agents.hybrid_pso import HybridPSOModel, train_hybrid_pso_model
from src.backtest.materialize import materialize_particle_pair
from src.backtest.rolling_pipeline import summarize_rolling_results
from src.scripts.common import prepare_project_context, subset_bundle_for_weeks
from src.utils.logger import configure_logging


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
    window_records: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    backtest_frames: list[pd.DataFrame] = []
    policy_metric_frames: list[pd.DataFrame] = []
    weekly_result_frames: list[pd.DataFrame] = []
    hourly_result_frames: list[pd.DataFrame] = []
    settlement_result_frames: list[pd.DataFrame] = []
    ablation_frames: list[pd.DataFrame] = []
    robustness_frames: list[pd.DataFrame] = []
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
        robustness_window_metrics = run_robustness_analysis(
            weekly_results=test_result.weekly_results,
            config=context["config"],
            tensor_bundle=test_bundle["tensor_bundle"],
            upper_particle=window_model.upper_best,
            lower_particle=window_model.lower_best,
            compiled_layout=test_bundle.get("compiled_parameter_layout"),
            window_name=window.window_name,
        )
        if not robustness_window_metrics.empty:
            robustness_frames.append(robustness_window_metrics)

    summary = summarize_rolling_results(window_records)
    rolling_backtest_metrics = pd.concat(backtest_frames, ignore_index=True) if backtest_frames else pd.DataFrame()
    rolling_weekly_results = pd.concat(weekly_result_frames, ignore_index=True) if weekly_result_frames else pd.DataFrame()
    rolling_hourly_results = pd.concat(hourly_result_frames, ignore_index=True) if hourly_result_frames else pd.DataFrame()
    rolling_settlement_results = pd.concat(settlement_result_frames, ignore_index=True) if settlement_result_frames else pd.DataFrame()
    holdout_weeks = list(context["split"].val) + list(context["split"].test)
    holdout_bundle = subset_bundle_for_weeks(context["bundle"], holdout_weeks)
    benchmark_metrics = evaluate_benchmark_strategies(holdout_bundle, context["config"])
    ablation_metrics = _aggregate_ablation_metrics(ablation_frames)
    robustness_metrics = aggregate_robustness_scenario_rows(
        pd.concat(robustness_frames, ignore_index=True) if robustness_frames else pd.DataFrame()
    )
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
