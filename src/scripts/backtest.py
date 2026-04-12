from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.train_ppo import evaluate_policy, load_model, train_model
from src.analysis.robustness import run_robustness_analysis
from src.analysis.sensitivity import run_sensitivity_analysis
from src.backtest.benchmarks import build_benchmark_actions
from src.backtest.simulator import simulate_strategy
from src.data.scenario_generator import build_bootstrap_sequence
from src.scripts.common import prepare_project_context
from src.utils.io import save_markdown
from src.utils.logger import configure_logging
from src.utils.plotting import save_bar_plot, save_line_plot, save_multi_line_plot


def _resolve_main_model(output_paths: dict[str, Path]) -> Path | None:
    for candidate in [output_paths["models"] / "ppo_best.zip", output_paths["models"] / "ppo_latest.zip"]:
        if candidate.exists():
            return candidate
    return None


def _ensure_main_model(context: dict[str, Any]):
    model_path = _resolve_main_model(context["output_paths"])
    if model_path is not None:
        return load_model(model_path)
    training = train_model(
        bundle=context["bundle"],
        train_weeks=context["train_sequence"],
        val_weeks=context["split"].val,
        config=context["config"],
        output_paths=context["output_paths"],
        run_name="ppo",
    )
    return training["model"]


def _benchmark_comparison_frame(results: dict[str, dict]) -> pd.DataFrame:
    metrics_frame = pd.DataFrame([payload["metrics"] for payload in results.values()])
    fixed_cost = float(metrics_frame.loc[metrics_frame["strategy"] == "fixed_lock", "total_procurement_cost"].iloc[0])
    rule_cost = float(metrics_frame.loc[metrics_frame["strategy"] == "rule_only", "total_procurement_cost"].iloc[0])
    dynamic_cost = float(metrics_frame.loc[metrics_frame["strategy"] == "dynamic_lock_only", "total_procurement_cost"].iloc[0])
    metrics_frame["cost_savings_vs_fixed_lock"] = fixed_cost - metrics_frame["total_procurement_cost"]
    metrics_frame["cost_savings_vs_rule_only"] = rule_cost - metrics_frame["total_procurement_cost"]
    metrics_frame["cost_gap_vs_dynamic_lock_only"] = metrics_frame["total_procurement_cost"] - dynamic_cost
    return metrics_frame.sort_values("total_procurement_cost").reset_index(drop=True)


def _save_core_figures(results: dict[str, dict], output_paths: dict[str, Path]) -> None:
    weekly = pd.concat([payload["weekly_results"] for payload in results.values()], ignore_index=True)

    weekly_cost = weekly.pivot(index="week_start", columns="strategy", values="procurement_cost_w").sort_index()
    save_multi_line_plot(
        weekly_cost,
        output_paths["figures"] / "cost_curve.png",
        title="周度采购成本曲线",
        xlabel="week_start",
        ylabel="procurement_cost_w",
    )

    ppo_reward = results["ppo"]["weekly_results"].sort_values("week_start")
    save_line_plot(
        ppo_reward["week_start"].astype(str),
        ppo_reward["reward"],
        output_paths["figures"] / "weekly_reward_curve.png",
        title="周度奖励曲线",
        xlabel="week_start",
        ylabel="reward",
    )

    benchmark_compare = _benchmark_comparison_frame(results).set_index("strategy")[["total_procurement_cost", "cvar"]]
    save_bar_plot(
        benchmark_compare,
        output_paths["figures"] / "benchmark_compare.png",
        title="主策略与基准策略比较",
        xlabel="strategy",
        ylabel="value",
    )


def _build_sensitivity_report(frame: pd.DataFrame) -> str:
    lines = ["# 敏感性分析报告", ""]
    for factor in frame["factor"].unique():
        lines.extend([f"## {factor}", ""])
        subset = frame.loc[frame["factor"] == factor].sort_values("value")
        for _, row in subset.iterrows():
            lines.append(
                f"- 取值 {row['value']}: 总采购成本={row['total_procurement_cost']:.2f}, "
                f"CVaR={row['cvar']:.2f}, 套保误差={row['hedge_error']:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def _build_robustness_report(frame: pd.DataFrame) -> str:
    lines = ["# 鲁棒性分析报告", ""]
    for experiment in frame["experiment"].unique():
        lines.extend([f"## {experiment}", ""])
        subset = frame.loc[frame["experiment"] == experiment]
        for _, row in subset.iterrows():
            lines.append(
                f"- 取值 {row['value']}: 总采购成本={row['total_procurement_cost']:.2f}, "
                f"CVaR={row['cvar']:.2f}, 套保误差={row['hedge_error']:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def _sample_stage1_candidates(search_cfg: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    ranges = search_cfg["stage1_ranges"]
    candidates = []
    for trial_id in range(1, int(search_cfg["stage1_trials"]) + 1):
        candidates.append(
            {
                "candidate_id": f"stage1_{trial_id:02d}",
                "learning_rate": float(rng.uniform(*ranges["learning_rate"])),
                "n_steps": int(rng.choice(ranges["n_steps_choices"])),
                "batch_size": int(rng.choice(ranges["batch_size_choices"])),
                "gamma": float(rng.uniform(*ranges["gamma"])),
                "gae_lambda": float(rng.uniform(*ranges["gae_lambda"])),
                "clip_range": float(rng.uniform(*ranges["clip_range"])),
                "ent_coef": float(rng.uniform(*ranges["ent_coef"])),
                "lambda_hedge_error": float(rng.uniform(*ranges["lambda_hedge_error"])),
                "lambda_hourly_smooth": float(rng.uniform(*ranges["lambda_hourly_smooth"])),
                "gamma_max": float(rng.uniform(*ranges["gamma_max"])),
                "band_base_multiplier": float(rng.uniform(*ranges["band_base_multiplier"])),
                "price_spike_shrink": float(rng.uniform(*ranges["price_spike_shrink"])),
            }
        )
    return candidates


def _apply_candidate_config(base_config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    config_variant = deepcopy(base_config)
    for key in ["learning_rate", "n_steps", "batch_size", "gamma", "gae_lambda", "clip_range", "ent_coef"]:
        config_variant[key] = candidate[key]
        config_variant["training"][key] = candidate[key]
    config_variant["reward"]["lambda_hedge_error"] = candidate["lambda_hedge_error"]
    config_variant["rules"]["gamma_max"] = candidate["gamma_max"]
    config_variant["rules"]["band_base_multiplier"] = candidate["band_base_multiplier"]
    config_variant["rules"]["price_spike_shrink"] = candidate["price_spike_shrink"]
    config_variant["reward"]["lambda_hourly_smooth"] = candidate["lambda_hourly_smooth"]
    return config_variant


def _evaluate_candidate_over_windows(
    context: dict[str, Any],
    candidate: dict[str, Any],
    stage_name: str,
    timesteps: int,
) -> pd.DataFrame:
    if not context["rolling_validation_windows"]:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    config_variant = _apply_candidate_config(context["config"], candidate)
    for window_index, window in enumerate(context["rolling_validation_windows"], start=1):
        window_sequence = build_bootstrap_sequence(
            train_weeks=window.train,
            sequence_length=int(config_variant["scenario"]["train_sequence_length"]),
            block_size=int(config_variant["scenario"]["block_size"]),
            seed=int(config_variant["scenario"]["bootstrap_seed"]) + window_index,
        )
        training = train_model(
            bundle=context["bundle"],
            train_weeks=window_sequence,
            val_weeks=window.val,
            config=config_variant,
            output_paths=context["output_paths"],
            run_name=f"{candidate['candidate_id']}_{window_index}",
            total_timesteps_override=timesteps,
            save_plots=False,
            persist_artifacts=False,
        )
        validation = evaluate_policy(
            model=training["model"],
            bundle=context["bundle"],
            weeks=window.val,
            config=config_variant,
            strategy_name=f"{candidate['candidate_id']}_{window.name}",
        )
        benchmark_actions = build_benchmark_actions(window.val, context["bundle"]["weekly_features"], config_variant)
        dynamic_baseline = simulate_strategy(
            bundle=context["bundle"],
            weeks=window.val,
            action_source=benchmark_actions["dynamic_lock_only"],
            config=config_variant,
            strategy_name="dynamic_lock_only_rolling",
        )
        rows.append(
            {
                "stage": stage_name,
                "candidate_id": candidate["candidate_id"],
                "window_name": window.name,
                "train_end_week": window.train[-1],
                "val_start_week": window.val[0],
                "val_end_week": window.val[-1],
                "val_total_procurement_cost": validation["metrics"]["total_procurement_cost"],
                "val_cvar": validation["metrics"]["cvar"],
                "val_mean_reward": validation["metrics"]["mean_reward"],
                "baseline_total_procurement_cost": dynamic_baseline["metrics"]["total_procurement_cost"],
                "baseline_cvar": dynamic_baseline["metrics"]["cvar"],
                "cost_gap_vs_dynamic_lock_only": (
                    validation["metrics"]["total_procurement_cost"] - dynamic_baseline["metrics"]["total_procurement_cost"]
                ),
                "cvar_gap_vs_dynamic_lock_only": validation["metrics"]["cvar"] - dynamic_baseline["metrics"]["cvar"],
                **candidate,
            }
        )
    return pd.DataFrame(rows)


def _summarize_rolling_validation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    summary = (
        frame.groupby(["stage", "candidate_id"], as_index=False)
        .agg(
            avg_val_total_procurement_cost=("val_total_procurement_cost", "mean"),
            avg_val_cvar=("val_cvar", "mean"),
            avg_val_mean_reward=("val_mean_reward", "mean"),
            avg_cost_gap_vs_dynamic_lock_only=("cost_gap_vs_dynamic_lock_only", "mean"),
            avg_cvar_gap_vs_dynamic_lock_only=("cvar_gap_vs_dynamic_lock_only", "mean"),
        )
    )
    summary["ranking_score"] = summary["avg_val_total_procurement_cost"] + summary["avg_cvar_gap_vs_dynamic_lock_only"].clip(lower=0.0) * 0.05
    return summary.sort_values(["ranking_score", "avg_val_total_procurement_cost"]).reset_index(drop=True)


def _build_rolling_validation_summary(detail_frame: pd.DataFrame, summary_frame: pd.DataFrame) -> str:
    lines = ["# 滚动验证摘要", ""]
    if detail_frame.empty or summary_frame.empty:
        lines.append("- 当前未启用滚动验证。")
        return "\n".join(lines)

    for stage in summary_frame["stage"].unique():
        lines.extend([f"## {stage}", ""])
        stage_summary = summary_frame.loc[summary_frame["stage"] == stage].head(6)
        for _, row in stage_summary.iterrows():
            lines.append(
                f"- {row['candidate_id']}: 平均验证成本={row['avg_val_total_procurement_cost']:.2f}, "
                f"平均成本差(dynamic_lock_only)={row['avg_cost_gap_vs_dynamic_lock_only']:.2f}, "
                f"平均CVaR差={row['avg_cvar_gap_vs_dynamic_lock_only']:.2f}"
            )
        lines.append("")

    lines.extend(["## 分窗口明细样本", ""])
    for _, row in detail_frame.head(12).iterrows():
        lines.append(
            f"- {row['stage']} / {row['candidate_id']} / {row['window_name']}: "
            f"验证成本={row['val_total_procurement_cost']:.2f}, "
            f"相对强基准成本差={row['cost_gap_vs_dynamic_lock_only']:.2f}, "
            f"CVaR差={row['cvar_gap_vs_dynamic_lock_only']:.2f}"
        )
    lines.append("")
    return "\n".join(lines)


def run_hparam_search(context: dict[str, Any]) -> pd.DataFrame:
    config = context["config"]
    search_cfg = config["search"]
    if not search_cfg.get("enabled", False):
        empty = pd.DataFrame()
        empty.to_csv(context["output_paths"]["metrics"] / "hparam_search_results.csv", index=False)
        empty.to_csv(context["output_paths"]["metrics"] / "rolling_validation_metrics.csv", index=False)
        save_markdown("# 滚动验证摘要\n\n- 当前配置未启用搜索与滚动验证。\n", context["output_paths"]["reports"] / "rolling_validation_summary.md")
        return empty

    rng = np.random.default_rng(int(search_cfg.get("random_seed", config["seed"])))
    stage1_candidates = _sample_stage1_candidates(search_cfg, rng)
    stage1_detail = pd.concat(
        [
            _evaluate_candidate_over_windows(
                context=context,
                candidate=candidate,
                stage_name="粗搜索",
                timesteps=int(search_cfg["stage1_timesteps"]),
            )
            for candidate in stage1_candidates
        ],
        ignore_index=True,
    )
    stage1_summary = _summarize_rolling_validation(stage1_detail)
    top_candidates = stage1_summary.head(int(search_cfg["stage2_topk"]))["candidate_id"].tolist()
    stage2_candidates = [candidate for candidate in stage1_candidates if candidate["candidate_id"] in top_candidates]
    stage2_detail = pd.concat(
        [
            _evaluate_candidate_over_windows(
                context=context,
                candidate={**candidate, "candidate_id": candidate["candidate_id"].replace("stage1", "stage2")},
                stage_name="精搜索",
                timesteps=int(search_cfg["stage2_timesteps"]),
            )
            for candidate in stage2_candidates
        ],
        ignore_index=True,
    )
    detail_frame = pd.concat([stage1_detail, stage2_detail], ignore_index=True)
    summary_frame = _summarize_rolling_validation(detail_frame)

    detail_frame.to_csv(context["output_paths"]["metrics"] / "rolling_validation_metrics.csv", index=False)
    summary_frame.to_csv(context["output_paths"]["metrics"] / "hparam_search_results.csv", index=False)
    save_markdown(
        _build_rolling_validation_summary(detail_frame, summary_frame),
        context["output_paths"]["reports"] / "rolling_validation_summary.md",
    )
    save_markdown(
        _build_rolling_validation_summary(detail_frame, summary_frame).replace("滚动验证摘要", "超参数搜索总结"),
        context["output_paths"]["reports"] / "hparam_search_summary.md",
    )
    return summary_frame


def _build_backtest_summary(context: dict[str, Any], results: dict[str, dict], metrics_frame: pd.DataFrame) -> str:
    ppo_metrics = results["ppo"]["metrics"]
    clip_stats = results["ppo"]["weekly_results"][
        ["bound_clip_count", "smooth_clip_count", "soft_clip_count", "non_negative_clip_count"]
    ].sum()
    lines = [
        "# 回测摘要",
        "",
        f"- 使用模型版本: {_resolve_main_model(context['output_paths'])}",
        f"- 回测时间范围: {context['split'].test[0]} 至 {context['split'].test[-1]}",
        f"- 结算口径: {context['config']['reporting']['settlement_note']}",
        f"- 中长期价格口径: {context['config']['reporting']['lt_price_note']}",
        f"- 奖励强基准: {context['config']['reward']['strong_baseline']}",
        f"- 根参数文件: {context['config']['config_path']}",
        f"- PPO 总采购成本: {ppo_metrics['total_procurement_cost']:.2f}",
        f"- PPO 周度成本波动率: {ppo_metrics['weekly_cost_volatility']:.2f}",
        f"- PPO CVaR: {ppo_metrics['cvar']:.2f}",
        f"- PPO 套保误差: {ppo_metrics['hedge_error']:.4f}",
        f"- 强边界裁剪次数: {int(clip_stats['bound_clip_count'])}",
        f"- 平滑压缩次数: {int(clip_stats['smooth_clip_count'])}",
        f"- soft_clip 触发次数: {int(clip_stats['soft_clip_count'])}",
        f"- 非负裁剪次数: {int(clip_stats['non_negative_clip_count'])}",
        "",
        "## 主策略与基准策略比较",
        "",
    ]
    for _, row in metrics_frame.iterrows():
        lines.append(
            f"- {row['strategy']}: 总采购成本={row['total_procurement_cost']:.2f}, "
            f"CVaR={row['cvar']:.2f}, 相对 dynamic_lock_only 成本差={row['cost_gap_vs_dynamic_lock_only']:.2f}"
        )
    lines.append("")
    return "\n".join(lines)


def run_backtest(context: dict[str, Any], model=None) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="backtest")
    logger.info("开始执行回测模块。")
    model = model or _ensure_main_model(context)

    ppo_result = evaluate_policy(
        model=model,
        bundle=context["bundle"],
        weeks=context["split"].test,
        config=context["config"],
        strategy_name="ppo",
    )
    benchmark_actions = build_benchmark_actions(
        weeks=context["split"].test,
        weekly_features=context["bundle"]["weekly_features"],
        config=context["config"],
    )
    results = {"ppo": ppo_result}
    for strategy, actions in benchmark_actions.items():
        results[strategy] = simulate_strategy(
            bundle=context["bundle"],
            weeks=context["split"].test,
            action_source=actions,
            config=context["config"],
            strategy_name=strategy,
        )

    metrics_frame = _benchmark_comparison_frame(results)
    metrics_frame.to_csv(context["output_paths"]["metrics"] / "backtest_metrics.csv", index=False)
    pd.concat([payload["weekly_results"] for payload in results.values()], ignore_index=True).to_csv(
        context["output_paths"]["metrics"] / "backtest_weekly_results.csv",
        index=False,
    )
    results["ppo"]["weekly_results"].to_csv(context["output_paths"]["metrics"] / "weekly_results.csv", index=False)
    pd.concat([payload["hourly_results"] for payload in results.values()], ignore_index=True).to_csv(
        context["output_paths"]["metrics"] / "backtest_hourly_rule_trace.csv",
        index=False,
    )
    results["ppo"]["hourly_results"].to_csv(context["output_paths"]["metrics"] / "hourly_rule_trace.csv", index=False)
    pd.concat([payload["settlement_results"] for payload in results.values()], ignore_index=True).to_csv(
        context["output_paths"]["metrics"] / "backtest_settlement_trace.csv",
        index=False,
    )

    _save_core_figures(results, context["output_paths"])

    sensitivity = run_sensitivity_analysis(context, ppo_result["actions"])
    sensitivity.to_csv(context["output_paths"]["metrics"] / "sensitivity_metrics.csv", index=False)
    save_markdown(_build_sensitivity_report(sensitivity), context["output_paths"]["reports"] / "sensitivity_report.md")

    robustness = run_robustness_analysis(context, model, ppo_result["actions"])
    robustness.to_csv(context["output_paths"]["metrics"] / "robustness_metrics.csv", index=False)
    save_markdown(_build_robustness_report(robustness), context["output_paths"]["reports"] / "robustness_report.md")

    hparam_search = run_hparam_search(context)
    save_markdown(
        _build_backtest_summary(context, results, metrics_frame),
        context["output_paths"]["reports"] / "backtest_summary.md",
    )
    logger.info("回测模块执行完成。")
    return {
        "results": results,
        "metrics_frame": metrics_frame,
        "sensitivity": sensitivity,
        "robustness": robustness,
        "hparam_search": hparam_search,
    }


def main() -> dict[str, Any]:
    context = prepare_project_context("/Users/dk/py/elec", logger_name="backtest")
    return run_backtest(context)


if __name__ == "__main__":
    main()
