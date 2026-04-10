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
    metrics_frame["cost_savings_vs_fixed_lock"] = fixed_cost - metrics_frame["total_procurement_cost"]
    metrics_frame["cost_savings_vs_rule_only"] = rule_cost - metrics_frame["total_procurement_cost"]
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


def run_hparam_search(context: dict[str, Any]) -> pd.DataFrame:
    config = context["config"]
    search_cfg = config["hparam_search"]
    rng = np.random.default_rng(int(config["seed"]))
    candidates = list(search_cfg["manual_grid"])

    for _ in range(int(search_cfg["random_search"]["n_trials"])):
        candidates.append(
            {
                "learning_rate": float(rng.uniform(*search_cfg["random_search"]["learning_rate_range"])),
                "n_steps": 128,
                "batch_size": 32,
                "ent_coef": float(rng.uniform(*search_cfg["random_search"]["ent_coef_range"])),
                "lambda_risk": float(rng.uniform(*search_cfg["random_search"]["lambda_risk_range"])),
                "lambda_he": float(rng.uniform(*search_cfg["random_search"]["lambda_he_range"])),
                "hedge_intensity_scale": float(rng.uniform(*search_cfg["random_search"]["hedge_intensity_scale_range"])),
            }
        )

    rows = []
    for trial_id, candidate in enumerate(candidates, start=1):
        trial_config = deepcopy(config)
        trial_config["learning_rate"] = float(candidate["learning_rate"])
        trial_config["n_steps"] = int(candidate["n_steps"])
        trial_config["batch_size"] = int(candidate["batch_size"])
        trial_config["ent_coef"] = float(candidate["ent_coef"])
        trial_config["cost"]["lambda_risk"] = float(candidate["lambda_risk"])
        trial_config["cost"]["lambda_he"] = float(candidate["lambda_he"])

        training = train_model(
            bundle=context["bundle"],
            train_weeks=context["train_sequence"],
            val_weeks=context["split"].val,
            config=trial_config,
            output_paths=context["output_paths"],
            run_name=f"hparam_trial_{trial_id}",
            total_timesteps_override=int(search_cfg["total_timesteps"]),
            save_plots=False,
        )
        validation = evaluate_policy(
            model=training["model"],
            bundle=context["bundle"],
            weeks=context["split"].val,
            config=trial_config,
            strategy_name=f"hparam_trial_{trial_id}_validation",
        )
        validation = simulate_strategy(
            bundle=context["bundle"],
            weeks=context["split"].val,
            action_source=validation["actions"],
            config=trial_config,
            strategy_name=f"hparam_trial_{trial_id}_validation",
            market_vol_scale=1.0,
            forecast_error_scale=float(candidate["hedge_intensity_scale"]),
        )
        metrics = validation["metrics"]
        rows.append(
            {
                "trial_id": trial_id,
                **candidate,
                "val_total_procurement_cost": metrics["total_procurement_cost"],
                "val_cvar": metrics["cvar"],
                "val_mean_reward": metrics["mean_reward"],
            }
        )

    frame = pd.DataFrame(rows).sort_values("val_total_procurement_cost").reset_index(drop=True)
    frame.to_csv(context["output_paths"]["metrics"] / "hparam_search_results.csv", index=False)
    summary_lines = ["# 超参数搜索总结", "", "## 最优试验", ""]
    for _, row in frame.head(5).iterrows():
        summary_lines.append(
            "- 试验 {trial_id}: lr={learning_rate:.6f}, n_steps={n_steps}, batch_size={batch_size}, "
            "ent_coef={ent_coef:.4f}, lambda_risk={lambda_risk:.4f}, lambda_he={lambda_he:.2f}, "
            "val_total_procurement_cost={val_total_procurement_cost:.2f}".format(**row.to_dict())
        )
    save_markdown("\n".join(summary_lines) + "\n", context["output_paths"]["reports"] / "hparam_search_summary.md")
    return frame


def _build_backtest_summary(context: dict[str, Any], results: dict[str, dict], metrics_frame: pd.DataFrame) -> str:
    ppo_metrics = results["ppo"]["metrics"]
    clip_stats = results["ppo"]["weekly_results"][
        ["bound_clip_count", "smooth_clip_count", "non_negative_clip_count"]
    ].sum()
    lines = [
        "# 回测摘要",
        "",
        f"- 使用模型版本: {_resolve_main_model(context['output_paths'])}",
        f"- 回测时间范围: {context['split'].test[0]} 至 {context['split'].test[-1]}",
        f"- warm-up 周: {', '.join([week.strftime('%Y-%m-%d') for week in context['split'].warmup]) if context['split'].warmup else '无'}",
        f"- 结算口径: {context['config']['reporting']['settlement_note']}",
        f"- 中长期价格口径: {context['config']['reporting']['lt_price_note']}",
        f"- PPO 总采购成本: {ppo_metrics['total_procurement_cost']:.2f}",
        f"- PPO 周度成本波动率: {ppo_metrics['weekly_cost_volatility']:.2f}",
        f"- PPO CVaR: {ppo_metrics['cvar']:.2f}",
        f"- PPO 套保误差: {ppo_metrics['hedge_error']:.4f}",
        f"- 约束裁剪统计: 上下限裁剪 {int(clip_stats['bound_clip_count'])} 次, 平滑裁剪 {int(clip_stats['smooth_clip_count'])} 次, 非负裁剪 {int(clip_stats['non_negative_clip_count'])} 次",
        "",
        "## 主策略与基准策略比较",
        "",
    ]
    for _, row in metrics_frame.iterrows():
        lines.append(
            f"- {row['strategy']}: 总采购成本={row['total_procurement_cost']:.2f}, "
            f"CVaR={row['cvar']:.2f}, 相对固定锁定策略节约={row['cost_savings_vs_fixed_lock']:.2f}, "
            f"相对规则对冲策略节约={row['cost_savings_vs_rule_only']:.2f}"
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
    pd.concat([payload["hourly_results"] for payload in results.values()], ignore_index=True).to_csv(
        context["output_paths"]["metrics"] / "backtest_hourly_rule_trace.csv",
        index=False,
    )
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
