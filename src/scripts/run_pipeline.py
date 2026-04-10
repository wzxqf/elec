from __future__ import annotations

import pandas as pd

from src.agents.train_ppo import evaluate_policy, train_model
from src.scripts.backtest import run_backtest
from src.scripts.common import prepare_project_context, split_to_dict
from src.utils.io import save_markdown


def _build_run_summary(context: dict, training: dict, validation: dict, backtest: dict) -> str:
    config = context["config"]
    data_quality = context["data_quality_report"]
    test_metrics = backtest["results"]["ppo"]["metrics"]
    validation_metrics = validation["metrics"]
    lines = [
        "# Run Summary",
        "",
        "## Data",
        "",
        f"- 数据文件: {context['csv_path']}",
        f"- 数据主样本区间: {config['sample_start']} 至 {config['sample_end']}",
        f"- 原始覆盖区间: {data_quality['start_before']} 至 {data_quality['end_before']}",
        f"- 重复时间戳个数: {data_quality['duplicate_timestamp_count']}",
        f"- 去重后缺失 15 分钟时点数: {data_quality['missing_timestamp_count_after_aggregation']}",
        "",
        "## Split",
        "",
    ]
    for key, values in split_to_dict(context["split"]).items():
        lines.append(f"- {key}: {', '.join(values) if values else '无'}")

    lines.extend(
        [
            "",
            "## PPO Hyperparameters",
            "",
            f"- learning_rate: {config['learning_rate']}",
            f"- n_steps: {config['n_steps']}",
            f"- batch_size: {config['batch_size']}",
            f"- n_epochs: {config['n_epochs']}",
            f"- gamma: {config['gamma']}",
            f"- ent_coef: {config['ent_coef']}",
            f"- seed: {config['seed']}",
            "",
            "## Model",
            "",
            f"- 最优模型目录: {training['best_model_path']}",
            f"- 最终模型路径: {training['model_path']}",
            f"- 训练 timesteps: {config['total_timesteps']}",
            f"- 训练设备: {training['device']}",
            "",
            "## Validation",
            "",
            f"- 累计采购成本: {validation_metrics['cumulative_procurement_cost']:.2f}",
            f"- CVaR(95%): {validation_metrics['cvar95']:.2f}",
            f"- 平均奖励: {validation_metrics['mean_reward']:.6f}",
            "",
            "## Backtest",
            "",
            f"- 累计采购成本: {test_metrics['cumulative_procurement_cost']:.2f}",
            f"- 成本波动率: {test_metrics['cost_volatility']:.2f}",
            f"- CVaR(95%): {test_metrics['cvar95']:.2f}",
            f"- 套保误差: {test_metrics['hedge_error']:.4f}",
            f"- 月均调整量: {test_metrics['avg_adjustment_mwh']:.4f}",
            "",
            "## Estimation Notes",
            "",
            "- 中长期价格: 使用上一自然月日前小时均价估算，若后续补充真实列可自动替换。",
            "- 实时结算: 使用日内价格作为代理结算口径。",
            "- 训练 episode: 仅在训练月内部进行 block bootstrap 重采样。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> dict:
    context = prepare_project_context("/Users/dk/py/elec")
    logger = context["logger"]
    logger.info("Start full pipeline.")

    training = train_model(
        bundle=context["bundle"],
        train_months=context["train_sequence"],
        val_months=context["split"].val,
        config=context["config"],
        output_paths=context["output_paths"],
    )
    validation = evaluate_policy(
        model=training["model"],
        bundle=context["bundle"],
        months=context["split"].val,
        config=context["config"],
        strategy_name="ppo_validation",
    )
    validation["monthly_results"].to_csv(context["output_paths"]["metrics"] / "validation_monthly_results.csv", index=False)
    pd.DataFrame([validation["metrics"]]).to_csv(
        context["output_paths"]["metrics"] / "validation_summary.csv",
        index=False,
    )

    backtest = run_backtest(context, model=training["model"])
    summary = _build_run_summary(context, training, validation, backtest)
    save_markdown(summary, context["output_paths"]["reports"] / "run_summary.md")
    logger.info("Pipeline complete.")
    return {
        "training": training,
        "validation": validation,
        "backtest": backtest,
    }


if __name__ == "__main__":
    main()
