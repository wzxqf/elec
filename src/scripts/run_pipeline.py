from __future__ import annotations

from pathlib import Path

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
        "# 运行总结",
        "",
        "## 数据情况",
        "",
        f"- 数据文件: {context['csv_path']}",
        f"- 数据主样本区间: {config['sample_start']} 至 {config['sample_end']}",
        f"- 原始覆盖区间: {data_quality['start_before']} 至 {data_quality['end_before']}",
        f"- 重复时间戳个数: {data_quality['duplicate_timestamp_count']}",
        f"- 去重后缺失 15 分钟时点数: {data_quality['missing_timestamp_count_after_aggregation']}",
        "",
        "## 样本划分",
        "",
    ]
    split_labels = {
        "warmup": "预热期",
        "train": "训练集",
        "val": "验证集",
        "test": "回测集",
    }
    for key, values in split_to_dict(context["split"]).items():
        lines.append(f"- {split_labels[key]}: {', '.join(values) if values else '无'}")

    lines.extend(
        [
            "",
            "## PPO 超参数",
            "",
            f"- learning_rate: {config['learning_rate']}",
            f"- n_steps: {config['n_steps']}",
            f"- batch_size: {config['batch_size']}",
            f"- n_epochs: {config['n_epochs']}",
            f"- gamma: {config['gamma']}",
            f"- ent_coef: {config['ent_coef']}",
            f"- seed: {config['seed']}",
            "",
            "## 模型信息",
            "",
            f"- 最优模型目录: {training['best_model_path']}",
            f"- 最终模型路径: {training['model_path']}",
            f"- 训练 timesteps: {config['total_timesteps']}",
            f"- 训练设备: {training['device']}",
            "",
            "## 验证结果",
            "",
            f"- 累计采购成本: {validation_metrics['cumulative_procurement_cost']:.2f}",
            f"- CVaR(95%): {validation_metrics['cvar95']:.2f}",
            f"- 平均奖励: {validation_metrics['mean_reward']:.6f}",
            "",
            "## 回测结果",
            "",
            f"- 累计采购成本: {test_metrics['cumulative_procurement_cost']:.2f}",
            f"- 成本波动率: {test_metrics['cost_volatility']:.2f}",
            f"- CVaR(95%): {test_metrics['cvar95']:.2f}",
            f"- 套保误差: {test_metrics['hedge_error']:.4f}",
            f"- 月均调整量: {test_metrics['avg_adjustment_mwh']:.4f}",
            "",
            "## 估算口径说明",
            "",
            "- 中长期价格: 使用上一自然月日前小时均价估算，若后续补充真实列可自动替换。",
            "- 实时结算: 使用日内价格作为代理结算口径。",
            "- 训练 episode: 仅在训练月内部进行 block bootstrap 重采样。",
            "",
        ]
    )
    return "\n".join(lines)


def _frame_to_markdown(frame: pd.DataFrame, index: bool = False) -> str:
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_float_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: round(float(value), 6))
    if index:
        printable = printable.reset_index()
    printable = printable.fillna("")
    headers = [str(column) for column in printable.columns]
    rows = [[str(value) for value in row] for row in printable.astype(object).values.tolist()]
    separator = ["---"] * len(headers)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    table_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table_lines)


def _build_detailed_run_report(context: dict, training: dict, validation: dict, backtest: dict) -> str:
    config = context["config"]
    split = split_to_dict(context["split"])
    bundle = context["bundle"]
    monthly_features = bundle["monthly_features"]
    monthly_metadata = bundle["monthly_metadata"]
    benchmark_frame = backtest["metrics_frame"].copy()
    validation_monthly = validation["monthly_results"].copy().sort_values("month").reset_index(drop=True)
    ppo_monthly = backtest["results"]["ppo"]["monthly_results"].copy().sort_values("month").reset_index(drop=True)
    ppo_actions = ppo_monthly[["month", "lock_ratio", "hedge_intensity", "q_lt_target_mwh", "lt_price_m"]].copy()

    model_profile = pd.DataFrame(
        [
            {
                "算法": "PPO",
                "策略网络": config["policy"],
                "训练设备": training["device"],
                "总训练步数": int(config["total_timesteps"]),
                "n_steps": int(config["n_steps"]),
                "batch_size": int(config["batch_size"]),
                "n_epochs": int(config["n_epochs"]),
                "gamma": float(config["gamma"]),
                "gae_lambda": float(config["gae_lambda"]),
                "clip_range": float(config["clip_range"]),
                "ent_coef": float(config["ent_coef"]),
                "vf_coef": float(config["vf_coef"]),
                "随机种子": int(config["seed"]),
                "月度特征维度": int(len(monthly_features.columns) - 1),
                "训练月份数": len(context["split"].train),
                "验证月份数": len(context["split"].val),
                "回测月份数": len(context["split"].test),
            }
        ]
    )

    practice_metrics = pd.DataFrame(
        [
            {"阶段": "验证集", **validation["metrics"]},
            {"阶段": "回测集_PPO", **backtest["results"]["ppo"]["metrics"]},
        ]
    )

    monthly_metadata_slice = (
        monthly_metadata.loc[
            monthly_metadata["month"].isin(context["split"].train + context["split"].val + context["split"].test)
        ]
        .sort_values("month")
        .reset_index(drop=True)
    )

    training_metrics = training["train_metrics"].copy().sort_values("timesteps").reset_index(drop=True)
    eval_metrics = training["eval_metrics"].copy().sort_values("timesteps").reset_index(drop=True)

    report_lines = [
        "# 详细运行报告",
        "",
        "## 一、实验与数据背景",
        "",
        f"- 项目目录: {config['project_root']}",
        f"- 主数据文件: {context['csv_path']}",
        f"- 主样本区间: {config['sample_start']} 至 {config['sample_end']}",
        f"- 原始记录数: {context['data_quality_report']['rows_before_cleaning']}",
        f"- 去重后记录数: {context['data_quality_report']['rows_after_duplicate_aggregation']}",
        f"- 主样本记录数: {context['data_quality_report']['rows_after_sample_filter']}",
        f"- 预热期: {', '.join(split['warmup']) if split['warmup'] else '无'}",
        f"- 训练集: {', '.join(split['train']) if split['train'] else '无'}",
        f"- 验证集: {', '.join(split['val']) if split['val'] else '无'}",
        f"- 回测集: {', '.join(split['test']) if split['test'] else '无'}",
        "",
        "## 二、模型数据",
        "",
        _frame_to_markdown(model_profile),
        "",
        "## 三、月度基础样本统计",
        "",
        _frame_to_markdown(
            monthly_metadata_slice[
                [
                    "month",
                    "forecast_monthly_net_demand_mwh",
                    "actual_monthly_net_demand_mwh",
                    "da_price_mean",
                    "id_price_mean",
                    "lt_price_m",
                    "hourly_net_load_da_vol",
                    "hourly_spread_vol",
                ]
            ]
        ),
        "",
        "## 四、训练过程数据",
        "",
        "### 4.1 训练指标尾部样本",
        "",
        _frame_to_markdown(training_metrics.tail(10)),
        "",
        "### 4.2 评估指标",
        "",
        _frame_to_markdown(eval_metrics),
        "",
        "## 五、算法应用效果",
        "",
        "### 5.1 验证集与回测集核心指标",
        "",
        _frame_to_markdown(practice_metrics),
        "",
        "### 5.2 回测基准策略比较",
        "",
        _frame_to_markdown(benchmark_frame),
        "",
        "### 5.3 PPO 月度决策与执行结果",
        "",
        _frame_to_markdown(ppo_monthly),
        "",
        "### 5.4 PPO 月度动作参数",
        "",
        _frame_to_markdown(ppo_actions),
        "",
        "### 5.5 验证集月度结果",
        "",
        _frame_to_markdown(validation_monthly),
        "",
        "## 六、论文写作可引用结论",
        "",
        f"- PPO 在回测集的累计采购成本为 {backtest['results']['ppo']['metrics']['cumulative_procurement_cost']:.2f}。",
        f"- PPO 在回测集的 CVaR(95%) 为 {backtest['results']['ppo']['metrics']['cvar95']:.2f}，套保误差为 {backtest['results']['ppo']['metrics']['hedge_error']:.4f}。",
        f"- 与固定锁定比例基准相比，成本节约为 {benchmark_frame.loc[benchmark_frame['strategy'] == 'ppo', 'cost_savings_vs_fixed_lock'].iloc[0]:.2f}。",
        f"- PPO 月均调整量为 {backtest['results']['ppo']['metrics']['avg_adjustment_mwh']:.4f}，可直接作为算法执行强度的实践指标。",
        "- 详细训练轨迹、月度动作和基准比较表已同步导出到 outputs/metrics/，可直接用于论文正文表格或附录整理。",
        "",
    ]
    return "\n".join(report_lines)


def _save_detailed_practice_tables(context: dict, training: dict, validation: dict, backtest: dict) -> None:
    metrics_dir: Path = context["output_paths"]["metrics"]

    model_profile = pd.DataFrame(
        [
            {
                "算法": "PPO",
                "策略网络": context["config"]["policy"],
                "训练设备": training["device"],
                "总训练步数": int(context["config"]["total_timesteps"]),
                "n_steps": int(context["config"]["n_steps"]),
                "batch_size": int(context["config"]["batch_size"]),
                "n_epochs": int(context["config"]["n_epochs"]),
                "gamma": float(context["config"]["gamma"]),
                "gae_lambda": float(context["config"]["gae_lambda"]),
                "clip_range": float(context["config"]["clip_range"]),
                "ent_coef": float(context["config"]["ent_coef"]),
                "vf_coef": float(context["config"]["vf_coef"]),
                "随机种子": int(context["config"]["seed"]),
            }
        ]
    )
    model_profile.to_csv(metrics_dir / "model_profile.csv", index=False)

    algorithm_effect = pd.concat(
        [
            pd.DataFrame([{"阶段": "验证集", **validation["metrics"]}]),
            pd.DataFrame([{"阶段": "回测集_PPO", **backtest["results"]["ppo"]["metrics"]}]),
            backtest["metrics_frame"].assign(阶段="回测基准比较"),
        ],
        ignore_index=True,
        sort=False,
    )
    algorithm_effect.to_csv(metrics_dir / "algorithm_practice_effect.csv", index=False)

    backtest["results"]["ppo"]["monthly_results"].sort_values("month").to_csv(
        metrics_dir / "ppo_monthly_practice_data.csv",
        index=False,
    )


def main() -> dict:
    context = prepare_project_context("/Users/dk/py/elec")
    logger = context["logger"]
    logger.info("开始执行全量流水线。")

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
    detailed_report = _build_detailed_run_report(context, training, validation, backtest)
    save_markdown(detailed_report, context["output_paths"]["reports"] / "detailed_run_report.md")
    _save_detailed_practice_tables(context, training, validation, backtest)
    logger.info("流水线执行完成。")
    return {
        "training": training,
        "validation": validation,
        "backtest": backtest,
    }


if __name__ == "__main__":
    main()
