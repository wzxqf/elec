from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import pandas as pd

from src.scripts.backtest import run_backtest
from src.scripts.common import prepare_project_context, split_to_dict
from src.scripts.evaluate import run_evaluate
from src.scripts.train import run_train
from src.utils.io import save_markdown
from src.utils.runtime_status import RuntimeStatusTracker


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_float_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: round(float(value), 6))
    printable = printable.fillna("")
    headers = [str(column) for column in printable.columns]
    rows = [[str(value) for value in row] for row in printable.astype(object).values.tolist()]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _main_result_key(results: dict[str, dict]) -> str:
    if "hpso_param_policy" in results:
        return "hpso_param_policy"
    if "hpso" in results:
        return "hpso"
    return "ppo"


def _model_iterations(config: dict[str, Any], main_key: str) -> int:
    if main_key == "hpso_param_policy":
        return int(config["hpso"]["swarm"]["iterations"])
    if main_key == "hpso":
        return int(config["hpso"]["upper"]["iterations"])
    return int(config["total_timesteps"])


def _action_semantics(main_key: str) -> str:
    if main_key == "hpso_param_policy":
        return "HPSO 参数化策略 theta 推断底仓残差 + 边际敞口带宽 + 24小时曲线"
    if main_key == "hpso":
        return "HPSO 搜索不设硬限的中长期合约调整量 + 诊断性边际敞口带宽"
    return "基准底仓残差 + 边际敞口带宽"


def _build_run_summary(context: dict[str, Any], training: dict[str, Any], validation: dict[str, Any], backtest: dict[str, Any]) -> str:
    config = context["config"]
    data_quality = context["data_quality_report"]
    split_dict = split_to_dict(context["split"])
    validation_metrics = validation["metrics"]
    main_key = _main_result_key(backtest["results"])
    test_metrics = backtest["results"][main_key]["metrics"]
    agent_feature_columns = context["bundle"].get("agent_feature_columns", [])
    runtime_profile = training.get("runtime_profile", {})
    lines = [
        "# 运行总结",
        "",
        "## 数据情况",
        "",
        f"- 数据文件: {context['csv_path']}",
        f"- 主样本区间: {config['sample_start']} 至 {config['sample_end']}",
        f"- 原始覆盖区间: {data_quality['start_before']} 至 {data_quality['end_before']}",
        f"- 重复时间戳个数: {data_quality['duplicate_timestamp_count']}",
        f"- 去重后缺失 15 分钟时点数: {data_quality['missing_timestamp_count_after_aggregation']}",
        f"- 周度样本数: {len(context['bundle']['weekly_metadata'])}",
        f"- 政策来源文件数: {len(context['bundle']['policy_inventory'])}",
        f"- 政策解析失败文件数: {len(context['bundle']['policy_failures'])}",
        f"- 根参数文件: {config['config_path']}",
        "",
        "## 样本划分",
        "",
        f"- 预热周: {', '.join(split_dict['warmup']) if split_dict['warmup'] else '无'}",
        f"- 训练周: {', '.join(split_dict['train'])}",
        f"- 验证周: {', '.join(split_dict['val'])}",
        f"- 回测周: {', '.join(split_dict['test'])}",
        "",
        "## 模型与训练",
        "",
        f"- 主算法: {main_key.upper()}",
        f"- 策略网络: {'参数化规则策略' if main_key == 'hpso_param_policy' else config.get('policy', 'N/A')}",
        f"- 训练步数: {_model_iterations(config, main_key)}",
        f"- 训练设备: {training['device']}",
        f"- 主算设备: {runtime_profile.get('rollout_compute_device', training['device'])}",
        f"- 并行 worker: {runtime_profile.get('parallel_workers', 1)}",
        f"- 是否使用 GPU: {'是' if training['gpu_used'] else '否'}",
        f"- 周度动作语义: {_action_semantics(main_key)}",
        f"- 实际特征数: {len(agent_feature_columns)}",
        f"- 最新模型路径: {training['model_path'] or 'HPSO 无模型文件'}",
        f"- 最优模型路径: {training['best_model_path'] or 'HPSO 无模型文件'}",
        f"- 奖励强基准: {config['reward']['strong_baseline']}",
        "",
        "## 验证结果",
        "",
        f"- 验证累计采购成本: {validation_metrics['total_procurement_cost']:.2f}",
        f"- 验证 CVaR: {validation_metrics['cvar']:.2f}",
        f"- 验证平均奖励: {validation_metrics['mean_reward']:.6f}",
        "",
        "## 回测结果",
        "",
        f"- 回测累计采购成本: {test_metrics['total_procurement_cost']:.2f}",
        f"- 回测周度成本波动率: {test_metrics['weekly_cost_volatility']:.2f}",
        f"- 回测 CVaR: {test_metrics['cvar']:.2f}",
        f"- 回测套保误差: {test_metrics['hedge_error']:.4f}",
        f"- 平均调节电量: {test_metrics['avg_adjustment_mwh']:.4f}",
        "",
        "## 估算口径与说明",
        "",
        f"- 中长期价格: {config['reporting']['lt_price_note']}",
        f"- 结算口径: {config['reporting']['settlement_note']}",
        f"- 训练样本构造: {config['scenario']['training_sequence_method']}",
        f"- 滚动验证窗口数: {len(context['rolling_validation_windows'])}",
        f"- 特征清单: {context['output_paths']['reports'] / 'feature_manifest.json'}",
        f"- 滚动验证摘要: {context['output_paths']['reports'] / 'rolling_validation_summary.md'}",
        f"- 政策规则汇总: {context['output_paths']['reports'] / 'policy_rule_summary.md'}",
        "",
    ]
    return "\n".join(lines)


def _build_detailed_run_report(context: dict[str, Any], training: dict[str, Any], validation: dict[str, Any], backtest: dict[str, Any]) -> str:
    bundle = context["bundle"]
    weekly_metadata = bundle["weekly_metadata"].copy().sort_values("week_start").reset_index(drop=True)
    weekly_features = bundle["weekly_features"].copy().sort_values("week_start").reset_index(drop=True)
    feature_manifest = bundle["feature_manifest"].copy().sort_values(["selected_for_agent", "column"], ascending=[False, True]).reset_index(drop=True)
    validation_weekly = validation["weekly_results"].copy().sort_values("week_start").reset_index(drop=True)
    main_key = _main_result_key(backtest["results"])
    ppo_weekly = backtest["results"][main_key]["weekly_results"].copy().sort_values("week_start").reset_index(drop=True)
    ppo_hourly = backtest["results"][main_key]["hourly_results"].copy().sort_values(["week_start", "hour"]).reset_index(drop=True)
    benchmark_frame = backtest["metrics_frame"].copy()
    policy_inventory = bundle["policy_inventory"].copy()
    policy_rules = bundle["policy_rule_table"].copy()
    policy_trace = bundle["policy_state_trace"].copy().sort_values("week_start").reset_index(drop=True)
    rolling_validation = backtest.get("hparam_search", pd.DataFrame()).copy()

    model_profile = pd.DataFrame(
        [
            {
                "算法": main_key.upper(),
                "策略网络": "参数化规则策略" if main_key == "hpso_param_policy" else context["config"].get("policy", "N/A"),
                "训练设备": training["device"],
                "总训练步数": _model_iterations(context["config"], main_key),
                "状态维度": int(len(bundle.get("agent_feature_columns", []))),
                "动作维度": 2,
                "动作一语义": "64维theta推断底仓残差" if main_key == "hpso_param_policy" else ("不设硬限的中长期合约调整量" if main_key == "hpso" else "dynamic_lock_only 基准底仓残差"),
                "动作二语义": "边际敞口带宽与24小时曲线参数" if main_key == "hpso_param_policy" else ("诊断性边际敞口带宽" if main_key == "hpso" else "边际敞口带宽"),
                "训练周数": len(context["split"].train),
                "验证周数": len(context["split"].val),
                "回测周数": len(context["split"].test),
            }
        ]
    )

    practice_effect = pd.DataFrame(
        [
            {"阶段": "验证集", **validation["metrics"]},
            {"阶段": f"回测集_{main_key.upper()}", **backtest["results"][main_key]["metrics"]},
        ]
    )

    lines = [
        "# 详细运行报告",
        "",
        "## 一、实验与数据背景",
        "",
        f"- 主数据文件: {context['csv_path']}",
        f"- 主样本区间: {context['config']['sample_start']} 至 {context['config']['sample_end']}",
        f"- 15分钟记录数: {len(bundle['quarter'])}",
        f"- 小时记录数: {len(bundle['hourly'])}",
        f"- 周度记录数: {len(weekly_metadata)}",
        f"- 中长期价格口径: {context['config']['reporting']['lt_price_note']}",
        f"- 结算口径: {context['config']['reporting']['settlement_note']}",
        "",
        "## 二、模型数据",
        "",
        _frame_to_markdown(model_profile),
        "",
        "### 2.1 状态空间清单",
        "",
        _frame_to_markdown(feature_manifest[["column", "source", "selected_for_agent", "report_only"]]),
        "",
        "## 三、政策规则与制度状态",
        "",
        "### 3.1 政策文件清单",
        "",
        _frame_to_markdown(policy_inventory[["file_name", "suffix", "publish_time", "parse_status"]]),
        "",
        "### 3.2 结构化规则样本",
        "",
        _frame_to_markdown(policy_rules.head(20)),
        "",
        "### 3.3 周度制度状态",
        "",
        _frame_to_markdown(policy_trace),
        "",
        "## 四、周度基础样本统计",
        "",
        _frame_to_markdown(
            weekly_metadata[
                [
                    "week_start",
                    "is_partial_week",
                    "forecast_weekly_net_demand_mwh",
                    "actual_weekly_net_demand_mwh",
                    "da_price_mean",
                    "id_price_mean",
                    "lt_price_w",
                    "lt_price_source",
                ]
            ]
        ),
        "",
        "## 五、训练过程数据",
        "",
        _frame_to_markdown(training["train_metrics"].tail(10) if not training["train_metrics"].empty else pd.DataFrame()),
        "",
        "## 六、算法应用效果",
        "",
        "### 6.1 核心指标",
        "",
        _frame_to_markdown(practice_effect),
        "",
        "### 6.2 基准策略比较",
        "",
        _frame_to_markdown(benchmark_frame),
        "",
        f"### 6.3 {main_key.upper()} 周度执行结果",
        "",
        _frame_to_markdown(ppo_weekly),
        "",
        f"### 6.4 {main_key.upper()} 小时级修正轨迹样本",
        "",
        _frame_to_markdown(ppo_hourly.head(48)),
        "",
        "### 6.5 验证集周度结果",
        "",
        _frame_to_markdown(validation_weekly),
        "",
        "### 6.6 滚动验证汇总",
        "",
        _frame_to_markdown(rolling_validation),
        "",
        "## 七、论文写作可引用结论",
        "",
        f"- {main_key.upper()} 在回测集总采购成本为 {backtest['results'][main_key]['metrics']['total_procurement_cost']:.2f}。",
        f"- {main_key.upper()} 在回测集 CVaR 为 {backtest['results'][main_key]['metrics']['cvar']:.2f}，套保误差为 {backtest['results'][main_key]['metrics']['hedge_error']:.4f}。",
        f"- 相对固定锁定比例策略，成本节约为 {benchmark_frame.loc[benchmark_frame['strategy'] == main_key, 'cost_savings_vs_fixed_lock'].iloc[0]:.2f}。",
        "- 模型参数、周度结果、小时时间轨迹和基准比较表均已导出为 CSV，可直接用于论文正文或附录。",
        "",
    ]
    return "\n".join(lines)


def _save_detailed_practice_tables(context: dict[str, Any], training: dict[str, Any], validation: dict[str, Any], backtest: dict[str, Any]) -> None:
    metrics_dir: Path = context["output_paths"]["metrics"]
    agent_feature_columns = context["bundle"].get("agent_feature_columns", [])
    main_key = _main_result_key(backtest["results"])

    model_profile = pd.DataFrame(
        [
            {
                "算法": main_key.upper(),
                "策略网络": "参数化规则策略" if main_key == "hpso_param_policy" else context["config"].get("policy", "N/A"),
                "训练设备": training["device"],
                "总训练步数": _model_iterations(context["config"], main_key),
                "n_steps": int(context["config"].get("n_steps", 0)),
                "batch_size": int(context["config"].get("batch_size", 0)),
                "n_epochs": int(context["config"].get("n_epochs", 0)),
                "gamma": float(context["config"].get("gamma", 0.0)),
                "gae_lambda": float(context["config"].get("gae_lambda", 0.0)),
                "clip_range": float(context["config"].get("clip_range", 0.0)),
                "ent_coef": float(context["config"].get("ent_coef", 0.0)),
                "vf_coef": float(context["config"].get("vf_coef", 0.0)),
                "随机种子": int(context["config"]["seed"]),
                "周度特征维度": int(len(agent_feature_columns)),
                "动作一语义": "64维theta推断底仓残差" if main_key == "hpso_param_policy" else ("不设硬限的中长期合约调整量" if main_key == "hpso" else "dynamic_lock_only 基准底仓残差"),
                "动作二语义": "边际敞口带宽与24小时曲线参数" if main_key == "hpso_param_policy" else ("诊断性边际敞口带宽" if main_key == "hpso" else "边际敞口带宽"),
            }
        ]
    )
    model_profile.to_csv(metrics_dir / "model_profile.csv", index=False)

    algorithm_effect = pd.concat(
        [
            pd.DataFrame([{"阶段": "验证集", **validation["metrics"]}]),
            pd.DataFrame([{"阶段": f"回测集_{main_key.upper()}", **backtest["results"][main_key]["metrics"]}]),
            backtest["metrics_frame"].assign(阶段="回测基准比较"),
        ],
        ignore_index=True,
        sort=False,
    )
    algorithm_effect.to_csv(metrics_dir / "algorithm_practice_effect.csv", index=False)
    backtest["results"][main_key]["weekly_results"].to_csv(metrics_dir / f"{main_key}_weekly_practice_data.csv", index=False)
    backtest["results"][main_key]["hourly_results"].to_csv(metrics_dir / f"{main_key}_hourly_rule_practice_data.csv", index=False)


def main() -> dict[str, Any]:
    status_path = Path(os.environ.get("ELEC_RUNTIME_STATUS_PATH", Path.cwd() / ".cache" / "runtime_status.json"))
    status_tracker = RuntimeStatusTracker(status_path)
    status_tracker.update(stage="初始化", phase_name="准备上下文", phase_progress=0.0, total_progress=0.0, message="加载配置与数据")
    context = prepare_project_context(Path.cwd(), logger_name="pipeline")
    context["runtime_status_path"] = status_path
    logger = context["logger"]
    logger.info("开始执行全量流水线。")
    status_tracker.update(stage="初始化", phase_name="准备上下文", phase_progress=1.0, total_progress=0.02, message="上下文准备完成")

    status_tracker.update(stage="训练", phase_name="训练模块", phase_progress=0.0, total_progress=0.02, message="开始训练")
    training = run_train(context)
    status_tracker.update(stage="验证", phase_name="验证模块", phase_progress=0.0, total_progress=1.0 / 3.0, message="开始验证")
    validation = run_evaluate(context, model=training["model"])
    status_tracker.update(stage="验证", phase_name="验证模块", phase_progress=1.0, total_progress=2.0 / 3.0, message="验证完成")
    status_tracker.update(stage="回测", phase_name="回测模块", phase_progress=0.0, total_progress=2.0 / 3.0, message="开始回测")
    backtest = run_backtest(context, model=training["model"])

    save_markdown(
        _build_run_summary(context, training, validation, backtest),
        context["output_paths"]["reports"] / "run_summary.md",
    )
    save_markdown(
        _build_detailed_run_report(context, training, validation, backtest),
        context["output_paths"]["reports"] / "detailed_run_report.md",
    )
    _save_detailed_practice_tables(context, training, validation, backtest)
    status_tracker.update(stage="完成", phase_name="全量流水线", phase_progress=1.0, total_progress=1.0, message="流水线执行完成")
    logger.info("流水线执行完成。")
    return {
        "training": training,
        "validation": validation,
        "backtest": backtest,
    }


if __name__ == "__main__":
    main()
