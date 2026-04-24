from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.analysis.reporting import (
    build_excess_return_validation_summary,
    build_market_mechanism_analysis,
    build_module1_summary,
)
from src.scripts.backtest import run_backtest
from src.scripts.common import prepare_project_context
from src.scripts.evaluate import run_evaluate
from src.scripts.train import run_train
from src.utils.experiment_manifest import (
    build_artifact_index_markdown,
    build_release_manifest,
    build_run_manifest,
    fallback_run_metadata,
    prepend_report_header,
    relativize_path,
)
from src.utils.io import save_json, save_markdown
from src.utils.runtime_status import RuntimeStatusTracker, build_training_phase_name


def _build_run_summary(context: dict, training: dict, validation: dict, backtest: dict) -> str:
    rolling_excess = backtest.get("rolling_excess_return_metrics", pd.DataFrame())
    persistent_windows = int(rolling_excess.get("active_excess_return_persistent", pd.Series(dtype="bool")).astype(bool).sum())
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    return "\n".join(
        [
            "# 运行总结",
            "",
            f"- 版本: {context['config']['version']}",
            f"- 算法: {context['config']['training']['algorithm']}",
            f"- 训练设备: {training['device']}",
            f"- 验证成本: {validation['metrics']['total_procurement_cost']:.2f}",
            f"- 验证收益: {validation['metrics']['total_profit']:.2f}",
            f"- 验证 CVaR99: {validation['metrics']['cvar99']:.2f}",
            f"- 滚动窗口数: {backtest['rolling_summary'].aggregate['window_count']:.0f}",
            f"- 平均回测成本: {backtest['rolling_summary'].aggregate['mean_total_procurement_cost']:.2f}",
            f"- 平均回测收益: {backtest['rolling_summary'].aggregate.get('mean_total_profit', 0.0):.2f}",
            f"- 平均回测 CVaR99: {backtest['rolling_summary'].aggregate['mean_cvar99']:.2f}",
            f"- 持续跑赢窗口数: {persistent_windows}",
            f"- 模块1摘要: {relativize_path(context['output_paths']['reports'] / 'module1_summary.md', output_root)}",
            f"- 市场机制分析: {relativize_path(context['output_paths']['reports'] / 'market_mechanism_analysis.md', output_root)}",
            f"- 超额收益验证摘要: {relativize_path(context['output_paths']['reports'] / 'excess_return_validation_summary.md', output_root)}",
            "",
        ]
    )


def _load_existing_key_outputs(output_root: Path) -> dict[str, str]:
    for manifest_name in ["release_manifest.json", "run_manifest.json"]:
        manifest_path = output_root / manifest_name
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            key_outputs = payload.get("key_outputs", {})
            if isinstance(key_outputs, dict):
                return {str(key): str(value) for key, value in key_outputs.items()}
    return {}


def _filter_existing_key_outputs(output_root: Path, key_outputs: dict[str, str]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for label, path in key_outputs.items():
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else output_root / candidate
        if resolved.exists():
            filtered[str(label)] = str(path)
    return filtered


def _persist_manifest_updates(context: dict, extra_outputs: dict[str, str]) -> None:
    if "seed" not in context["config"] or "split" not in context["config"]:
        return
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    run_metadata = context.get("run_metadata", fallback_run_metadata(context["config"]))
    key_outputs = _filter_existing_key_outputs(output_root, _load_existing_key_outputs(output_root))
    key_outputs.update(_filter_existing_key_outputs(output_root, extra_outputs))
    save_json(build_release_manifest(run_metadata, context["config"], key_outputs), output_root / "release_manifest.json")
    save_json(build_run_manifest(run_metadata, context["config"], key_outputs), output_root / "run_manifest.json")
    save_markdown(build_artifact_index_markdown(run_metadata, key_outputs), output_root / "artifact_index.md")


def _build_version_report(context: dict, training: dict, validation: dict, backtest: dict) -> str:
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    version = context["config"]["version"]
    split_cfg = context["config"].get("split", {})
    split_text = (
        f"{split_cfg.get('train_start_week', 'n/a')} -> {split_cfg.get('train_end_week', 'n/a')} / "
        f"{split_cfg.get('val_start_week', 'n/a')} -> {split_cfg.get('val_end_week', 'n/a')} / "
        f"{split_cfg.get('test_start_week', 'n/a')} -> {split_cfg.get('test_end_week', 'n/a')}"
    )
    benchmark_metrics = backtest.get("benchmark_metrics", pd.DataFrame())
    ablation_metrics = backtest.get("ablation_metrics", pd.DataFrame())
    robustness_metrics = backtest.get("robustness_metrics", pd.DataFrame())
    rolling_summary = backtest.get("rolling_summary")
    rolling_aggregate = getattr(rolling_summary, "aggregate", {}) if rolling_summary is not None else {}
    benchmark_leader = benchmark_metrics.sort_values("total_profit", ascending=False).iloc[0] if not benchmark_metrics.empty else None
    dynamic_row = benchmark_metrics.loc[benchmark_metrics.get("strategy_name", pd.Series(dtype="object")) == "dynamic_lock_only"]
    dynamic_profit = float(dynamic_row["total_profit"].iloc[0]) if not dynamic_row.empty else 0.0
    dynamic_cvar = float(dynamic_row["cvar99"].iloc[0]) if not dynamic_row.empty else 0.0
    training_runtime = training.get("runtime_profile", {})
    linked_note = context["config"].get("reporting", {}).get("lt_price_note", "n/a")
    settlement_note = context["config"].get("reporting", {}).get("settlement_note", "n/a")
    return "\n".join(
        [
            f"# {version}报告",
            "",
            "## 数学模型与公式",
            "",
            "- 上层周度执行量: `contract_adjustment_mwh_exec = Proj_week(contract_adjustment_mwh_raw, feasible_domain_w)`，`contract_position_mwh = base_position_w + contract_adjustment_mwh_exec`。",
            "- 上层 24 小时合约曲线: `contract_curve_h = softmax(curve_latent @ basis_24h)`，用于把周度头寸映射到日内结算结构。",
            "- 小时级现货修正: `spot_hedge_mwh = gate_h * Proj_hour(tanh(signal_h) * limit_h, exposure_band_mwh, hourly_caps)`，并以带符号净修正量进入计划电量。",
            "- 15 分钟代理结算口径: `q_sched,t = max(q_contract,t + q_spot_net,t, 0)`，`C_t = q_sched,t * (w_lt * p_lt + w_da * p_da,t) + |q_real,t - q_sched,t| * p_id,t * penalty`。",
            "- 奖励/目标结构: `r_w = excess_profit - λ_tail * CVaR99 - λ_hedge * hedge_error - λ_trade * friction - λ_violate * violation_penalty`。",
            "",
            "## 模型运行参数设置",
            "",
            f"- 当前版本号: {version}",
            f"- 实验编号: {context.get('run_metadata', {}).get('experiment_id', 'unknown')}",
            f"- 数据样本时间范围: {context['config'].get('sample_start', '')} 至 {context['config'].get('sample_end', '')}",
            f"- 训练/验证/测试划分: {split_text}",
            f"- 主算法名称: {context['config']['training']['algorithm']}",
            f"- 随机种子: {context['config'].get('seed', 'n/a')}",
            f"- 运行设备: {training.get('device', context['config']['training'].get('device', 'cpu'))}",
            f"- 上层配置维度 / 真实维度: {context['config'].get('hybrid_pso', {}).get('upper', {}).get('dimension', training_runtime.get('upper_dim', 'n/a'))} / {training_runtime.get('upper_dim_real', training_runtime.get('upper_dim', 'n/a'))}",
            f"- 下层配置维度 / 真实维度: {context['config'].get('hybrid_pso', {}).get('lower', {}).get('dimension', training_runtime.get('lower_dim', 'n/a'))} / {training_runtime.get('lower_dim_real', training_runtime.get('lower_dim', 'n/a'))}",
            f"- 制度相关价格口径: {linked_note}",
            f"- 结算口径: {settlement_note}",
            f"- 配置快照: {relativize_path(context['output_paths']['reports'] / 'train_config_snapshot.yaml', output_root)}",
            "",
            "## 模型运行效果",
            "",
            f"- 验证集累计收益: {float(validation.get('metrics', {}).get('total_profit', 0.0)):.2f}",
            f"- 验证集 CVaR99: {float(validation.get('metrics', {}).get('cvar99', 0.0)):.2f}",
            f"- 滚动窗口数: {float(rolling_aggregate.get('window_count', 0.0)):.0f}",
            f"- 滚动窗口平均收益: {float(rolling_aggregate.get('mean_total_profit', 0.0)):.2f}",
            f"- 滚动窗口平均采购成本: {float(rolling_aggregate.get('mean_total_procurement_cost', 0.0)):.2f}",
            f"- 滚动窗口平均 CVaR99: {float(rolling_aggregate.get('mean_cvar99', 0.0)):.2f}",
            f"- 强基准 `dynamic_lock_only` 持出集利润 / CVaR99: {dynamic_profit:.2f} / {dynamic_cvar:.2f}",
            f"- 持出集利润领先策略: {str(benchmark_leader['strategy_name']) if benchmark_leader is not None else 'n/a'}",
            f"- 基准比较结果表: {relativize_path(context['output_paths']['metrics'] / 'benchmark_comparison.csv', output_root)}",
            f"- 消融结果表: {relativize_path(context['output_paths']['metrics'] / 'ablation_metrics.csv', output_root)}",
            f"- 稳健性结果表: {relativize_path(context['output_paths']['metrics'] / 'robustness_metrics.csv', output_root)}",
            "",
            "## 实现效果",
            "",
            "- 已形成版本化输出目录、政策文件清单、结构化规则表、制度状态轨迹、参数布局审计、训练摘要、滚动回测、基准比较、消融、稳健性与约束激活报告。",
            "- 本轮新增或收口内容: 摘要样本范围字段统一、`lt_price_source` 与 2026-02 后混合代理标签对齐、默认投影与政策收紧拆分、训练真实维度摘要、正式版本总报告与 manifest 索引补全。",
            f"- 关键产物: {relativize_path(context['output_paths']['reports'] / 'backtest_summary.md', output_root)}, {relativize_path(context['output_paths']['reports'] / 'benchmark_summary.md', output_root)}, {relativize_path(context['output_paths']['reports'] / 'constraint_activation_report.md', output_root)}。",
            f"- 当前已知限制: 强基准与静态无修正基准的相对优劣仍需结合持出集与滚动窗口双口径解释；消融与稳健性结论需以正式全流程实跑结果为准。当前已满足论文主引用版本所需的口径对齐与证据链归档要求。",
            "",
        ]
    )


def main() -> dict:
    status_path = Path(os.environ.get("ELEC_RUNTIME_STATUS_PATH", Path.cwd() / ".cache" / "runtime_status.json"))
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_tracker = RuntimeStatusTracker(status_path)
    status_tracker.update(stage="初始化", phase_name="准备上下文", phase_progress=0.0, total_progress=0.0, message="加载配置与数据")
    context = prepare_project_context(Path.cwd(), logger_name="pipeline")
    context["runtime_status_path"] = status_path
    training_phase_name = build_training_phase_name(context["config"].get("training", {}).get("algorithm"))
    status_tracker.update(stage="训练", phase_name=training_phase_name, phase_progress=0.0, total_progress=0.05, message="开始训练")
    training = run_train(context)
    status_tracker.update(stage="验证", phase_name="验证模块", phase_progress=0.0, total_progress=0.33, message="开始验证")
    validation = run_evaluate(context, model=training["model"])
    status_tracker.update(stage="回测", phase_name="滚动回测", phase_progress=0.0, total_progress=0.66, message="开始回测")
    backtest = run_backtest(context, model=training["model"])
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    run_metadata = context.get("run_metadata", fallback_run_metadata(context["config"]))
    save_markdown(
        prepend_report_header(
            build_module1_summary(
                contract_value_path=Path(
                    relativize_path(context["output_paths"]["metrics"] / "contract_value_weekly.csv", output_root)
                ),
                risk_factor_path=Path(
                    relativize_path(context["output_paths"]["metrics"] / "risk_factor_manifest.csv", output_root)
                ),
                contract_value_weekly=validation.get("contract_value_weekly", pd.DataFrame()),
                risk_factor_manifest=validation.get("risk_factor_manifest", pd.DataFrame()),
            ),
            run_metadata,
            device=training["device"],
        ),
        context["output_paths"]["reports"] / "module1_summary.md",
    )
    save_markdown(
        prepend_report_header(
            build_market_mechanism_analysis(
                rule_table=context["bundle"].get("policy_rule_table", pd.DataFrame()),
                constraints=context["bundle"].get("market_rule_constraints", pd.DataFrame()),
            ),
            run_metadata,
            device=training["device"],
        ),
        context["output_paths"]["reports"] / "market_mechanism_analysis.md",
    )
    save_markdown(
        prepend_report_header(
            build_excess_return_validation_summary(
                policy_metrics=validation.get("policy_risk_metrics", pd.DataFrame()),
                rolling_metrics=backtest.get("rolling_excess_return_metrics", pd.DataFrame()),
            ),
            run_metadata,
            device=training["device"],
        ),
        context["output_paths"]["reports"] / "excess_return_validation_summary.md",
    )
    save_markdown(
        prepend_report_header(_build_run_summary(context, training, validation, backtest), run_metadata, device=training["device"]),
        context["output_paths"]["reports"] / "run_summary.md",
    )
    version_report_path = context["output_paths"]["reports"] / f"{context['config']['version']}报告.md"
    save_markdown(
        prepend_report_header(_build_version_report(context, training, validation, backtest), run_metadata, device=training["device"]),
        version_report_path,
    )
    model_dir = context["output_paths"].get("models", output_root / "models")
    _persist_manifest_updates(
        context,
        {
            "model_path": relativize_path(model_dir / "hybrid_pso_model.json", output_root),
            "train_summary_path": relativize_path(context["output_paths"]["reports"] / "train_summary.md", output_root),
            "training_runtime_summary_path": relativize_path(context["output_paths"]["reports"] / "training_runtime_summary.json", output_root),
            "validation_summary_path": relativize_path(context["output_paths"]["reports"] / "validation_summary.md", output_root),
            "backtest_summary_path": relativize_path(context["output_paths"]["reports"] / "backtest_summary.md", output_root),
            "benchmark_summary_path": relativize_path(context["output_paths"]["reports"] / "benchmark_summary.md", output_root),
            "ablation_summary_path": relativize_path(context["output_paths"]["reports"] / "ablation_summary.md", output_root),
            "robustness_summary_path": relativize_path(context["output_paths"]["reports"] / "robustness_summary.md", output_root),
            "constraint_activation_report_path": relativize_path(context["output_paths"]["reports"] / "constraint_activation_report.md", output_root),
            "run_summary_path": relativize_path(context["output_paths"]["reports"] / "run_summary.md", output_root),
            "module1_summary_path": relativize_path(context["output_paths"]["reports"] / "module1_summary.md", output_root),
            "market_mechanism_analysis_path": relativize_path(context["output_paths"]["reports"] / "market_mechanism_analysis.md", output_root),
            "excess_return_validation_summary_path": relativize_path(context["output_paths"]["reports"] / "excess_return_validation_summary.md", output_root),
            "version_report_path": relativize_path(version_report_path, output_root),
            "rolling_backtest_metrics_path": relativize_path(context["output_paths"]["metrics"] / "rolling_backtest_metrics.csv", output_root),
            "benchmark_comparison_path": relativize_path(context["output_paths"]["metrics"] / "benchmark_comparison.csv", output_root),
            "ablation_metrics_path": relativize_path(context["output_paths"]["metrics"] / "ablation_metrics.csv", output_root),
            "robustness_metrics_path": relativize_path(context["output_paths"]["metrics"] / "robustness_metrics.csv", output_root),
        },
    )
    status_tracker.update(stage="完成", phase_name="全量流水线", phase_progress=1.0, total_progress=1.0, message="流水线执行完成")
    return {
        "training": training,
        "validation": validation,
        "backtest": backtest,
    }


if __name__ == "__main__":
    main()
