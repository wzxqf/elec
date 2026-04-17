from __future__ import annotations

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
from src.utils.experiment_manifest import fallback_run_metadata, prepend_report_header, relativize_path
from src.utils.io import save_markdown
from src.utils.runtime_status import RuntimeStatusTracker


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


def main() -> dict:
    status_path = Path(os.environ.get("ELEC_RUNTIME_STATUS_PATH", Path.cwd() / ".cache" / "runtime_status.json"))
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_tracker = RuntimeStatusTracker(status_path)
    status_tracker.update(stage="初始化", phase_name="准备上下文", phase_progress=0.0, total_progress=0.0, message="加载配置与数据")
    context = prepare_project_context(Path.cwd(), logger_name="pipeline")
    context["runtime_status_path"] = status_path
    status_tracker.update(stage="训练", phase_name="Hybrid PSO 训练", phase_progress=0.0, total_progress=0.05, message="开始训练")
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
    status_tracker.update(stage="完成", phase_name="全量流水线", phase_progress=1.0, total_progress=1.0, message="流水线执行完成")
    return {
        "training": training,
        "validation": validation,
        "backtest": backtest,
    }


if __name__ == "__main__":
    main()
