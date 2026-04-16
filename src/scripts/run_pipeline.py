from __future__ import annotations

import os
from pathlib import Path

from src.scripts.backtest import run_backtest
from src.scripts.common import prepare_project_context
from src.scripts.evaluate import run_evaluate
from src.scripts.train import run_train
from src.utils.io import save_markdown
from src.utils.runtime_status import RuntimeStatusTracker


def _build_run_summary(context: dict, training: dict, validation: dict, backtest: dict) -> str:
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
    save_markdown(_build_run_summary(context, training, validation, backtest), context["output_paths"]["reports"] / "run_summary.md")
    status_tracker.update(stage="完成", phase_name="全量流水线", phase_progress=1.0, total_progress=1.0, message="流水线执行完成")
    return {
        "training": training,
        "validation": validation,
        "backtest": backtest,
    }


if __name__ == "__main__":
    main()
