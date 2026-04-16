from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.hybrid_pso import load_hybrid_pso_model
from src.backtest.materialize import materialize_particle_pair
from src.scripts.common import prepare_project_context, subset_bundle_for_weeks
from src.scripts.train import run_train
from src.utils.io import save_markdown
from src.utils.logger import configure_logging


def build_validation_summary(context: dict[str, Any], validation: dict[str, Any], model_path: Path) -> str:
    metrics = validation["metrics"]
    return "\n".join(
        [
            "# 验证摘要",
            "",
            f"- 模型路径: {model_path}",
            f"- 验证周范围: {context['split'].val[0]} 至 {context['split'].val[-1]}",
            f"- 主算法: {context['config']['training']['algorithm']}",
            f"- 累计采购成本: {metrics['total_procurement_cost']:.2f}",
            f"- 累计经济收益: {metrics['total_profit']:.2f}",
            f"- 周度成本波动率: {metrics['weekly_cost_volatility']:.2f}",
            f"- CVaR99: {metrics['cvar99']:.2f}",
            f"- 套保误差: {metrics['hedge_error']:.4f}",
            "",
        ]
    )


def run_evaluate(context: dict[str, Any], model=None) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="evaluate")
    logger.info("开始执行验证模块。")
    model_path = context["output_paths"]["models"] / "hybrid_pso_model.json"
    if model is None:
        if not model_path.exists():
            model = run_train(context)["model"]
        else:
            model = load_hybrid_pso_model(model_path)

    val_bundle = subset_bundle_for_weeks(context["bundle"], context["split"].val)
    validation = materialize_particle_pair(
        tensor_bundle=val_bundle["tensor_bundle"],
        upper_particle=model.upper_best,
        lower_particle=model.lower_best,
        strategy_name="hybrid_pso_validation",
        config=context["config"],
    )
    validation.weekly_results.to_csv(context["output_paths"]["metrics"] / "validation_weekly_results.csv", index=False)
    pd.DataFrame([validation.metrics]).to_csv(context["output_paths"]["metrics"] / "validation_metrics.csv", index=False)
    save_markdown(
        build_validation_summary(
            context,
            {
                "metrics": validation.metrics,
            },
            model_path,
        ),
        context["output_paths"]["reports"] / "validation_summary.md",
    )
    logger.info("验证模块执行完成。")
    return {
        "weekly_results": validation.weekly_results,
        "hourly_results": validation.hourly_results,
        "settlement_results": validation.settlement_results,
        "metrics": validation.metrics,
    }


def main() -> dict[str, Any]:
    context = prepare_project_context(Path.cwd(), logger_name="evaluate")
    return run_evaluate(context)


if __name__ == "__main__":
    main()
