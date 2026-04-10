from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.train_ppo import evaluate_policy, load_model
from src.scripts.common import prepare_project_context
from src.scripts.train import run_train
from src.utils.io import save_markdown
from src.utils.logger import configure_logging


def _resolve_model_path(output_paths: dict[str, Path]) -> Path:
    for candidate in [output_paths["models"] / "ppo_best.zip", output_paths["models"] / "ppo_latest.zip"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到已训练模型，请先执行训练。")


def build_validation_summary(context: dict[str, Any], validation: dict[str, Any], model_path: Path) -> str:
    metrics = validation["metrics"]
    lines = [
        "# 验证摘要",
        "",
        f"- 模型路径: {model_path}",
        f"- 验证周范围: {context['split'].val[0]} 至 {context['split'].val[-1]}",
        f"- 单轮 episode 跑通: 是",
        f"- 奖励有限值检查: {'通过' if np.isfinite(metrics['mean_reward']) else '未通过'}",
        f"- NaN / inf 检查: 通过",
        f"- 累计采购成本: {metrics['total_procurement_cost']:.2f}",
        f"- 周度成本波动率: {metrics['weekly_cost_volatility']:.2f}",
        f"- CVaR: {metrics['cvar']:.2f}",
        f"- 套保误差: {metrics['hedge_error']:.4f}",
        "",
    ]
    return "\n".join(lines)


def run_evaluate(context: dict[str, Any], model=None) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="evaluate")
    logger.info("开始执行验证模块。")
    if model is None:
        try:
            model_path = _resolve_model_path(context["output_paths"])
            model = load_model(model_path)
        except FileNotFoundError:
            model = run_train(context)["model"]
            model_path = _resolve_model_path(context["output_paths"])
    else:
        model_path = _resolve_model_path(context["output_paths"])

    validation = evaluate_policy(
        model=model,
        bundle=context["bundle"],
        weeks=context["split"].val,
        config=context["config"],
        strategy_name="ppo_validation",
    )

    metrics = validation["metrics"]
    numeric_values = pd.Series([value for value in metrics.values() if isinstance(value, (int, float))], dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("验证结果存在 NaN 或 inf。")

    validation["weekly_results"].to_csv(context["output_paths"]["metrics"] / "validation_weekly_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(context["output_paths"]["metrics"] / "validation_metrics.csv", index=False)
    save_markdown(
        build_validation_summary(context, validation, model_path),
        context["output_paths"]["reports"] / "validation_summary.md",
    )
    logger.info("验证模块执行完成。")
    return validation


def main() -> dict[str, Any]:
    context = prepare_project_context("/Users/dk/py/elec", logger_name="evaluate")
    return run_evaluate(context)


if __name__ == "__main__":
    main()
