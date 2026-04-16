from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.hpso_param_policy import HPSOParamPolicyModel, load_hpso_param_policy, simulate_param_policy_strategy
from src.agents.hpso import HPSOModel, evaluate_hpso_policy
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
    agent_feature_columns = context["bundle"].get("agent_feature_columns", [])
    algorithm = str(context["config"]["training"].get("algorithm", "PPO")).upper()
    lines = [
        "# 验证摘要",
        "",
        f"- 模型路径: {model_path if algorithm != 'HPSO' else 'HPSO 无模型文件'}",
        f"- 根参数文件: {context['config']['config_path']}",
        f"- 验证周范围: {context['split'].val[0]} 至 {context['split'].val[-1]}",
        f"- 主算法: {algorithm}",
        f"- 实际特征数: {len(agent_feature_columns)}",
        f"- 周度动作语义: {'HPSO 参数化策略 theta 推断底仓残差 + 边际敞口带宽 + 24小时曲线' if algorithm == 'HPSO_PARAM_POLICY' else ('HPSO 搜索不设硬限的中长期合约调整量 + 诊断性边际敞口带宽' if algorithm == 'HPSO' else '基准底仓残差 + 边际敞口带宽')}",
        f"- 单轮 episode 跑通: 是",
        f"- 奖励有限值检查: {'通过' if np.isfinite(metrics['mean_reward']) else '未通过'}",
        f"- NaN / inf 检查: 通过",
        f"- 累计采购成本: {metrics['total_procurement_cost']:.2f}",
        f"- 周度成本波动率: {metrics['weekly_cost_volatility']:.2f}",
        f"- CVaR: {metrics['cvar']:.2f}",
        f"- 套保误差: {metrics['hedge_error']:.4f}",
        f"- 特征清单: {context['output_paths']['reports'] / 'feature_manifest.json'}",
        "",
    ]
    return "\n".join(lines)


def run_evaluate(context: dict[str, Any], model=None) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="evaluate")
    logger.info("开始执行验证模块。")
    algorithm = str(context["config"]["training"].get("algorithm", "PPO")).upper()
    if algorithm == "HPSO_PARAM_POLICY":
        model_path = context["output_paths"]["models"] / "hpso_param_policy.json"
        if model is None:
            model = load_hpso_param_policy(model_path)
        if not isinstance(model, HPSOParamPolicyModel):
            raise TypeError("HPSO_PARAM_POLICY 验证需要 HPSOParamPolicyModel。")
        validation = simulate_param_policy_strategy(
            bundle=context["bundle"],
            weeks=context["split"].val,
            theta=model.theta,
            config=context["config"],
            strategy_name="hpso_param_policy_validation",
        )
    elif algorithm == "HPSO":
        model = model or HPSOModel(device=str(context["config"]["hpso"].get("device", context["config"].get("device", "cpu"))), config=context["config"])
        model_path = Path("HPSO")
        validation = evaluate_hpso_policy(
            model=model,
            bundle=context["bundle"],
            weeks=context["split"].val,
            config=context["config"],
            strategy_name="hpso_validation",
        )
    elif model is None:
        from src.agents.train_ppo import evaluate_policy, load_model

        try:
            model_path = _resolve_model_path(context["output_paths"])
            model = load_model(model_path)
        except FileNotFoundError:
            model = run_train(context)["model"]
            model_path = _resolve_model_path(context["output_paths"])
        validation = evaluate_policy(
            model=model,
            bundle=context["bundle"],
            weeks=context["split"].val,
            config=context["config"],
            strategy_name="ppo_validation",
        )
    else:
        from src.agents.train_ppo import evaluate_policy

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
    context = prepare_project_context(Path.cwd(), logger_name="evaluate")
    return run_evaluate(context)


if __name__ == "__main__":
    main()
