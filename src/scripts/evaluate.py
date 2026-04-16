from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.excess_return import build_policy_risk_adjusted_metrics
from src.analysis.module1 import build_contract_value_weekly, build_risk_factor_manifest, enrich_weekly_results
from src.agents.hybrid_pso import load_hybrid_pso_model
from src.backtest.materialize import materialize_particle_pair
from src.scripts.common import prepare_project_context, subset_bundle_for_weeks
from src.scripts.train import run_train
from src.utils.io import save_markdown
from src.utils.logger import configure_logging


def build_validation_summary(
    context: dict[str, Any],
    validation: dict[str, Any],
    model_path: Path,
    policy_metrics: pd.DataFrame | None = None,
) -> str:
    metrics = validation["metrics"]
    policy_metrics = policy_metrics if policy_metrics is not None else pd.DataFrame()
    mean_adjusted = float(policy_metrics.get("policy_risk_adjusted_excess_return_w", pd.Series(dtype="float64")).mean() or 0.0)
    mean_penalty = float(policy_metrics.get("policy_risk_penalty_w", pd.Series(dtype="float64")).mean() or 0.0)
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
            f"- 政策风险惩罚均值: {mean_penalty:.4f}",
            f"- 政策风险调整后超额收益均值: {mean_adjusted:.4f}",
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
        compiled_layout=val_bundle.get("compiled_parameter_layout"),
    )
    validation.weekly_results.to_csv(context["output_paths"]["metrics"] / "validation_weekly_results.csv", index=False)
    pd.DataFrame([validation.metrics]).to_csv(context["output_paths"]["metrics"] / "validation_metrics.csv", index=False)
    analysis_input = enrich_weekly_results(
        validation.weekly_results,
        weekly_metadata=val_bundle["weekly_metadata"],
        weekly_features=val_bundle["weekly_features"],
        policy_state_trace=val_bundle["policy_state_trace"],
    )
    contract_value_weekly = build_contract_value_weekly(analysis_input)
    risk_factor_manifest = build_risk_factor_manifest(analysis_input)
    epsilon = float(context["config"].get("analysis_v035", {}).get("sharpe_epsilon", 1.0e-6))
    policy_risk_metrics = build_policy_risk_adjusted_metrics(analysis_input, epsilon=epsilon)
    contract_value_weekly.to_csv(context["output_paths"]["metrics"] / "contract_value_weekly.csv", index=False)
    risk_factor_manifest.to_csv(context["output_paths"]["metrics"] / "risk_factor_manifest.csv", index=False)
    policy_risk_metrics.to_csv(context["output_paths"]["metrics"] / "policy_risk_adjusted_metrics.csv", index=False)
    save_markdown(
        build_validation_summary(
            context,
            {
                "metrics": validation.metrics,
            },
            model_path,
            policy_risk_metrics,
        ),
        context["output_paths"]["reports"] / "validation_summary.md",
    )
    logger.info("验证模块执行完成。")
    return {
        "weekly_results": validation.weekly_results,
        "hourly_results": validation.hourly_results,
        "settlement_results": validation.settlement_results,
        "metrics": validation.metrics,
        "contract_value_weekly": contract_value_weekly,
        "risk_factor_manifest": risk_factor_manifest,
        "policy_risk_metrics": policy_risk_metrics,
    }


def main() -> dict[str, Any]:
    context = prepare_project_context(Path.cwd(), logger_name="evaluate")
    return run_evaluate(context)


if __name__ == "__main__":
    main()
