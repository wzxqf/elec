from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents.train_ppo import evaluate_policy, load_model
from src.scripts.common import prepare_project_context


def main() -> dict:
    context = prepare_project_context("/Users/dk/py/elec")
    logger = context["logger"]
    model_path = context["output_paths"]["models"] / "ppo_elec_env.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    model = load_model(model_path, context["config"])
    evaluation = evaluate_policy(
        model=model,
        bundle=context["bundle"],
        months=context["split"].val,
        config=context["config"],
        strategy_name="ppo_validation",
    )
    evaluation["monthly_results"].to_csv(context["output_paths"]["metrics"] / "validation_monthly_results.csv", index=False)
    metrics_frame = pd.DataFrame([evaluation["metrics"]])
    metrics_frame.to_csv(context["output_paths"]["metrics"] / "validation_summary.csv", index=False)
    logger.info("Validation complete.")
    return evaluation


if __name__ == "__main__":
    main()
