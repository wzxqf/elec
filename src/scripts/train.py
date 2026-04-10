from __future__ import annotations

from src.agents.train_ppo import train_model
from src.scripts.common import prepare_project_context


def main() -> dict:
    context = prepare_project_context("/Users/dk/py/elec")
    logger = context["logger"]
    logger.info("开始执行 PPO 训练。")
    training = train_model(
        bundle=context["bundle"],
        train_months=context["train_sequence"],
        val_months=context["split"].val,
        config=context["config"],
        output_paths=context["output_paths"],
    )
    logger.info("训练完成，模型已保存至 %s", training["model_path"])
    return training


if __name__ == "__main__":
    main()
