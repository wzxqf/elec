from __future__ import annotations

from typing import Any

from src.agents.train_ppo import train_model
from src.scripts.common import prepare_project_context, split_to_dict
from src.utils.io import save_markdown
from src.utils.logger import configure_logging


def build_train_summary(context: dict[str, Any], training: dict[str, Any]) -> str:
    config = context["config"]
    lines = [
        "# 训练摘要",
        "",
        "## 数据与样本",
        "",
        f"- 训练数据起止: {context['split'].train[0]} 至 {context['split'].train[-1]}",
        f"- 训练周数: {len(context['split'].train)}",
        f"- 重采样方式: {config['scenario']['training_sequence_method']}",
        f"- 重采样序列长度: {len(context['train_sequence'])}",
        f"- 预热周: {', '.join(split_to_dict(context['split'])['warmup']) if context['split'].warmup else '无'}",
        f"- 政策来源文件数: {len(context['bundle']['policy_inventory'])}",
        f"- 政策解析失败文件数: {len(context['bundle']['policy_failures'])}",
        "",
        "## 超参数",
        "",
        f"- policy: {config['policy']}",
        f"- learning_rate: {config['learning_rate']}",
        f"- n_steps: {config['n_steps']}",
        f"- batch_size: {config['batch_size']}",
        f"- gamma: {config['gamma']}",
        f"- gae_lambda: {config['gae_lambda']}",
        f"- clip_range: {config['clip_range']}",
        f"- ent_coef: {config['ent_coef']}",
        f"- vf_coef: {config['vf_coef']}",
        f"- max_grad_norm: {config['max_grad_norm']}",
        "",
        "## 训练结果",
        "",
        f"- 最终训练轮次: {config['total_timesteps']}",
        f"- 训练设备: {training['device']}",
        f"- 是否使用 GPU: {'是' if training['gpu_used'] else '否'}",
        f"- 最新模型路径: {training['model_path']}",
        f"- 最优模型路径: {training['best_model_path']}",
        f"- 训练指标文件: {context['output_paths']['metrics'] / 'ppo_train_metrics.csv'}",
        f"- 评估指标文件: {context['output_paths']['metrics'] / 'ppo_eval_metrics.csv'}",
        f"- 奖励强基准: {config['reward']['strong_baseline']}",
        f"- 中长期价格口径: {config['reporting']['lt_price_note']}",
        f"- 异常与警告记录: 详见 outputs/logs/train.log",
        "",
    ]
    return "\n".join(lines)


def run_train(context: dict[str, Any]) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="train")
    logger.info("开始执行训练模块。")
    training = train_model(
        bundle=context["bundle"],
        train_weeks=context["train_sequence"],
        val_weeks=context["split"].val,
        config=context["config"],
        output_paths=context["output_paths"],
        run_name="ppo",
    )
    summary = build_train_summary(context, training)
    save_markdown(summary, context["output_paths"]["reports"] / "train_summary.md")
    logger.info("训练模块执行完成。")
    return training


def main() -> dict[str, Any]:
    context = prepare_project_context("/Users/dk/py/elec", logger_name="train")
    return run_train(context)


if __name__ == "__main__":
    main()
