from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.hpso_param_policy import train_hpso_param_policy
from src.agents.hpso import train_hpso_model
from src.scripts.common import prepare_project_context, split_to_dict
from src.utils.io import save_markdown
from src.utils.logger import configure_logging
from src.utils.runtime_status import RuntimeStatusTracker


def build_train_summary(context: dict[str, Any], training: dict[str, Any]) -> str:
    config = context["config"]
    agent_feature_columns = context["bundle"].get("agent_feature_columns", [])
    algorithm = str(config["training"].get("algorithm", "PPO")).upper()
    runtime_profile = training.get("runtime_profile", {})
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
        f"- 根参数文件: {config['config_path']}",
        f"- 主算法: {algorithm}",
        f"- 实际特征数: {len(agent_feature_columns)}",
        "",
        "## 超参数",
        "",
    ]
    if algorithm == "HPSO_PARAM_POLICY":
        hpso_cfg = config["hpso"]
        lines.extend(
            [
                f"- HPSO 参数维度: {hpso_cfg['parameter_dimension']}",
                f"- 粒子数/迭代数: {hpso_cfg['swarm']['particles']} / {hpso_cfg['swarm']['iterations']}",
                f"- bootstrap 序列长度/块大小: {hpso_cfg['bootstrap']['sequence_length']} / {hpso_cfg['bootstrap']['block_size']}",
                "- 验证与回测语义: 加载训练后 theta 推断，不重新优化目标周",
                "- 下层修正语义: 因果滚动修正",
            ]
        )
    elif algorithm == "HPSO":
        hpso_cfg = config["hpso"]
        lines.extend(
            [
                f"- 上层粒子数/迭代数: {hpso_cfg['upper']['particles']} / {hpso_cfg['upper']['iterations']}",
                f"- 下层粒子数/迭代数: {hpso_cfg['lower']['particles']} / {hpso_cfg['lower']['iterations']}",
                f"- 退火初温/降温率: {hpso_cfg['upper']['initial_temperature']} / {hpso_cfg['upper']['cooling_rate']}",
                f"- 上层/下层 BP 精修步数: {hpso_cfg['upper'].get('backprop_steps', 0)} / {hpso_cfg['lower'].get('backprop_steps', 0)}",
                "- 合约与现货合约硬限制: 已取消，边界字段仅作诊断与初始搜索尺度",
                f"- CPU 降级允许: {'是' if hpso_cfg.get('allow_cpu', True) else '否'}",
            ]
        )
    else:
        lines.extend(
            [
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
            ]
        )
    lines.extend(
        [
            "",
            "## 训练结果",
            "",
            f"- 最终训练轮次: {config['hpso']['swarm']['iterations'] if algorithm == 'HPSO_PARAM_POLICY' else (config['hpso']['upper']['iterations'] if algorithm == 'HPSO' else config['total_timesteps'])}",
            f"- 训练设备: {training['device']}",
            f"- 主算设备: {runtime_profile.get('rollout_compute_device', training['device'])}",
            f"- 并行 worker: {runtime_profile.get('parallel_workers', 1)}",
            f"- 是否使用 GPU: {'是' if training['gpu_used'] else '否'}",
            f"- 周度动作语义: {'HPSO 参数化策略 theta 推断底仓残差 + 边际敞口带宽 + 24小时曲线' if algorithm == 'HPSO_PARAM_POLICY' else ('HPSO 搜索不设硬限的中长期合约调整量 + 诊断性边际敞口带宽' if algorithm == 'HPSO' else '基准底仓残差 + 边际敞口带宽')}",
            f"- 最新模型路径: {training['model_path'] or 'HPSO 无模型文件'}",
            f"- 最优模型路径: {training['best_model_path'] or 'HPSO 无模型文件'}",
            f"- 训练指标文件: {context['output_paths']['metrics'] / ('hpso_weekly_practice_data.csv' if algorithm in {'HPSO', 'HPSO_PARAM_POLICY'} else 'ppo_train_metrics.csv')}",
            f"- 评估指标文件: {context['output_paths']['metrics'] / ('hpso_convergence_curve.csv' if algorithm in {'HPSO', 'HPSO_PARAM_POLICY'} else 'ppo_eval_metrics.csv')}",
            f"- 奖励强基准: {config['reward']['strong_baseline']}",
            f"- 中长期价格口径: {config['reporting']['lt_price_note']}",
            f"- 滚动验证窗口数: {len(context['rolling_validation_windows'])}",
            f"- 特征清单: {context['output_paths']['reports'] / 'feature_manifest.json'}",
            f"- 异常与警告记录: 详见 {context['output_paths']['logs'] / 'train.log'}",
            "",
        ]
    )
    return "\n".join(lines)


def run_train(context: dict[str, Any]) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="train")
    logger.info("开始执行训练模块。")
    algorithm = str(context["config"]["training"].get("algorithm", "PPO")).upper()
    status_tracker = RuntimeStatusTracker(context["runtime_status_path"])
    status_tracker.update(stage="训练", phase_name="训练模块", phase_progress=0.0, total_progress=0.0, message="准备训练数据")
    if algorithm == "HPSO_PARAM_POLICY":
        training = train_hpso_param_policy(
            bundle=context["bundle"],
            train_weeks=context["train_sequence"],
            config=context["config"],
            output_paths=context["output_paths"],
            run_name="hpso_param_policy",
            status_tracker=status_tracker,
        )
    elif algorithm == "HPSO":
        training = train_hpso_model(
            bundle=context["bundle"],
            train_weeks=context["train_sequence"],
            val_weeks=context["split"].val,
            config=context["config"],
            output_paths=context["output_paths"],
            run_name="hpso",
        )
    else:
        from src.agents.train_ppo import train_model

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
    status_tracker.update(stage="训练", phase_name="训练模块", phase_progress=1.0, total_progress=1.0 / 3.0, message="训练完成")
    logger.info("训练模块执行完成。")
    return training


def main() -> dict[str, Any]:
    context = prepare_project_context(Path.cwd(), logger_name="train")
    return run_train(context)


if __name__ == "__main__":
    main()
