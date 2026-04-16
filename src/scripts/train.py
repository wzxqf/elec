from pathlib import Path
from typing import Any

from src.agents.hybrid_pso import save_hybrid_pso_model, train_hybrid_pso_model
from src.scripts.common import prepare_project_context, split_to_dict, subset_bundle_for_weeks
from src.utils.io import save_markdown
from src.utils.logger import configure_logging
from src.utils.runtime_status import RuntimeStatusTracker


def build_train_summary(context: dict[str, Any], training: dict[str, Any]) -> str:
    split = split_to_dict(context["split"])
    runtime = training["runtime_profile"]
    return "\n".join(
        [
            "# 训练摘要",
            "",
            f"- 版本: {context['config']['version']}",
            f"- 算法: {context['config']['training']['algorithm']}",
            f"- 训练周: {', '.join(split['train'])}",
            f"- 验证周: {', '.join(split['val'])}",
            f"- 设备: {runtime['score_kernel_device']}",
            f"- 上层粒子数: {runtime['upper_particles']}",
            f"- 下层粒子数: {runtime['lower_particles']}",
            f"- 上层真实维度: {runtime.get('upper_dim', 'n/a')}",
            f"- 下层真实维度: {runtime.get('lower_dim', 'n/a')}",
            f"- 迭代轮数: {runtime['iterations']}",
            f"- 最优目标值: {training['model'].best_score:.4f}",
            f"- 模型路径: {context['output_paths']['models'] / 'hybrid_pso_model.json'}",
            f"- 训练轨迹: {context['output_paths']['metrics'] / 'hybrid_pso_training_trace.csv'}",
            f"- 参数布局摘要: {context['output_paths']['reports'] / 'parameter_layout_summary.md'}",
            "",
        ]
    )


def run_train(context: dict[str, Any]) -> dict[str, Any]:
    logger = configure_logging(context["output_paths"]["logs"], name="train")
    status_path = context.get("runtime_status_path", Path.cwd() / ".cache" / "train_runtime_status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_tracker = RuntimeStatusTracker(status_path)
    logger.info("开始执行 v0.36 训练模块。")
    train_bundle = subset_bundle_for_weeks(context["bundle"], context["split"].train)
    status_tracker.update(stage="训练", phase_name="Hybrid PSO 训练", phase_progress=0.0, total_progress=0.1, message="编译张量包")
    training_result = train_hybrid_pso_model(
        train_bundle["tensor_bundle"],
        context["config"],
        compiled_layout=train_bundle.get("compiled_parameter_layout"),
    )
    save_hybrid_pso_model(training_result.model, context["output_paths"]["models"] / "hybrid_pso_model.json")
    training_result.training_trace.to_csv(context["output_paths"]["metrics"] / "hybrid_pso_training_trace.csv", index=False)
    summary = build_train_summary(
        context,
        {
            "model": training_result.model,
            "runtime_profile": training_result.runtime_profile,
        },
    )
    save_markdown(summary, context["output_paths"]["reports"] / "train_summary.md")
    status_tracker.update(stage="训练", phase_name="Hybrid PSO 训练", phase_progress=1.0, total_progress=0.33, message="训练完成")
    logger.info("训练模块执行完成。")
    return {
        "model": training_result.model,
        "runtime_profile": training_result.runtime_profile,
        "training_trace": training_result.training_trace,
        "device": training_result.runtime_profile["score_kernel_device"],
        "gpu_used": str(training_result.runtime_profile["score_kernel_device"]).startswith("cuda"),
        "model_path": context["output_paths"]["models"] / "hybrid_pso_model.json",
    }


def main() -> dict[str, Any]:
    context = prepare_project_context(Path.cwd(), logger_name="train")
    return run_train(context)


if __name__ == "__main__":
    main()
