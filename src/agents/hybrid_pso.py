from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.model_layout.schema import CompiledParameterLayout
from src.training.score_kernel import batch_score_particles
from src.training.tensor_bundle import TrainingTensorBundle


class VectorList(list[float]):
    @property
    def shape(self) -> tuple[int]:
        return (len(self),)


@dataclass(frozen=True)
class HybridPSOModel:
    upper_best: VectorList
    lower_best: VectorList
    best_score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HybridPSOTrainResult:
    model: HybridPSOModel
    runtime_profile: dict[str, Any]
    training_trace: pd.DataFrame


def save_hybrid_pso_model(model: HybridPSOModel, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(model), ensure_ascii=False, indent=2), encoding="utf-8")


def load_hybrid_pso_model(path: str | Path) -> HybridPSOModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HybridPSOModel(
        upper_best=VectorList(payload["upper_best"]),
        lower_best=VectorList(payload["lower_best"]),
        best_score=float(payload["best_score"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _resolve_device(config: dict[str, Any]) -> str:
    requested = str(config.get("training", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _randn(generator: torch.Generator, shape: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


def _infer_release_version(config: dict[str, Any]) -> str:
    project = config.get("project", {})
    project_version = project.get("version")
    if project_version:
        return str(project_version)

    algorithm = str(config.get("training", {}).get("algorithm", "HYBRID_PSO_V040")).upper()
    match = re.search(r"_V(\d+)$", algorithm)
    if match:
        normalized = match.group(1).lstrip("0") or "0"
        normalized = normalized.rstrip("0") or "0"
        return f"v0.{normalized}"
    return "v0.45"


def _resolve_optimizer(hybrid_cfg: dict[str, Any]) -> dict[str, float]:
    optimizer = dict(hybrid_cfg.get("optimizer", {}))
    return {
        "init_scale": float(optimizer.get("init_scale", 0.10)),
        "inertia": float(optimizer.get("inertia", 0.65)),
        "cognitive": float(optimizer.get("cognitive", 1.35)),
        "social": float(optimizer.get("social", 1.35)),
        "position_clip_abs": float(optimizer.get("position_clip_abs", 1.0)),
    }


def train_hybrid_pso_model(
    tensor_bundle: TrainingTensorBundle,
    config: dict[str, Any],
    compiled_layout: CompiledParameterLayout | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> HybridPSOTrainResult:
    hybrid_cfg = config["hybrid_pso"]
    device = _resolve_device(config)
    seed = int(hybrid_cfg.get("seed", 42))
    upper_cfg = hybrid_cfg["upper"]
    lower_cfg = hybrid_cfg["lower"]
    upper_particles = int(upper_cfg["particles"])
    lower_particles = int(lower_cfg["particles"])
    iterations = int(max(upper_cfg["iterations"], lower_cfg["iterations"]))
    layout = compiled_layout
    if layout is None:
        upper_dim = int(upper_cfg["dimension"])
        lower_dim = int(lower_cfg["dimension"])
    else:
        upper_dim = int(layout.upper.total_dimension)
        lower_dim = int(layout.lower.total_dimension)
    optimizer = _resolve_optimizer(hybrid_cfg)

    generator = torch.Generator(device=device if device != "cpu" else "cpu")
    generator.manual_seed(seed)
    upper = optimizer["init_scale"] * _randn(generator, (upper_particles, upper_dim), device)
    lower = optimizer["init_scale"] * _randn(generator, (lower_particles, lower_dim), device)
    upper_velocity = torch.zeros_like(upper)
    lower_velocity = torch.zeros_like(lower)
    upper_best = upper.clone()
    lower_best = lower.clone()
    upper_best_score = torch.full((upper_particles,), float("inf"), device=device)
    lower_best_score = torch.full((lower_particles,), float("inf"), device=device)
    global_best_score = float("inf")
    global_upper_best = upper[0].clone()
    global_lower_best = lower[0].clone()

    trace_rows: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        scored = batch_score_particles(tensor_bundle, upper, lower, device=device, config=config, compiled_layout=compiled_layout)
        total_score = scored.total_score
        upper_score = total_score.min(dim=1).values
        lower_score = total_score.min(dim=0).values
        improved_upper = upper_score < upper_best_score
        improved_lower = lower_score < lower_best_score
        upper_best = torch.where(improved_upper.unsqueeze(1), upper, upper_best)
        lower_best = torch.where(improved_lower.unsqueeze(1), lower, lower_best)
        upper_best_score = torch.minimum(upper_best_score, upper_score)
        lower_best_score = torch.minimum(lower_best_score, lower_score)

        best_flat_index = int(torch.argmin(total_score).item())
        best_upper_index = best_flat_index // lower_particles
        best_lower_index = best_flat_index % lower_particles
        iteration_best_score = float(total_score[best_upper_index, best_lower_index].item())
        if iteration_best_score < global_best_score:
            global_best_score = iteration_best_score
            global_upper_best = upper[best_upper_index].clone()
            global_lower_best = lower[best_lower_index].clone()

        trace_rows.append(
            {
                "iteration": iteration,
                "best_score": global_best_score,
                "mean_score": float(total_score.mean().item()),
                "upper_particles": upper_particles,
                "lower_particles": lower_particles,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "iteration": iteration,
                    "iterations": iterations,
                    "phase_progress": iteration / iterations,
                    "best_score": global_best_score,
                    "mean_score": float(total_score.mean().item()),
                }
            )

        rand_upper_1 = torch.rand_like(upper)
        rand_upper_2 = torch.rand_like(upper)
        rand_lower_1 = torch.rand_like(lower)
        rand_lower_2 = torch.rand_like(lower)
        upper_velocity = (
            optimizer["inertia"] * upper_velocity
            + optimizer["cognitive"] * rand_upper_1 * (upper_best - upper)
            + optimizer["social"] * rand_upper_2 * (global_upper_best.unsqueeze(0) - upper)
        )
        lower_velocity = (
            optimizer["inertia"] * lower_velocity
            + optimizer["cognitive"] * rand_lower_1 * (lower_best - lower)
            + optimizer["social"] * rand_lower_2 * (global_lower_best.unsqueeze(0) - lower)
        )
        upper = torch.clamp(upper + upper_velocity, -optimizer["position_clip_abs"], optimizer["position_clip_abs"])
        lower = torch.clamp(lower + lower_velocity, -optimizer["position_clip_abs"], optimizer["position_clip_abs"])

    model = HybridPSOModel(
        upper_best=VectorList(global_upper_best.detach().cpu().tolist()),
        lower_best=VectorList(global_lower_best.detach().cpu().tolist()),
        best_score=global_best_score,
        metadata={
            "version": _infer_release_version(config),
            "algorithm": str(config.get("training", {}).get("algorithm", "HYBRID_PSO_V040")),
            "score_kernel_device": device,
        },
    )
    return HybridPSOTrainResult(
        model=model,
        runtime_profile={
            "optimizer_device": device,
            "score_kernel_device": device,
            "upper_particles": upper_particles,
            "lower_particles": lower_particles,
            "iterations": iterations,
            "upper_dim": upper_dim,
            "lower_dim": lower_dim,
            "upper_dim_real": upper_dim,
            "lower_dim_real": lower_dim,
            "init_scale": optimizer["init_scale"],
            "inertia": optimizer["inertia"],
            "cognitive": optimizer["cognitive"],
            "social": optimizer["social"],
            "position_clip_abs": optimizer["position_clip_abs"],
        },
        training_trace=pd.DataFrame(trace_rows),
    )
