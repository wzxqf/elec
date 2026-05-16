from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from src.analysis.hourly_spot_experiments import (
    HourlySpotExperimentCandidate,
    build_hourly_spot_experiment_grid,
    derive_hourly_spot_baseline,
    summarize_hourly_spot_guardrails,
)
from src.scripts.common import prepare_project_context, subset_bundle_for_weeks
from src.training.score_kernel import batch_score_particles


def _deepcopy_config(config: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _candidate_config(base_config: dict[str, Any], candidate: HourlySpotExperimentCandidate) -> dict[str, Any]:
    config = _deepcopy_config(base_config)
    _set_nested(config, ("score_kernel", "hourly_signal", "transform"), candidate.signal_transform)
    _set_nested(config, ("score_kernel", "hourly_signal", "signal_clip_abs"), candidate.signal_clip_abs)
    _set_nested(config, ("score_kernel", "hourly_gate", "enabled"), True)
    _set_nested(config, ("score_kernel", "hourly_gate", "mode"), candidate.gate_mode)
    _set_nested(config, ("score_kernel", "hourly_gate", "signal_deadband"), candidate.signal_deadband)
    _set_nested(config, ("score_kernel", "hourly_gate", "temperature"), candidate.gate_temperature)
    _set_nested(config, ("score_kernel", "hourly_limit", "base_multiplier"), candidate.hourly_limit_base_multiplier)
    _set_nested(config, ("score_kernel", "hourly_limit", "shrink_multiplier"), candidate.hourly_limit_shrink_multiplier)
    _set_nested(config, ("economics", "friction_cost_yuan_per_mwh"), candidate.friction_cost_yuan_per_mwh)
    _set_nested(config, ("reward", "lambda_trade"), candidate.lambda_trade)
    return config


def _parse_particle(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("particle snapshot must be a list")
    return [float(item) for item in parsed]


def _write_isolated_config(project_root: Path, base_config_path: Path, output_root: Path) -> Path:
    raw_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    raw_config.setdefault("outputs", {})["root"] = str(output_root.as_posix())
    raw_config.setdefault("hourly_spot_experiment", {})["enabled"] = True
    target = output_root / "context_config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return target.resolve()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    text_frame = frame.copy()
    for column in text_frame.columns:
        text_frame[column] = text_frame[column].map(lambda value: f"{value:.6g}" if isinstance(value, float) else str(value))
    columns = list(text_frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in text_frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _evaluate_candidate(
    *,
    context: dict[str, Any],
    snapshots: pd.DataFrame,
    candidate: HourlySpotExperimentCandidate,
) -> dict[str, Any]:
    config = _candidate_config(context["config"], candidate)
    weekly_profit = 0.0
    weekly_baseline = 0.0
    weekly_excess = 0.0
    cvar_values: list[float] = []
    hedge_values: list[float] = []
    spot_abs_sum = 0.0
    nonzero_hours = 0
    valid_hours = 0
    friction_sum = 0.0

    window_lookup = {window.window_name: window for window in context["rolling_plan"]}
    for row in snapshots.itertuples(index=False):
        window = window_lookup[str(row.window_name)]
        test_bundle = subset_bundle_for_weeks(context["bundle"], window.test_weeks)
        tensor_bundle = test_bundle["tensor_bundle"]
        upper = torch.as_tensor(_parse_particle(str(row.upper_best)), dtype=torch.float32, device=tensor_bundle.weekly_feature_tensor.device).view(1, -1)
        lower = torch.as_tensor(_parse_particle(str(row.lower_best)), dtype=torch.float32, device=tensor_bundle.weekly_feature_tensor.device).view(1, -1)
        scored = batch_score_particles(
            tensor_bundle=tensor_bundle,
            upper_particles=upper,
            lower_particles=lower,
            device=tensor_bundle.device,
            config=config,
            compiled_layout=test_bundle.get("compiled_parameter_layout"),
        )
        weekly_profit += float(scored.weekly_profit.sum().detach().cpu().item())
        weekly_baseline += float(scored.weekly_profit_baseline.sum().detach().cpu().item())
        weekly_excess += float(scored.weekly_excess_profit.sum().detach().cpu().item())
        cvar_values.extend(float(value) for value in scored.weekly_cvar99.flatten().detach().cpu().tolist())
        hedge_values.extend(float(value) for value in scored.weekly_hedge_error.flatten().detach().cpu().tolist())
        spot = scored.spot_hedge_mwh.detach().cpu()
        mask = tensor_bundle.hourly_valid_mask.detach().cpu()
        spot_abs_sum += float(spot.abs().sum().item())
        nonzero_hours += int((spot.abs() > 1.0e-6).sum().item())
        valid_hours += int(mask.sum().item())
        friction_sum += float(scored.weekly_friction_cost.sum().detach().cpu().item())

    row = {
        **asdict(candidate),
        "total_profit_w": weekly_profit,
        "sum_profit_baseline_w": weekly_baseline,
        "sum_excess_profit_w": weekly_excess,
        "mean_cvar99_w": float(pd.Series(cvar_values, dtype="float64").mean() if cvar_values else 0.0),
        "mean_hedge_error_w": float(pd.Series(hedge_values, dtype="float64").mean() if hedge_values else 0.0),
        "nonzero_hour_share": float(nonzero_hours / max(valid_hours, 1)),
        "spot_abs_sum_mwh": spot_abs_sum,
        "friction_cost_sum": friction_sum,
    }
    return row


def _evaluate_candidates(
    *,
    context: dict[str, Any],
    snapshots: pd.DataFrame,
    candidates: list[HourlySpotExperimentCandidate],
    workers: int,
) -> list[dict[str, Any]]:
    worker_count = max(int(workers), 1)
    if worker_count == 1:
        return [_evaluate_candidate(context=context, snapshots=snapshots, candidate=candidate) for candidate in candidates]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_evaluate_candidate, context=context, snapshots=snapshots, candidate=candidate)
            for candidate in candidates
        ]
        return [future.result() for future in futures]


def run_search(project_root: Path, *, output_root: Path, top: int, workers: int = 1) -> Path:
    config_path = project_root / "experiment_config.yaml"
    isolated_config = _write_isolated_config(project_root, config_path, output_root)
    context = prepare_project_context(project_root, logger_name="hourly_spot_param_search", config_path=isolated_config)
    version = str(context["config"]["version"])
    snapshots_path = project_root / "outputs" / version / "raw" / "metrics" / "rolling_parameter_snapshots.csv"
    snapshots = pd.read_csv(snapshots_path)
    candidates = build_hourly_spot_experiment_grid(context["config"])
    guardrails = context["config"].get("hourly_spot_experiment", {}).get("guardrails", {})
    rows = _evaluate_candidates(context=context, snapshots=snapshots, candidates=candidates, workers=workers)
    result_frame = pd.DataFrame(rows)
    baseline = derive_hourly_spot_baseline(
        result_frame,
        config=context["config"],
        cvar_tolerance=float(guardrails.get("cvar99_multiplier_max", 1.03)),
    )
    result = summarize_hourly_spot_guardrails(result_frame, baseline=baseline)
    run_id = datetime.now(timezone.utc).strftime("param-search-%Y%m%dT%H%M%SZ")
    result_dir = output_root / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "hourly_spot_param_search.csv"
    result.to_csv(result_path, index=False, encoding="utf-8")
    summary_path = result_dir / "hourly_spot_param_search_summary.md"
    top_frame = result.head(top)
    summary_lines = [
        "# 小时级现货参数扫描",
        "",
        f"- candidate_count: {len(result)}",
        f"- workers: {max(int(workers), 1)}",
        f"- baseline_sum_excess_profit_w: {baseline['sum_excess_profit_w']:.6f}",
        f"- baseline_mean_cvar99_w: {baseline['mean_cvar99_w']:.6f}",
        f"- baseline_mean_hedge_error_w: {baseline['mean_hedge_error_w']:.6f}",
        "",
        _markdown_table(top_frame),
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(result_path)
    print(summary_path)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search hourly spot hedge parameters using saved rolling particles.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--output-root", default=".cache/param_search_outputs", help="Isolated output root.")
    parser.add_argument("--top", type=int, default=12, help="Rows to include in the markdown summary.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count for candidate evaluation.")
    args = parser.parse_args()
    run_search(Path(args.project_root).resolve(), output_root=Path(args.output_root).resolve(), top=args.top, workers=args.workers)


if __name__ == "__main__":
    main()
