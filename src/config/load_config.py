from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


SUPPORTED_HYBRID_PSO_ALGORITHMS = {
    "HYBRID_PSO_V033",
    "HYBRID_PSO_V036",
    "HYBRID_PSO_V038",
    "HYBRID_PSO_V040",
}


REQUIRED_SECTIONS = [
    "project",
    "data",
    "outputs",
    "split",
    "rolling_validation",
    "rolling_retrain",
    "feature_selection",
    "reward",
    "score_kernel",
    "training",
    "policy_regime",
    "reporting",
    "analysis",
    "policy_deep",
    "policy_projection",
    "upper_strategy",
    "lower_strategy",
    "economics",
    "hybrid_pso",
    "parameter_compiler",
]


def _require_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"experiment_config.yaml 缺少必需配置节: {key}")
    return value


def _require_keys(section_name: str, section: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise KeyError(f"experiment_config.yaml 的 {section_name} 缺少字段: {', '.join(missing)}")


def _resolve_project_path(project_root: Path, value: Any) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def _normalize_project_paths(
    project_root: Path,
    project: dict[str, Any],
    data: dict[str, Any],
    outputs: dict[str, Any],
) -> None:
    project["project_root"] = _resolve_project_path(project_root, project["project_root"])
    data["policy_directory"] = _resolve_project_path(project_root, data["policy_directory"])
    data["data_candidates"] = [_resolve_project_path(project_root, candidate) for candidate in data["data_candidates"]]
    if "root" in outputs:
        outputs["root"] = _resolve_project_path(project_root, outputs["root"])
    else:
        for key, value in list(outputs.items()):
            outputs[key] = _resolve_project_path(project_root, value)


def _normalize_hybrid_pso(hybrid_pso: dict[str, Any]) -> None:
    optimizer = dict(hybrid_pso.get("optimizer", {}))
    hybrid_pso["optimizer"] = {
        "init_scale": float(optimizer.get("init_scale", 0.10)),
        "inertia": float(optimizer.get("inertia", 0.65)),
        "cognitive": float(optimizer.get("cognitive", 1.35)),
        "social": float(optimizer.get("social", 1.35)),
        "position_clip_abs": float(optimizer.get("position_clip_abs", 1.0)),
    }


def _normalize_score_kernel(score_kernel: dict[str, Any]) -> None:
    hourly_signal = dict(score_kernel.get("hourly_signal", {}))
    session_weights = dict(score_kernel.get("session_weights", {}))
    hourly_limit = dict(score_kernel.get("hourly_limit", {}))
    score_kernel.update(
        {
            "contract_adjustment_scale_ratio": float(score_kernel.get("contract_adjustment_scale_ratio", 0.30)),
            "contract_adjustment_feature_scale": float(score_kernel.get("contract_adjustment_feature_scale", 0.15)),
            "contract_adjustment_policy_scale": float(score_kernel.get("contract_adjustment_policy_scale", 0.05)),
            "exposure_band_base_ratio": float(score_kernel.get("exposure_band_base_ratio", 0.20)),
            "exposure_band_feature_scale": float(score_kernel.get("exposure_band_feature_scale", 0.10)),
            "contract_position_base_ratio": float(score_kernel.get("contract_position_base_ratio", 0.60)),
            "baseline_position_ratio": float(score_kernel.get("baseline_position_ratio", 0.55)),
            "baseline_projection_penalty_scale": float(score_kernel.get("baseline_projection_penalty_scale", 0.05)),
            "lt_settlement_weight": float(score_kernel.get("lt_settlement_weight", 0.60)),
            "da_settlement_weight": float(score_kernel.get("da_settlement_weight", 0.40)),
            "hourly_signal": {
                "spread_weight": float(hourly_signal.get("spread_weight", 0.02)),
                "load_dev_weight": float(hourly_signal.get("load_dev_weight", 0.01)),
                "renewable_weight": float(hourly_signal.get("renewable_weight", 0.01)),
                "spread_abs_weight": float(hourly_signal.get("spread_abs_weight", 0.005)),
                "renewable_abs_weight": float(hourly_signal.get("renewable_abs_weight", 0.004)),
            },
            "session_weights": {
                "business_hour": float(session_weights.get("business_hour", 0.50)),
                "peak_hour": float(session_weights.get("peak_hour", 0.30)),
                "valley_hour": float(session_weights.get("valley_hour", -0.20)),
                "renewable_valley_mix": float(session_weights.get("renewable_valley_mix", 0.50)),
                "renewable_business_mix": float(session_weights.get("renewable_business_mix", 0.50)),
            },
            "hourly_limit": {
                "base_multiplier": float(hourly_limit.get("base_multiplier", 0.50)),
                "shrink_multiplier": float(hourly_limit.get("shrink_multiplier", 0.50)),
            },
        }
    )


def load_runtime_config(project_root: str | Path, filename: str = "experiment_config.yaml") -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config_path = project_root / filename
    if not config_path.exists():
        raise FileNotFoundError(f"未找到根目录独立参数文件: {config_path}")

    root_config = load_yaml(config_path)
    for section_name in REQUIRED_SECTIONS:
        _require_section(root_config, section_name)

    project = dict(_require_section(root_config, "project"))
    data = dict(_require_section(root_config, "data"))
    outputs = dict(_require_section(root_config, "outputs"))
    split = dict(_require_section(root_config, "split"))
    rolling_validation = dict(_require_section(root_config, "rolling_validation"))
    rolling_retrain = dict(_require_section(root_config, "rolling_retrain"))
    feature_selection = dict(_require_section(root_config, "feature_selection"))
    reward = dict(_require_section(root_config, "reward"))
    score_kernel = dict(_require_section(root_config, "score_kernel"))
    training = dict(_require_section(root_config, "training"))
    policy_regime = dict(_require_section(root_config, "policy_regime"))
    reporting = dict(_require_section(root_config, "reporting"))
    analysis = dict(_require_section(root_config, "analysis"))
    hybrid_pso = dict(_require_section(root_config, "hybrid_pso"))
    parameter_compiler = dict(_require_section(root_config, "parameter_compiler"))
    policy_deep = dict(_require_section(root_config, "policy_deep"))
    policy_projection = dict(_require_section(root_config, "policy_projection"))
    upper_strategy = dict(_require_section(root_config, "upper_strategy"))
    lower_strategy = dict(_require_section(root_config, "lower_strategy"))
    economics = dict(_require_section(root_config, "economics"))

    _require_keys("project", project, ["version", "project_root"])
    _require_keys("data", data, ["sample_start", "sample_end", "buffer_end", "policy_directory", "data_candidates"])
    _normalize_project_paths(project_root, project, data, outputs)
    algorithm = str(training.get("algorithm", "")).upper()
    if algorithm in SUPPORTED_HYBRID_PSO_ALGORITHMS:
        _require_keys("training", training, ["algorithm", "seed", "device", "allow_cpu"])
        _require_keys("hybrid_pso", hybrid_pso, ["seed", "upper", "lower"])
        _normalize_hybrid_pso(hybrid_pso)
        _normalize_score_kernel(score_kernel)
    else:
        supported = " / ".join(sorted(SUPPORTED_HYBRID_PSO_ALGORITHMS))
        raise ValueError(f"不支持的 training.algorithm: {algorithm}。当前仅支持 {supported}。")
    _require_keys("reward", reward, ["baseline_strategy", "cvar_alpha", "lambda_tail", "lambda_hedge", "lambda_trade", "lambda_violate"])
    _require_keys("feature_selection", feature_selection, ["enabled", "feature_include_for_agent", "feature_exclude_for_agent", "feature_keep_for_report_only"])
    _require_keys("policy_projection", policy_projection, ["mode", "clip_method", "violation_penalty_scale"])
    _require_keys("upper_strategy", upper_strategy, ["contract_curve_hours", "feature_columns", "parameter_layout"])
    _require_keys("lower_strategy", lower_strategy, ["feature_columns", "parameter_layout"])
    _require_keys("economics", economics, ["retail_tariff_yuan_per_mwh", "imbalance_penalty_multiplier", "adjustment_cost_yuan_per_mwh", "friction_cost_yuan_per_mwh"])

    runtime = {
        "config_path": str(config_path.resolve()),
        "raw_experiment_config": root_config,
        "project": project,
        "version": project["version"],
        "project_root": project["project_root"],
        "sample_start": data["sample_start"],
        "sample_end": data["sample_end"],
        "buffer_end": data["buffer_end"],
        "policy_directory": data["policy_directory"],
        "data_candidates": data["data_candidates"],
        "outputs": outputs,
        "split": split,
        "rolling_validation": rolling_validation,
        "rolling_retrain": rolling_retrain,
        "feature_selection": feature_selection,
        "reward": reward,
        "score_kernel": score_kernel,
        "training": training,
        "policy_regime": policy_regime,
        "reporting": reporting,
        "analysis": analysis,
        "hybrid_pso": hybrid_pso,
        "parameter_compiler": parameter_compiler,
        "policy_deep": policy_deep,
        "policy_projection": policy_projection,
        "upper_strategy": upper_strategy,
        "lower_strategy": lower_strategy,
        "economics": economics,
    }

    for optional_section in [
        "cost",
        "constraints",
        "env",
        "rules",
        "scenario",
        "benchmarks",
        "sensitivity",
        "robustness",
        "search",
        "hpso",
        "policy",
        "analysis_v035",
        "policy_feasible_domain",
    ]:
        value = root_config.get(optional_section)
        if isinstance(value, dict):
            runtime[optional_section] = value

    runtime.update(data)
    runtime.update(training)
    runtime["algorithm"] = algorithm
    runtime["seed"] = training["seed"]
    runtime["device"] = training["device"]
    runtime["rules"] = root_config.get("rules", {})
    return runtime
