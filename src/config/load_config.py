from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


REQUIRED_SECTIONS = [
    "project",
    "data",
    "outputs",
    "split",
    "rolling_validation",
    "rolling_retrain",
    "feature_selection",
    "reward",
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


def load_runtime_config(project_root: str | Path, filename: str = "experiment_config.yaml") -> dict[str, Any]:
    project_root = Path(project_root)
    config_path = project_root / filename
    if not config_path.exists():
        raise FileNotFoundError(f"未找到根目录独立参数文件: {config_path}")

    root_config = load_yaml(config_path)
    for section_name in REQUIRED_SECTIONS:
        _require_section(root_config, section_name)

    project = _require_section(root_config, "project")
    data = _require_section(root_config, "data")
    outputs = _require_section(root_config, "outputs")
    split = _require_section(root_config, "split")
    rolling_validation = _require_section(root_config, "rolling_validation")
    rolling_retrain = _require_section(root_config, "rolling_retrain")
    feature_selection = _require_section(root_config, "feature_selection")
    reward = _require_section(root_config, "reward")
    training = _require_section(root_config, "training")
    policy_regime = _require_section(root_config, "policy_regime")
    reporting = _require_section(root_config, "reporting")
    analysis = _require_section(root_config, "analysis")
    hybrid_pso = _require_section(root_config, "hybrid_pso")
    policy_deep = _require_section(root_config, "policy_deep")
    policy_projection = _require_section(root_config, "policy_projection")
    upper_strategy = _require_section(root_config, "upper_strategy")
    lower_strategy = _require_section(root_config, "lower_strategy")
    economics = _require_section(root_config, "economics")

    _require_keys("project", project, ["version", "project_root"])
    _require_keys("data", data, ["sample_start", "sample_end", "buffer_end", "policy_directory", "data_candidates"])
    algorithm = str(training.get("algorithm", "")).upper()
    if algorithm == "HYBRID_PSO_V033":
        _require_keys("training", training, ["algorithm", "seed", "device", "allow_cpu"])
        _require_keys("hybrid_pso", hybrid_pso, ["seed", "upper", "lower"])
    else:
        raise ValueError(f"不支持的 training.algorithm: {algorithm}。v0.33 仅支持 HYBRID_PSO_V033。")
    _require_keys("reward", reward, ["baseline_strategy", "cvar_alpha", "lambda_tail", "lambda_hedge", "lambda_trade", "lambda_violate"])
    _require_keys("feature_selection", feature_selection, ["enabled", "feature_include_for_agent", "feature_exclude_for_agent", "feature_keep_for_report_only"])
    _require_keys("policy_projection", policy_projection, ["mode", "clip_method", "violation_penalty_scale"])
    _require_keys("upper_strategy", upper_strategy, ["contract_curve_hours", "feature_columns", "parameter_layout"])
    _require_keys("lower_strategy", lower_strategy, ["feature_columns", "parameter_layout"])
    _require_keys("economics", economics, ["retail_tariff_yuan_per_mwh", "imbalance_penalty_multiplier", "adjustment_cost_yuan_per_mwh", "friction_cost_yuan_per_mwh"])

    runtime = {
        "config_path": str(config_path),
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
        "training": training,
        "policy_regime": policy_regime,
        "reporting": reporting,
        "analysis": analysis,
        "hybrid_pso": hybrid_pso,
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
