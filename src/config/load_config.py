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
    "cost",
    "constraints",
    "reward",
    "env",
    "rules",
    "training",
    "scenario",
    "benchmarks",
    "feature_selection",
    "policy_regime",
    "reporting",
    "analysis",
    "sensitivity",
    "robustness",
    "search",
    "hpso",
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
    cost = _require_section(root_config, "cost")
    constraints = _require_section(root_config, "constraints")
    reward = _require_section(root_config, "reward")
    env = _require_section(root_config, "env")
    rules = _require_section(root_config, "rules")
    training = _require_section(root_config, "training")
    scenario = _require_section(root_config, "scenario")
    benchmarks = _require_section(root_config, "benchmarks")
    feature_selection = _require_section(root_config, "feature_selection")
    policy_regime = _require_section(root_config, "policy_regime")
    reporting = _require_section(root_config, "reporting")
    analysis = _require_section(root_config, "analysis")
    sensitivity = _require_section(root_config, "sensitivity")
    robustness = _require_section(root_config, "robustness")
    search = _require_section(root_config, "search")
    hpso = _require_section(root_config, "hpso")

    _require_keys("project", project, ["version", "project_root"])
    _require_keys("data", data, ["sample_start", "sample_end", "buffer_end", "policy_directory", "data_candidates"])
    _require_keys("training", training, ["policy", "total_timesteps", "eval_freq", "checkpoint_freq", "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "seed", "device", "use_vec_normalize"])
    _require_keys("constraints", constraints, ["lock_ratio_min", "lock_ratio_max", "delta_h_max", "delta_lock_cap"])
    _require_keys("feature_selection", feature_selection, ["enabled", "feature_include_for_agent", "feature_exclude_for_agent", "feature_keep_for_report_only"])
    _require_keys("hpso", hpso, ["device", "allow_cpu", "seed", "upper", "lower", "objective_weights"])

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
        "cost": cost,
        "constraints": constraints,
        "reward": reward,
        "env": env,
        "rules": {"rules": rules, **rules},
        "rules_only": rules,
        "training": training,
        "scenario": scenario,
        "benchmarks": benchmarks,
        "feature_selection": feature_selection,
        "policy_regime": policy_regime,
        "reporting": reporting,
        "analysis": analysis,
        "sensitivity": sensitivity,
        "robustness": robustness,
        "search": search,
        "hpso": hpso,
    }

    runtime.update(data)
    runtime.update(training)
    runtime["rules"] = rules
    return runtime
