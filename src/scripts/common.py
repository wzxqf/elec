from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.benchmarks import build_benchmark_actions
from src.backtest.simulator import simulate_strategy
from src.data.loader import load_raw_total_csv, locate_total_csv
from src.data.preprocess import build_data_quality_markdown, clean_total_data
from src.data.scenario_generator import WeekSplit, build_bootstrap_sequence, build_week_split
from src.data.weekly_builder import build_weekly_bundle
from src.policy.policy_parser import parse_policy_environment
from src.policy.policy_regime import build_policy_state_trace
from src.policy.policy_tables import build_policy_rule_summary_markdown
from src.utils.io import dump_yaml, load_yaml, merge_configs, resolve_output_paths, save_markdown
from src.utils.logger import configure_logging
from src.utils.seeds import set_global_seed


def _robust_summary(series: pd.Series, eps: float) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"median": 0.0, "iqr": max(eps, 1.0)}
    return {
        "median": float(clean.median()),
        "iqr": float(max(clean.quantile(0.75) - clean.quantile(0.25), eps)),
    }


def load_project_config(project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root)
    default = load_yaml(project_root / "configs" / "default.yaml")
    data = load_yaml(project_root / "configs" / "data.yaml")
    ppo = load_yaml(project_root / "configs" / "ppo.yaml")
    rules = load_yaml(project_root / "configs" / "rules.yaml")
    backtest = load_yaml(project_root / "configs" / "backtest.yaml")
    analysis = load_yaml(project_root / "configs" / "analysis.yaml")
    return merge_configs(default, data, ppo, rules, backtest, analysis)


def prepare_project_context(project_root: str | Path, logger_name: str = "pipeline") -> dict[str, Any]:
    config = load_project_config(project_root)
    output_paths = resolve_output_paths(config)
    logger = configure_logging(output_paths["logs"], name=logger_name)
    set_global_seed(int(config["seed"]))

    csv_path = locate_total_csv(config["project_root"], config["data_candidates"])
    raw = load_raw_total_csv(csv_path)
    cleaned, report = clean_total_data(
        raw,
        sample_start=config["sample_start"],
        sample_end=config["sample_end"],
        expected_freq_minutes=int(config["expected_frequency_minutes"]),
    )
    save_markdown(build_data_quality_markdown(report), output_paths["reports"] / "data_quality_report.md")

    bundle = build_weekly_bundle(cleaned, config)
    policy_result = parse_policy_environment(config["policy_directory"])
    policy_trace = build_policy_state_trace(bundle["weekly_metadata"], policy_result.rule_table, policy_result.inventory, config)
    save_markdown(
        build_policy_rule_summary_markdown(policy_result.inventory, policy_result.rule_table, policy_result.failures),
        output_paths["reports"] / "policy_rule_summary.md",
    )
    policy_result.inventory.to_csv(output_paths["metrics"] / "policy_file_inventory.csv", index=False)
    policy_result.inventory.to_csv(output_paths["metrics"] / "policy_metadata_index.csv", index=False)
    policy_result.rule_table.to_csv(output_paths["metrics"] / "policy_rule_table.csv", index=False)
    policy_result.failures.to_csv(output_paths["metrics"] / "policy_parse_failures.csv", index=False)
    policy_trace.to_csv(output_paths["metrics"] / "policy_state_trace.csv", index=False)

    bundle["weekly_metadata"] = bundle["weekly_metadata"].merge(policy_trace, on="week_start", how="left")
    bundle["weekly_features"] = bundle["weekly_features"].merge(
        policy_trace.drop(
            columns=[
                "policy_sources",
                "policy_names",
                "failed_policy_files",
                "active_state_groups",
                "mechanism_stage_label",
                "forward_price_linkage_type",
                "forward_mechanism_execution_type",
                "forward_ancillary_coupling_type",
                "forward_info_forecast_boundary_type",
            ],
            errors="ignore",
        ),
        on="week_start",
        how="left",
    )
    bundle["policy_inventory"] = policy_result.inventory
    bundle["policy_metadata_index"] = policy_result.inventory
    bundle["policy_rule_table"] = policy_result.rule_table
    bundle["policy_state_trace"] = policy_trace
    bundle["policy_failures"] = policy_result.failures
    bundle["policy_parse_failures"] = policy_result.failures

    split = build_week_split(config, bundle["weekly_features"], bundle["weekly_metadata"])
    all_eval_weeks = sorted(set(split.train + split.val + split.test))
    benchmark_actions = build_benchmark_actions(all_eval_weeks, bundle["weekly_features"], config)
    dynamic_baseline = simulate_strategy(bundle, all_eval_weeks, benchmark_actions["dynamic_lock_only"], config, "dynamic_lock_only")
    fixed_baseline = simulate_strategy(bundle, all_eval_weeks, benchmark_actions["fixed_lock"], config, "fixed_lock")
    rule_baseline = simulate_strategy(bundle, all_eval_weeks, benchmark_actions["rule_only"], config, "rule_only")

    reward_reference = dynamic_baseline["weekly_results"][
        ["week_start", "procurement_cost_w", "risk_term_w"]
    ].rename(
        columns={
            "procurement_cost_w": "baseline_cost_w",
            "risk_term_w": "baseline_risk_w",
        }
    )
    train_week_set = {pd.Timestamp(week) for week in split.train}
    dynamic_train = dynamic_baseline["weekly_results"].loc[
        dynamic_baseline["weekly_results"]["week_start"].isin(train_week_set)
    ]
    fixed_train = fixed_baseline["weekly_results"].loc[fixed_baseline["weekly_results"]["week_start"].isin(train_week_set)]
    rule_train = rule_baseline["weekly_results"].loc[rule_baseline["weekly_results"]["week_start"].isin(train_week_set)]
    delta_cost_series = pd.concat(
        [
            fixed_train["procurement_cost_w"].reset_index(drop=True) - dynamic_train["procurement_cost_w"].reset_index(drop=True),
            rule_train["procurement_cost_w"].reset_index(drop=True) - dynamic_train["procurement_cost_w"].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    reward_robust_stats = {
        "delta_cost": _robust_summary(delta_cost_series, float(config["reward"]["robust_eps"])),
    }
    bundle["reward_reference"] = reward_reference
    bundle["reward_robust_stats"] = reward_robust_stats
    bundle["baseline_dynamic_results"] = dynamic_baseline["weekly_results"]
    bundle["baseline_fixed_results"] = fixed_baseline["weekly_results"]
    bundle["baseline_rule_results"] = rule_baseline["weekly_results"]

    train_sequence = build_bootstrap_sequence(
        train_weeks=split.train,
        sequence_length=int(config["scenario"]["train_sequence_length"]),
        block_size=int(config["scenario"]["block_size"]),
        seed=int(config["scenario"]["bootstrap_seed"]),
    )

    bundle["feature_manifest"].to_csv(output_paths["metrics"] / "feature_manifest.csv", index=False)
    bundle["weekly_metadata"].to_csv(output_paths["metrics"] / "weekly_metadata.csv", index=False)
    bundle["weekly_features"].to_csv(output_paths["metrics"] / "weekly_features.csv", index=False)
    reward_reference.to_csv(output_paths["metrics"] / "reward_reference_dynamic_baseline.csv", index=False)
    dump_yaml(config, output_paths["reports"] / "train_config_snapshot.yaml")

    logger.info("已加载数据文件: %s", csv_path)
    logger.info("15分钟记录数: %s", len(bundle["quarter"]))
    logger.info("小时记录数: %s", len(bundle["hourly"]))
    logger.info("周度样本数: %s", len(bundle["weekly_metadata"]))
    logger.info("政策文件数: %s", len(policy_result.inventory))
    logger.info("政策规则数: %s", len(policy_result.rule_table))
    logger.info("政策解析失败文件数: %s", len(policy_result.failures))
    logger.info("奖励强基准: %s", config["reward"]["strong_baseline"])
    logger.info("预热周: %s", [week.strftime("%Y-%m-%d") for week in split.warmup])
    logger.info("训练周: %s", [week.strftime("%Y-%m-%d") for week in split.train])
    logger.info("验证周: %s", [week.strftime("%Y-%m-%d") for week in split.val])
    logger.info("回测周: %s", [week.strftime("%Y-%m-%d") for week in split.test])

    return {
        "config": config,
        "output_paths": output_paths,
        "logger": logger,
        "csv_path": csv_path,
        "raw": raw,
        "cleaned": cleaned,
        "data_quality_report": report,
        "bundle": bundle,
        "split": split,
        "train_sequence": train_sequence,
    }


def split_to_dict(split: WeekSplit) -> dict[str, list[str]]:
    return {
        "warmup": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.warmup],
        "train": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.train],
        "val": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.val],
        "test": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.test],
    }
