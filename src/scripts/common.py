from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_raw_total_csv, locate_total_csv
from src.data.preprocess import build_data_quality_markdown, clean_total_data
from src.data.scenario_generator import WeekSplit, build_bootstrap_sequence, build_week_split
from src.data.weekly_builder import build_weekly_bundle
from src.utils.io import dump_yaml, load_yaml, merge_configs, resolve_output_paths, save_markdown
from src.utils.logger import configure_logging
from src.utils.seeds import set_global_seed


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
    split = build_week_split(config, bundle["weekly_features"], bundle["weekly_metadata"])
    train_sequence = build_bootstrap_sequence(
        train_weeks=split.train,
        sequence_length=int(config["scenario"]["train_sequence_length"]),
        block_size=int(config["scenario"]["block_size"]),
        seed=int(config["scenario"]["bootstrap_seed"]),
    )

    bundle["feature_manifest"].to_csv(output_paths["metrics"] / "feature_manifest.csv", index=False)
    bundle["weekly_metadata"].to_csv(output_paths["metrics"] / "weekly_metadata.csv", index=False)
    bundle["weekly_features"].to_csv(output_paths["metrics"] / "weekly_features.csv", index=False)
    dump_yaml(config, output_paths["reports"] / "train_config_snapshot.yaml")

    logger.info("已加载数据文件: %s", csv_path)
    logger.info("15分钟记录数: %s", len(bundle["quarter"]))
    logger.info("小时记录数: %s", len(bundle["hourly"]))
    logger.info("周度样本数: %s", len(bundle["weekly_metadata"]))
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
