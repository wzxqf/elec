from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.feature_engineering import prepare_datasets
from src.data.loader import load_raw_total_csv, locate_total_csv
from src.data.preprocess import build_data_quality_markdown, clean_total_data
from src.data.scenario_generator import MonthSplit, build_bootstrap_sequence, build_month_split
from src.utils.io import dump_yaml, load_yaml, merge_configs, resolve_output_paths, save_markdown
from src.utils.logger import configure_logging


def load_project_config(project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root)
    default = load_yaml(project_root / "configs" / "default.yaml")
    ppo = load_yaml(project_root / "configs" / "ppo.yaml")
    data = load_yaml(project_root / "configs" / "data.yaml")
    backtest = load_yaml(project_root / "configs" / "backtest.yaml")
    return merge_configs(default, ppo, data, backtest)


def prepare_project_context(project_root: str | Path) -> dict[str, Any]:
    config = load_project_config(project_root)
    output_paths = resolve_output_paths(config)
    logger = configure_logging(output_paths["logs"], name="pipeline")
    csv_path = locate_total_csv(config["project_root"], config["data_candidates"])
    raw = load_raw_total_csv(csv_path)
    cleaned, report = clean_total_data(
        raw,
        sample_start=config["sample_start"],
        sample_end=config["sample_end"],
        expected_freq_minutes=int(config["expected_frequency_minutes"]),
    )
    save_markdown(build_data_quality_markdown(report), output_paths["reports"] / "data_quality_report.md")
    bundle = prepare_datasets(cleaned, config["policy_events"])
    split = build_month_split(config, bundle["monthly_features"])
    train_sequence = build_bootstrap_sequence(
        train_months=split.train,
        sequence_length=int(config["scenario"]["train_sequence_length"]),
        block_size=int(config["scenario"]["block_size"]),
        seed=int(config["scenario"]["bootstrap_seed"]),
    )
    dump_yaml(config, output_paths["logs"] / "config_snapshot.yaml")

    logger.info("Loaded data from %s", csv_path)
    logger.info("Training months: %s", [month.strftime("%Y-%m") for month in split.train])
    logger.info("Validation months: %s", [month.strftime("%Y-%m") for month in split.val])
    logger.info("Test months: %s", [month.strftime("%Y-%m") for month in split.test])

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


def split_to_dict(split: MonthSplit) -> dict[str, list[str]]:
    return {
        "warmup": [month.strftime("%Y-%m") for month in split.warmup],
        "train": [month.strftime("%Y-%m") for month in split.train],
        "val": [month.strftime("%Y-%m") for month in split.val],
        "test": [month.strftime("%Y-%m") for month in split.test],
    }
