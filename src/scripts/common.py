import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.state_audit import build_state_schema_markdown, build_tensor_bundle_audit_markdown
from src.backtest.rolling_pipeline import build_rolling_retrain_plan
from src.analysis.model_layout_reporting import build_parameter_layout_markdown, build_parameter_layout_payload
from src.config.load_config import load_runtime_config
from src.data.loader import load_raw_total_csv, locate_total_csv
from src.data.preprocess import build_data_quality_markdown, clean_total_data
from src.data.scenario_generator import WeekSplit, build_week_split
from src.data.weekly_builder import build_weekly_bundle
from src.model_layout.compiler import compile_parameter_layout
from src.policy.feasible_domain import compile_feasible_domain
from src.policy.market_constraints import (
    build_market_rule_constraints,
    build_market_rule_constraints_markdown,
    validate_market_rule_alignment,
)
from src.policy_deep.regime_builder import build_policy_deep_context
from src.training.tensor_bundle import compile_training_tensor_bundle
from src.utils.experiment_manifest import (
    build_artifact_index_markdown,
    build_feasible_domain_summary,
    build_parameter_layout_audit_markdown,
    build_release_manifest,
    build_run_manifest,
    build_run_metadata,
    prepend_report_header,
    relativize_path,
)
from src.utils.io import dump_yaml, resolve_output_paths, save_json, save_markdown
from src.utils.logger import configure_logging
from src.utils.seeds import set_global_seed


DEFAULT_CONFIG_FILENAME = "experiment_config.yaml"
CONFIG_ENV_VAR = "ELEC_CONFIG_PATH"
REALIZED_WEEKLY_AGENT_EXCLUDE_COLUMNS = {
    "actual_weekly_net_demand_mwh",
    "da_id_cross_corr_w",
    "extreme_price_spike_flag_w",
    "extreme_event_flag_w",
}


def resolve_project_config_path(project_root: str | Path, config_path: str | Path | None = None) -> Path:
    root = Path(project_root).resolve()
    reference = config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_FILENAME
    candidate = Path(reference).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def load_project_config(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_config_path = resolve_project_config_path(project_root, config_path=config_path)
    return load_runtime_config(project_root, filename=str(resolved_config_path))


def _build_feature_manifest(
    weekly_features: pd.DataFrame,
    base_manifest: pd.DataFrame,
    policy_trace: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    include_set = set(config["feature_selection"].get("feature_include_for_agent", []))
    exclude_set = set(config["feature_selection"].get("feature_exclude_for_agent", []))
    report_only_set = set(config["feature_selection"].get("feature_keep_for_report_only", []))
    policy_head_set = set(_resolve_policy_head_columns(policy_trace, config))
    rows: list[dict[str, Any]] = []
    agent_columns: list[str] = []
    policy_columns = set(policy_trace.columns) - {"week_start"}
    base_rows = base_manifest.set_index("column").to_dict(orient="index") if not base_manifest.empty else {}
    for column in weekly_features.columns:
        if column == "week_start":
            continue
        numeric = pd.api.types.is_numeric_dtype(weekly_features[column])
        source = base_rows.get(column, {}).get("source", "政策状态" if column in policy_columns else "周度聚合")
        realized_weekly = column in REALIZED_WEEKLY_AGENT_EXCLUDE_COLUMNS
        selected = bool(
            numeric
            and not realized_weekly
            and column not in exclude_set
            and (column in include_set or column not in policy_columns)
        )
        selected_for_policy_head = bool(numeric and column in policy_head_set)
        report_only = bool(realized_weekly or column in report_only_set or (numeric and not selected and not selected_for_policy_head))
        rows.append(
            {
                "column": column,
                "source": source,
                "is_numeric": numeric,
                "selected_for_agent": selected,
                "selected_for_policy_head": selected_for_policy_head,
                "report_only": report_only,
            }
        )
        if selected:
            agent_columns.append(column)
    return pd.DataFrame(rows), agent_columns


def _resolve_policy_head_columns(policy_trace: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    numeric_policy_columns = [
        column
        for column in policy_trace.columns
        if column != "week_start" and pd.api.types.is_numeric_dtype(policy_trace[column])
    ]
    compiler_cfg = config.get("parameter_compiler", {}).get("upper", {})
    blocks = compiler_cfg.get("blocks", {})
    policy_block = blocks.get("policy_feature_weights", {})
    include_columns = policy_block.get("include")
    if include_columns:
        return [str(column) for column in include_columns if str(column) in numeric_policy_columns]
    return numeric_policy_columns


def _align_lt_price_metadata(weekly_metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = weekly_metadata.copy()
    linked_active = pd.to_numeric(metadata.get("lt_price_linked_active", pd.Series(0.0, index=metadata.index)), errors="coerce").fillna(0.0)
    lt_price_w = pd.to_numeric(metadata.get("lt_price_w", pd.Series(np.nan, index=metadata.index)), errors="coerce")
    fixed_ratio_raw = pd.to_numeric(metadata.get("fixed_price_ratio_max", pd.Series(0.4, index=metadata.index)), errors="coerce").fillna(0.4)
    linked_ratio_raw = pd.to_numeric(metadata.get("linked_price_ratio_min", pd.Series(0.6, index=metadata.index)), errors="coerce").fillna(0.6)
    total_ratio = (fixed_ratio_raw + linked_ratio_raw).replace(0.0, 1.0)

    metadata["lt_price_source"] = np.where(
        lt_price_w.notna(),
        np.where(linked_active >= 0.5, "linked_mix_40da_60id", "prev_week_da_proxy"),
        metadata.get("lt_price_source", pd.Series("warmup_unavailable", index=metadata.index)),
    )
    metadata["lt_price_fixed_ratio"] = np.where(lt_price_w.notna(), np.where(linked_active >= 0.5, fixed_ratio_raw / total_ratio, 1.0), 0.0)
    metadata["lt_price_linked_ratio"] = np.where(lt_price_w.notna(), np.where(linked_active >= 0.5, linked_ratio_raw / total_ratio, 0.0), 0.0)
    return metadata


def subset_bundle_for_weeks(bundle: dict[str, Any], weeks: list[pd.Timestamp]) -> dict[str, Any]:
    week_set = {pd.Timestamp(week) for week in weeks}
    subset = dict(bundle)
    for key in ["weekly_features", "weekly_metadata", "policy_state_trace"]:
        subset[key] = bundle[key].loc[bundle[key]["week_start"].isin(week_set)].reset_index(drop=True)
    for key in ["hourly", "quarter"]:
        subset[key] = bundle[key].loc[bundle[key]["week_start"].isin(week_set)].reset_index(drop=True)
    if "feasible_domain" in bundle:
        feasible_domain = bundle["feasible_domain"]
        subset["feasible_domain"] = type(feasible_domain)(
            weekly_bounds=feasible_domain.weekly_bounds.loc[feasible_domain.weekly_bounds["week_start"].isin(week_set)].reset_index(drop=True),
            hourly_bounds=feasible_domain.hourly_bounds.loc[feasible_domain.hourly_bounds["week_start"].isin(week_set)].reset_index(drop=True),
            settlement_semantics=feasible_domain.settlement_semantics.loc[
                feasible_domain.settlement_semantics["week_start"].isin(week_set)
            ].reset_index(drop=True),
        )
    subset["tensor_bundle"] = compile_training_tensor_bundle(subset, device=bundle["tensor_bundle"].device)
    return subset


def prepare_project_context(
    project_root: str | Path,
    logger_name: str = "pipeline",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_project_config(project_root, config_path=config_path)
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
    data_quality_markdown = build_data_quality_markdown(report)

    bundle = build_weekly_bundle(cleaned, config)
    policy_deep = build_policy_deep_context(
        policy_directory=config["policy_directory"],
        weekly_metadata=bundle["weekly_metadata"],
        config=config,
    )
    market_constraints = build_market_rule_constraints(config, policy_deep.rule_table)
    market_constraint_violations = validate_market_rule_alignment(config, policy_deep.rule_table, policy_deep.state_trace)

    bundle["weekly_metadata"] = bundle["weekly_metadata"].merge(policy_deep.state_trace, on="week_start", how="left")
    bundle["weekly_metadata"] = _align_lt_price_metadata(bundle["weekly_metadata"])
    bundle["weekly_features"] = bundle["weekly_features"].merge(
        policy_deep.state_trace.drop(columns=["policy_sources", "policy_names", "failed_policy_files", "active_state_groups"], errors="ignore"),
        on="week_start",
        how="left",
    )
    bundle["weekly_features"] = bundle["weekly_features"].drop(
        columns=["lt_price_source", "lt_price_fixed_ratio", "lt_price_linked_ratio"],
        errors="ignore",
    ).merge(
        bundle["weekly_metadata"][["week_start", "lt_price_source", "lt_price_fixed_ratio", "lt_price_linked_ratio"]],
        on="week_start",
        how="left",
    )
    bundle["policy_inventory"] = policy_deep.inventory
    bundle["policy_document_units"] = policy_deep.document_units
    bundle["policy_candidate_rules"] = policy_deep.candidate_rules
    bundle["policy_reviewed_rules"] = policy_deep.reviewed_rules
    bundle["policy_rule_table"] = policy_deep.rule_table
    bundle["policy_state_trace"] = policy_deep.state_trace
    bundle["policy_failures"] = policy_deep.failures
    bundle["market_rule_constraints"] = market_constraints
    bundle["market_rule_constraint_violations"] = market_constraint_violations

    feature_manifest, agent_feature_columns = _build_feature_manifest(
        weekly_features=bundle["weekly_features"],
        base_manifest=bundle["feature_manifest"],
        policy_trace=policy_deep.state_trace,
        config=config,
    )
    bundle["feature_manifest"] = feature_manifest
    bundle["agent_feature_columns"] = agent_feature_columns
    feasible_domain = compile_feasible_domain(
        config=config,
        weekly_metadata=bundle["weekly_metadata"],
        policy_state_trace=policy_deep.state_trace,
    )
    bundle["feasible_domain"] = feasible_domain
    compiled_parameter_layout = compile_parameter_layout(config=config, bundle=bundle)
    bundle["compiled_parameter_layout"] = compiled_parameter_layout
    bundle["tensor_bundle"] = compile_training_tensor_bundle(bundle, device=config["training"]["device"])

    split = build_week_split(config, bundle["weekly_features"], bundle["weekly_metadata"])
    rolling_plan = build_rolling_retrain_plan(config, sorted(pd.to_datetime(bundle["weekly_features"]["week_start"]).tolist()))

    policy_deep.inventory.to_csv(output_paths["metrics"] / "policy_file_inventory.csv", index=False)
    policy_deep.document_units.to_csv(output_paths["metrics"] / "policy_document_units.csv", index=False)
    policy_deep.candidate_rules.to_csv(output_paths["metrics"] / "policy_rule_candidates.csv", index=False)
    policy_deep.reviewed_rules.to_csv(output_paths["metrics"] / "policy_rule_reviewed.csv", index=False)
    policy_deep.rule_table.to_csv(output_paths["metrics"] / "policy_rule_table.csv", index=False)
    policy_deep.state_trace.to_csv(output_paths["metrics"] / "policy_state_trace.csv", index=False)
    policy_deep.failures.to_csv(output_paths["metrics"] / "policy_parse_failures.csv", index=False)
    market_constraints.to_csv(output_paths["metrics"] / "market_rule_constraints.csv", index=False)
    feasible_domain.weekly_bounds.to_csv(output_paths["metrics"] / "feasible_domain_manifest.csv", index=False)
    market_rule_constraints_markdown = build_market_rule_constraints_markdown(
        config=config,
        constraints=market_constraints,
        rule_table=policy_deep.rule_table,
        violations=market_constraint_violations,
    )
    feasible_domain_summary_markdown = build_feasible_domain_summary(feasible_domain)
    bundle["weekly_metadata"].to_csv(output_paths["metrics"] / "weekly_metadata.csv", index=False)
    bundle["weekly_features"].to_csv(output_paths["metrics"] / "weekly_features.csv", index=False)
    feature_manifest.to_csv(output_paths["metrics"] / "feature_manifest.csv", index=False)
    save_json(
        {
            "version": config["version"],
            "agent_feature_columns": agent_feature_columns,
            "manifest": feature_manifest.to_dict(orient="records"),
        },
        output_paths["reports"] / "feature_manifest.json",
    )
    layout_payload = build_parameter_layout_payload(compiled_parameter_layout)
    save_json(layout_payload, output_paths["reports"] / "compiled_parameter_layout.json")
    run_metadata = build_run_metadata(
        config=config,
        output_root=output_paths["root"],
        compiled_layout_payload=layout_payload,
        constraints=market_constraints,
    )
    state_schema_markdown = build_state_schema_markdown(bundle, bundle["tensor_bundle"])
    tensor_bundle_audit_markdown = build_tensor_bundle_audit_markdown(bundle, bundle["tensor_bundle"])
    save_markdown(prepend_report_header(data_quality_markdown, run_metadata), output_paths["reports"] / "data_quality_report.md")
    save_markdown(
        prepend_report_header(market_rule_constraints_markdown, run_metadata),
        output_paths["reports"] / "market_rule_constraints.md",
    )
    save_markdown(
        prepend_report_header(feasible_domain_summary_markdown, run_metadata),
        output_paths["reports"] / "policy_feasible_domain_summary.md",
    )
    save_markdown(
        prepend_report_header(build_parameter_layout_markdown(compiled_parameter_layout), run_metadata),
        output_paths["reports"] / "parameter_layout_summary.md",
    )
    save_markdown(
        prepend_report_header(build_parameter_layout_audit_markdown(compiled_parameter_layout), run_metadata),
        output_paths["reports"] / "parameter_layout_audit.md",
    )
    save_markdown(
        prepend_report_header(state_schema_markdown, run_metadata),
        output_paths["reports"] / "state_schema_snapshot.md",
    )
    save_markdown(
        prepend_report_header(tensor_bundle_audit_markdown, run_metadata),
        output_paths["reports"] / "tensor_bundle_audit.md",
    )
    dump_yaml(config["raw_experiment_config"], output_paths["reports"] / "train_config_snapshot.yaml")
    key_outputs = {
        "compiled_layout_path": relativize_path(output_paths["reports"] / "compiled_parameter_layout.json", output_paths["root"]),
        "parameter_layout_audit_path": relativize_path(output_paths["reports"] / "parameter_layout_audit.md", output_paths["root"]),
        "state_schema_snapshot_path": relativize_path(output_paths["reports"] / "state_schema_snapshot.md", output_paths["root"]),
        "tensor_bundle_audit_path": relativize_path(output_paths["reports"] / "tensor_bundle_audit.md", output_paths["root"]),
        "feasible_domain_manifest_path": relativize_path(output_paths["metrics"] / "feasible_domain_manifest.csv", output_paths["root"]),
        "policy_rule_table_path": relativize_path(output_paths["metrics"] / "policy_rule_table.csv", output_paths["root"]),
        "policy_state_trace_path": relativize_path(output_paths["metrics"] / "policy_state_trace.csv", output_paths["root"]),
        "feature_manifest_path": relativize_path(output_paths["reports"] / "feature_manifest.json", output_paths["root"]),
    }
    save_json(build_release_manifest(run_metadata, config, key_outputs), output_paths["root"] / "release_manifest.json")
    save_json(build_run_manifest(run_metadata, config, key_outputs), output_paths["root"] / "run_manifest.json")
    save_markdown(build_artifact_index_markdown(run_metadata, key_outputs), output_paths["root"] / "artifact_index.md")
    logger.info("已加载数据文件: %s", csv_path)
    logger.info("训练算法: %s", config["training"]["algorithm"])
    logger.info("政策来源文件数: %s", len(policy_deep.inventory))
    logger.info("政策候选规则数: %s", len(policy_deep.candidate_rules))
    logger.info("政策正式规则数: %s", len(policy_deep.rule_table))
    logger.info("滚动重训窗口数: %s", len(rolling_plan))
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
        "rolling_plan": rolling_plan,
        "run_metadata": run_metadata,
    }


def split_to_dict(split: WeekSplit) -> dict[str, list[str]]:
    return {
        "warmup": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.warmup],
        "train": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.train],
        "val": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.val],
        "test": [pd.Timestamp(week).strftime("%Y-%m-%d") for week in split.test],
    }
