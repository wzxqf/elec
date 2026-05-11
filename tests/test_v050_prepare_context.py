from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.scripts.common import prepare_project_context
from src.utils.experiment_manifest import merge_key_outputs_preserving_existing


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_temp_config(output_root: Path) -> Path:
    config_path = output_root.parent / "experiment_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = (PROJECT_ROOT / "experiment_config.yaml").read_text(encoding="utf-8")
    text = text.replace("  root: outputs", f"  root: {output_root.as_posix()}")
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_manifest_key_output_merge_preserves_full_pipeline_outputs() -> None:
    output_root = Path(".cache") / "tests" / f"manifest-merge-{uuid4().hex}" / "outputs" / "v0.50"
    reports_dir = output_root / "reports"
    metrics_dir = output_root / "raw" / "metrics"
    metadata_dir = output_root / "raw" / "metadata"
    reports_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "compiled_parameter_layout.json").write_text("{}\n", encoding="utf-8")
    (reports_dir / "v0.50_human_report.md").write_text("# report\n", encoding="utf-8")
    (metrics_dir / "benchmark_comparison.csv").write_text("strategy_name,total_profit\nx,1\n", encoding="utf-8")
    (output_root / "release_manifest.json").write_text(
        json.dumps(
            {
                "key_outputs": {
                    "human_report_path": "reports/v0.50_human_report.md",
                    "benchmark_comparison_path": "raw/metrics/benchmark_comparison.csv",
                    "stale_missing_path": "reports/missing.md",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    merged = merge_key_outputs_preserving_existing(
        output_root,
        {
            "compiled_layout_path": "raw/metadata/compiled_parameter_layout.json",
        },
    )

    assert merged["compiled_layout_path"] == "raw/metadata/compiled_parameter_layout.json"
    assert merged["human_report_path"] == "reports/v0.50_human_report.md"
    assert merged["benchmark_comparison_path"] == "raw/metrics/benchmark_comparison.csv"
    assert "stale_missing_path" not in merged


def test_prepare_project_context_exports_manifests_and_audits() -> None:
    output_root = Path(".cache") / "tests" / f"prepare-context-{uuid4().hex}" / "outputs"
    config_path = _write_temp_config(output_root)
    context = prepare_project_context(PROJECT_ROOT, logger_name="test_prepare_context_current", config_path=config_path)

    assert "feasible_domain" in context["bundle"]
    assert (context["output_paths"]["metrics"] / "feasible_domain_manifest.csv").exists()
    assert (context["output_paths"]["metadata"] / "compiled_parameter_layout.json").exists()
    assert (context["output_paths"]["metadata"] / "train_config_snapshot.yaml").exists()
    assert (context["output_paths"]["root"] / "release_manifest.json").exists()
    assert (context["output_paths"]["root"] / "run_manifest.json").exists()
    assert (context["output_paths"]["root"] / "artifact_index.md").exists()

    release_manifest = json.loads((context["output_paths"]["root"] / "release_manifest.json").read_text(encoding="utf-8"))
    feature_manifest = pd.read_csv(context["output_paths"]["metrics"] / "feature_manifest.csv")

    assert release_manifest["version"] == context["config"]["version"]
    assert release_manifest["config_hash"]
    assert release_manifest["compiled_layout_hash"]
    assert release_manifest["data_range"]["sample_start"]
    assert release_manifest["enabled_constraints"]
    assert release_manifest["compiled_layout_path"].startswith("raw/metadata/")
    assert "selected_for_policy_head" in feature_manifest.columns
    assert "report_only" in feature_manifest.columns
