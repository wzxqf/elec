from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.scripts.common import prepare_project_context


def test_prepare_project_context_exports_manifests_and_audits() -> None:
    context = prepare_project_context(Path.cwd(), logger_name="test_prepare_context_current")

    assert "feasible_domain" in context["bundle"]
    assert (context["output_paths"]["metrics"] / "feasible_domain_manifest.csv").exists()
    assert (context["output_paths"]["reports"] / "policy_feasible_domain_summary.md").exists()
    assert (context["output_paths"]["reports"] / "parameter_layout_audit.md").exists()
    assert (context["output_paths"]["root"] / "release_manifest.json").exists()
    assert (context["output_paths"]["root"] / "run_manifest.json").exists()
    assert (context["output_paths"]["root"] / "artifact_index.md").exists()

    release_manifest = json.loads((context["output_paths"]["root"] / "release_manifest.json").read_text(encoding="utf-8"))
    feasible_domain_summary = (context["output_paths"]["reports"] / "policy_feasible_domain_summary.md").read_text(encoding="utf-8")
    feature_manifest = pd.read_csv(context["output_paths"]["metrics"] / "feature_manifest.csv")

    assert release_manifest["version"] == context["config"]["version"]
    assert release_manifest["config_hash"]
    assert release_manifest["compiled_layout_hash"]
    assert release_manifest["data_range"]["sample_start"]
    assert release_manifest["enabled_constraints"]
    assert "summary_scope: static_policy_bound_inventory" in feasible_domain_summary
    assert "policy_tightening_week_count" in feasible_domain_summary
    assert "default_bound_week_count" in feasible_domain_summary
    assert "runtime_projection_reference" in feasible_domain_summary
    assert "policy_tightening_trigger_count" not in feasible_domain_summary
    assert "default_projection_trigger_count" not in feasible_domain_summary
    assert "selected_for_policy_head" in feature_manifest.columns
    assert "report_only" in feature_manifest.columns
