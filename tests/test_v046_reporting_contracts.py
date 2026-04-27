from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.scripts import run_pipeline
from src.utils.versioning import build_version_report_filename, load_project_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROJECT_VERSION = load_project_version(PROJECT_ROOT)
CURRENT_VERSION_REPORT_FILENAME = build_version_report_filename(CURRENT_PROJECT_VERSION)


def test_run_pipeline_emits_current_integrated_version_report(monkeypatch) -> None:
    output_root = Path(".cache") / "tests" / f"pipeline-version-report-{uuid4().hex}"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    models_dir = output_root / "models"
    logs_dir = output_root / "logs"
    for path in [metrics_dir, reports_dir, models_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    original_prepare = run_pipeline.prepare_project_context
    original_train = run_pipeline.run_train
    original_evaluate = run_pipeline.run_evaluate
    original_backtest = run_pipeline.run_backtest

    def fake_prepare_project_context(*args, **kwargs):
        return {
            "config": {
                "version": CURRENT_PROJECT_VERSION,
                "seed": 42,
                "sample_start": "2025-11-01 00:00:00",
                "sample_end": "2026-03-20 23:45:00",
                "split": {
                    "train_start_week": "2025-11-03 00:00:00",
                    "train_end_week": "2026-01-26 00:00:00",
                    "val_start_week": "2026-02-02 00:00:00",
                    "val_end_week": "2026-02-23 00:00:00",
                    "test_start_week": "2026-03-02 00:00:00",
                    "test_end_week": "2026-03-16 00:00:00",
                },
                "training": {"algorithm": "HYBRID_PSO_V040", "device": "cpu"},
            },
            "output_paths": {
                "root": output_root,
                "reports": reports_dir,
                "metrics": metrics_dir,
                "models": models_dir,
                "logs": logs_dir,
            },
            "run_metadata": {
                "version": CURRENT_PROJECT_VERSION,
                "experiment_id": f"{CURRENT_PROJECT_VERSION}-test",
                "config_hash": "abc123",
                "compiled_layout_hash": "layout123",
                "run_timestamp": "2026-04-17T12:00:00+08:00",
                "device": "cpu",
                "data_range": {
                    "sample_start": "2025-11-01 00:00:00",
                    "sample_end": "2026-03-20 23:45:00",
                },
                "enabled_constraints": ["c1"],
            },
            "bundle": {
                "policy_rule_table": pd.DataFrame({"rule_id": ["r1"], "state_name": ["renewable_mechanism_active"]}),
                "market_rule_constraints": pd.DataFrame({"constraint_id": ["c1"], "model_mapping": ["renewable_mechanism_active"]}),
            },
        }

    def fake_run_train(context):
        (models_dir / "hybrid_pso_model.json").write_text("{}", encoding="utf-8")
        return {
            "model": object(),
            "device": "cpu",
            "runtime_profile": {
                "score_kernel_device": "cpu",
                "upper_particles": 8,
                "lower_particles": 6,
                "iterations": 3,
                "upper_dim": 140,
                "lower_dim": 32,
                "upper_dim_real": 170,
                "lower_dim_real": 32,
            },
            "model_path": models_dir / "hybrid_pso_model.json",
        }

    def fake_run_evaluate(context, model=None):
        return {
            "metrics": {
                "total_procurement_cost": 100.0,
                "total_profit": 20.0,
                "weekly_cost_volatility": 3.0,
                "cvar99": 4.0,
                "hedge_error": 0.05,
            },
            "contract_value_weekly": pd.DataFrame({"contract_value_w": [325.0], "stability_score_w": [0.8]}),
            "risk_factor_manifest": pd.DataFrame({"factor_category": ["spot_price_volatility"]}),
            "policy_risk_metrics": pd.DataFrame(
                {
                    "policy_risk_penalty_w": [1.0],
                    "policy_risk_adjusted_excess_return_w": [2.0],
                    "excess_profit_w": [3.0],
                }
            ),
        }

    def fake_run_backtest(context, model=None):
        class _Summary:
            aggregate = {
                "window_count": 10.0,
                "mean_total_procurement_cost": 300.0,
                "mean_total_profit": 40.0,
                "mean_cvar99": 5.0,
            }

        return {
            "rolling_summary": _Summary(),
            "rolling_excess_return_metrics": pd.DataFrame(
                {
                    "window_name": ["window_01"],
                    "window_policy_risk_adjusted_sharpe": [1.2],
                    "active_excess_return_persistent": [True],
                }
            ),
            "benchmark_metrics": pd.DataFrame(
                {
                    "strategy_name": ["dynamic_lock_only", "static_no_spot_adjustment"],
                    "total_profit": [100.0, 110.0],
                    "cvar99": [40.0, 45.0],
                }
            ),
            "ablation_metrics": pd.DataFrame(
                {
                    "variant_name": ["full_model", "no_policy_projection"],
                    "total_profit": [120.0, 90.0],
                    "cvar99": [60.0, 80.0],
                }
            ),
            "robustness_metrics": pd.DataFrame(
                {
                    "scenario_name": ["policy_cutoff_2026-02-01"],
                    "total_profit": [150.0],
                    "mean_cvar99": [55.0],
                }
            ),
        }

    monkeypatch.setattr(run_pipeline, "prepare_project_context", fake_prepare_project_context)
    monkeypatch.setattr(run_pipeline, "run_train", fake_run_train)
    monkeypatch.setattr(run_pipeline, "run_evaluate", fake_run_evaluate)
    monkeypatch.setattr(run_pipeline, "run_backtest", fake_run_backtest)

    try:
        run_pipeline.main()
    finally:
        monkeypatch.setattr(run_pipeline, "prepare_project_context", original_prepare)
        monkeypatch.setattr(run_pipeline, "run_train", original_train)
        monkeypatch.setattr(run_pipeline, "run_evaluate", original_evaluate)
        monkeypatch.setattr(run_pipeline, "run_backtest", original_backtest)

    version_report = reports_dir / CURRENT_VERSION_REPORT_FILENAME
    assert version_report.exists()
    report_text = version_report.read_text(encoding="utf-8")
    assert "数学模型与公式" in report_text
    assert "带符号净修正量" in report_text
    assert "q_spot_net" in report_text
    assert "模型运行参数设置" in report_text
    assert "模型运行效果" in report_text
    assert "实现效果" in report_text

    release_manifest = json.loads((output_root / "release_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    artifact_index = (output_root / "artifact_index.md").read_text(encoding="utf-8")

    expected_version_report_path = f"reports/{CURRENT_VERSION_REPORT_FILENAME}"
    assert release_manifest["key_outputs"]["version_report_path"] == expected_version_report_path
    assert run_manifest["key_outputs"]["version_report_path"] == expected_version_report_path
    assert f"version_report_path: {expected_version_report_path}" in artifact_index
    assert "train_summary_path" not in release_manifest["key_outputs"]
    assert "training_runtime_summary_path" not in release_manifest["key_outputs"]
    assert "validation_summary_path" not in release_manifest["key_outputs"]
    assert "benchmark_summary_path" not in release_manifest["key_outputs"]
    assert "- train_summary_path:" not in artifact_index
    assert "- validation_summary_path:" not in artifact_index
