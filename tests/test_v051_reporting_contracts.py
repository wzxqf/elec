from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.scripts import run_pipeline
from src.analysis.reporting import build_excess_return_validation_summary, build_hourly_spot_activation_summary
from src.utils.versioning import load_project_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROJECT_VERSION = load_project_version(PROJECT_ROOT)


def test_run_pipeline_emits_current_integrated_version_report(monkeypatch) -> None:
    output_root = Path(".cache") / "tests" / f"pipeline-version-report-{uuid4().hex}"
    raw_dir = output_root / "raw"
    metrics_dir = raw_dir / "metrics"
    reports_dir = output_root / "reports"
    models_dir = raw_dir / "models"
    logs_dir = raw_dir / "logs"
    metadata_dir = raw_dir / "metadata"
    for path in [metrics_dir, reports_dir, models_dir, logs_dir, metadata_dir]:
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
                    "raw": raw_dir,
                    "metadata": metadata_dir,
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
        (reports_dir / "train_summary.md").write_text("# stale train report\n", encoding="utf-8")
        (metadata_dir / "training_runtime_summary.json").write_text("{}\n", encoding="utf-8")
        (metrics_dir / "hybrid_pso_training_trace.csv").write_text("iteration,best_score\n1,0\n", encoding="utf-8")
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
        (reports_dir / "validation_summary.md").write_text("# stale validation report\n", encoding="utf-8")
        (metrics_dir / "validation_weekly_results.csv").write_text("week_start,profit_w\n2026-02-02,1\n", encoding="utf-8")
        (metrics_dir / "validation_metrics.csv").write_text("total_profit\n1\n", encoding="utf-8")
        (metrics_dir / "contract_value_weekly.csv").write_text("contract_value_w\n1\n", encoding="utf-8")
        (metrics_dir / "risk_factor_manifest.csv").write_text("factor_category\nspot_price_volatility\n", encoding="utf-8")
        (metrics_dir / "policy_risk_adjusted_metrics.csv").write_text(
            "policy_risk_adjusted_excess_return_w\n1\n", encoding="utf-8"
        )
        return {
            "metrics": {
                "total_procurement_cost": 100.0,
                "total_profit": 20.0,
                "weekly_cost_volatility": 3.0,
                "cvar99": 4.0,
                "hedge_error": 0.05,
            },
            "weekly_results": pd.DataFrame(
                {
                    "week_start": ["2026-02-02"],
                    "lock_ratio_final": [0.48],
                    "contract_position_mwh": [1926659.375],
                    "profit_w": [20.0],
                    "cvar99_w": [4.0],
                    "hedge_error_w": [0.05],
                }
            ),
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
        for filename in [
            "backtest_summary.md",
            "benchmark_summary.md",
            "ablation_summary.md",
            "robustness_summary.md",
            "constraint_activation_report.md",
            "rolling_audit_summary.md",
        ]:
            (reports_dir / filename).write_text(f"# stale {filename}\n", encoding="utf-8")
        for filename in [
            "rolling_window_schedule.csv",
            "rolling_parameter_snapshots.csv",
            "rolling_validation_metrics.csv",
            "rolling_backtest_metrics.csv",
            "rolling_weekly_results.csv",
            "rolling_hourly_results.csv",
            "rolling_settlement_results.csv",
            "rolling_excess_return_metrics.csv",
            "backtest_metrics.csv",
            "benchmark_comparison.csv",
            "ablation_metrics.csv",
            "robustness_metrics.csv",
        ]:
            (metrics_dir / filename).write_text("value\n1\n", encoding="utf-8")

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
            "rolling_weekly_results": pd.DataFrame(
                {
                    "window_name": ["window_01"],
                    "week_start": ["2026-01-12"],
                    "lock_ratio_final": [0.52],
                    "contract_position_mwh": [1888000.0],
                    "profit_w": [40.0],
                    "cvar99_w": [5.0],
                    "hedge_error_w": [0.07],
                }
            ),
            "backtest_metrics": pd.DataFrame(
                {
                    "window_name": ["window_01"],
                    "strategy": ["window_01_test"],
                    "total_procurement_cost": [300.0],
                    "total_profit": [40.0],
                    "cvar99": [5.0],
                    "hedge_error": [0.07],
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

    human_report = reports_dir / f"{CURRENT_PROJECT_VERSION}_human_report.md"
    ai_report = reports_dir / f"{CURRENT_PROJECT_VERSION}_ai_structured_report.json"
    assert human_report.exists()
    assert ai_report.exists()
    generated_report_names = sorted(path.name for path in reports_dir.iterdir() if path.is_file())
    assert generated_report_names == [ai_report.name, human_report.name]
    report_text = human_report.read_text(encoding="utf-8")
    assert "数学模型与公式" in report_text
    assert "带符号净修正量" in report_text
    assert "q_spot_net" in report_text
    assert "模型运行参数设置" in report_text
    assert "模型运行效果" in report_text
    assert "实现效果" in report_text
    assert "完整数学模型" in report_text
    assert "全部数学结果" in report_text
    assert "J(theta_u, theta_l)" in report_text
    assert "CVaR_alpha" in report_text
    assert "约束投影" in report_text
    assert "正式验证周度结果" in report_text
    assert "2026-02-02" in report_text
    assert "滚动回测窗口结果" in report_text
    assert "window_01_test" in report_text
    assert "基准策略持出集结果" in report_text
    assert "消融结果" in report_text
    assert "稳健性结果" in report_text
    assert "完整数学结果产物索引" in report_text
    assert "raw/metrics/rolling_settlement_results.csv" in report_text
    assert f"reports/{CURRENT_PROJECT_VERSION}_ai_structured_report.json" in report_text
    assert "{version}" not in report_text

    ai_payload = json.loads(ai_report.read_text(encoding="utf-8"))
    assert ai_payload["report_type"] == "ai_structured_project_report"
    assert ai_payload["version"] == CURRENT_PROJECT_VERSION
    assert "mathematical_model" in ai_payload
    assert "mathematical_results" in ai_payload
    assert ai_payload["output_layout"]["human_report_path"] == f"reports/{human_report.name}"
    assert ai_payload["output_layout"]["raw_root"] == "raw"

    release_manifest = json.loads((output_root / "release_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    artifact_index = (output_root / "artifact_index.md").read_text(encoding="utf-8")

    expected_human_report_path = f"reports/{human_report.name}"
    expected_ai_report_path = f"reports/{ai_report.name}"
    assert release_manifest["key_outputs"]["human_report_path"] == expected_human_report_path
    assert run_manifest["key_outputs"]["human_report_path"] == expected_human_report_path
    assert release_manifest["key_outputs"]["ai_structured_report_path"] == expected_ai_report_path
    assert run_manifest["key_outputs"]["ai_structured_report_path"] == expected_ai_report_path
    assert f"human_report_path: {expected_human_report_path}" in artifact_index
    assert f"ai_structured_report_path: {expected_ai_report_path}" in artifact_index
    expected_key_outputs = {
        "model_path": "raw/models/hybrid_pso_model.json",
        "training_runtime_summary_path": "raw/metadata/training_runtime_summary.json",
        "training_trace_path": "raw/metrics/hybrid_pso_training_trace.csv",
        "validation_metrics_path": "raw/metrics/validation_metrics.csv",
        "validation_weekly_results_path": "raw/metrics/validation_weekly_results.csv",
        "contract_value_weekly_path": "raw/metrics/contract_value_weekly.csv",
        "risk_factor_manifest_path": "raw/metrics/risk_factor_manifest.csv",
        "policy_risk_adjusted_metrics_path": "raw/metrics/policy_risk_adjusted_metrics.csv",
        "rolling_window_schedule_path": "raw/metrics/rolling_window_schedule.csv",
        "rolling_parameter_snapshots_path": "raw/metrics/rolling_parameter_snapshots.csv",
        "rolling_validation_metrics_path": "raw/metrics/rolling_validation_metrics.csv",
        "rolling_backtest_metrics_path": "raw/metrics/rolling_backtest_metrics.csv",
        "rolling_weekly_results_path": "raw/metrics/rolling_weekly_results.csv",
        "rolling_hourly_results_path": "raw/metrics/rolling_hourly_results.csv",
        "rolling_settlement_results_path": "raw/metrics/rolling_settlement_results.csv",
        "rolling_excess_return_metrics_path": "raw/metrics/rolling_excess_return_metrics.csv",
        "backtest_metrics_path": "raw/metrics/backtest_metrics.csv",
        "benchmark_comparison_path": "raw/metrics/benchmark_comparison.csv",
        "ablation_metrics_path": "raw/metrics/ablation_metrics.csv",
        "robustness_metrics_path": "raw/metrics/robustness_metrics.csv",
        "human_report_path": expected_human_report_path,
        "ai_structured_report_path": expected_ai_report_path,
    }
    for key, relative_path in expected_key_outputs.items():
        assert release_manifest["key_outputs"][key] == relative_path
        assert run_manifest["key_outputs"][key] == relative_path
        assert f"- {key}: {relative_path}" in artifact_index


def test_excess_return_summary_does_not_label_partial_wins_as_persistent() -> None:
    text = build_excess_return_validation_summary(
        policy_metrics=pd.DataFrame({"excess_profit_w": [1.0], "policy_risk_penalty_w": [0.0], "policy_risk_adjusted_excess_return_w": [1.0]}),
        rolling_metrics=pd.DataFrame(
            {
                "strong_baseline_family_outperformed": [True, False],
                "active_excess_return_persistent": [False, False],
                "window_policy_risk_adjusted_sharpe": [0.0, 0.0],
            }
        ),
    )

    assert "跑赢强基准族窗口数: 1/2" in text
    assert "持续超额收益窗口数: 0/2" in text
    assert "部分滚动窗口跑赢强基准族，但未满足持续性口径" in text


def test_build_hourly_spot_activation_summary_reports_nonzero_hours() -> None:
    frame = pd.DataFrame(
        {
            "spot_hedge_nonzero_hours_w": [2, 3],
            "spot_hedge_abs_mwh_w": [10.0, 15.0],
            "friction_cost_w": [12.0, 18.0],
        }
    )
    summary = build_hourly_spot_activation_summary(frame)
    assert "nonzero_hours=5" in summary
    assert "spot_abs_sum_mwh=25.000000" in summary
    assert "friction_cost_sum=30.000000" in summary
