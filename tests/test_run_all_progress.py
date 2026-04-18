from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd

import run_all
from src.scripts import run_pipeline
from src.scripts import train as train_script


def test_read_status_snapshot_returns_defaults_for_missing_file() -> None:
    status_path = Path(".cache") / "tests" / f"missing-status-{uuid4().hex}.json"
    snapshot = run_all.read_status_snapshot(status_path)

    assert snapshot["stage"] == "等待启动"
    assert snapshot["phase_name"] == "初始化"
    assert snapshot["phase_progress"] == 0.0
    assert snapshot["total_progress"] == 0.0


def test_read_status_snapshot_loads_json_payload() -> None:
    status_path = Path(".cache") / "tests" / f"runtime-status-{uuid4().hex}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "stage": "训练",
                "phase_name": "HPSO 参数训练",
                "phase_progress": 0.375,
                "total_progress": 0.125,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshot = run_all.read_status_snapshot(status_path)

    assert snapshot["stage"] == "训练"
    assert snapshot["phase_name"] == "HPSO 参数训练"
    assert snapshot["phase_progress"] == 0.375
    assert snapshot["total_progress"] == 0.125


def test_format_progress_line_includes_stage_progress_and_resources() -> None:
    line = run_all.format_progress_line(
        {
            "stage": "训练",
            "phase_name": "HPSO 参数训练",
            "phase_progress": 0.5,
            "total_progress": 0.25,
            "message": "iter 40/80",
        },
        {
            "process_cpu_percent": 88.4,
            "process_memory_gb": 3.2,
            "gpu_util_percent": 67.0,
            "gpu_memory_used_gb": 5.5,
            "gpu_memory_total_gb": 16.0,
        },
        elapsed_seconds=125.0,
        width=220,
    )

    assert "训练" in line
    assert "HPSO 参数训练" in line
    assert "总进度 25.0%" in line
    assert "阶段 50.0%" in line
    assert "CPU 88.4%" in line
    assert "GPU 67.0%" in line
    assert "显存 5.5/16.0GB" in line
    assert "iter 40/80" in line
    assert "\n" not in line


def test_run_pipeline_honors_runtime_status_env(monkeypatch) -> None:
    status_path = Path(".cache") / "tests" / f"pipeline-status-{uuid4().hex}.json"
    monkeypatch.setenv("ELEC_RUNTIME_STATUS_PATH", str(status_path))
    output_root = Path(".cache") / "tests" / f"pipeline-{uuid4().hex}"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    original_prepare = run_pipeline.prepare_project_context
    original_train = run_pipeline.run_train
    original_evaluate = run_pipeline.run_evaluate
    original_backtest = run_pipeline.run_backtest

    def fake_prepare_project_context(*args, **kwargs):
        return {
            "config": {"version": "v0.36", "training": {"algorithm": "HYBRID_PSO_V036"}},
            "output_paths": {"reports": reports_dir, "metrics": metrics_dir},
            "bundle": {
                "policy_rule_table": pd.DataFrame({"rule_id": ["r1"], "state_name": ["renewable_mechanism_active"]}),
                "market_rule_constraints": pd.DataFrame({"constraint_id": ["c1"], "model_mapping": ["renewable_mechanism_active"]}),
            },
        }

    def fake_run_train(context):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        assert payload["stage"] == "训练"
        assert payload["phase_name"] == "HYBRID_PSO_V036 训练"
        return {"model": object(), "device": "cpu"}

    def fake_run_evaluate(context, model=None):
        return {
            "metrics": {"total_procurement_cost": 1.0, "total_profit": 2.0, "cvar99": 3.0},
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
            aggregate = {"window_count": 1.0, "mean_total_procurement_cost": 3.0, "mean_total_profit": 4.0, "mean_cvar99": 5.0}

        return {
            "rolling_summary": _Summary(),
            "rolling_excess_return_metrics": pd.DataFrame(
                {
                    "window_name": ["window_01"],
                    "window_policy_risk_adjusted_sharpe": [1.2],
                    "active_excess_return_persistent": [True],
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
        monkeypatch.delenv("ELEC_RUNTIME_STATUS_PATH", raising=False)

    final_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert final_payload["stage"] == "完成"
    assert (reports_dir / "module1_summary.md").exists()
    assert (reports_dir / "market_mechanism_analysis.md").exists()
    assert (reports_dir / "excess_return_validation_summary.md").exists()


def test_run_train_persists_hybrid_pso_iteration_progress(monkeypatch) -> None:
    output_root = Path(".cache") / "tests" / f"train-progress-{uuid4().hex}"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    models_dir = output_root / "models"
    logs_dir = output_root / "logs"
    for path in [metrics_dir, reports_dir, models_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    status_path = output_root / "logs" / "runtime_status.json"
    context = {
        "config": {
            "version": "v0.44",
            "sample_start": "2025-11-01 00:00:00",
            "sample_end": "2026-03-20 23:45:00",
            "training": {"algorithm": "HYBRID_PSO_V040", "device": "cpu"},
            "hybrid_pso": {
                "upper": {"dimension": 6},
                "lower": {"dimension": 4},
            },
        },
        "split": SimpleNamespace(
            warmup=["2025-12-29"],
            train=["2026-01-05"],
            val=["2026-01-12"],
            test=["2026-01-19"],
        ),
        "bundle": {"tensor_bundle": object()},
        "runtime_status_path": status_path,
        "output_paths": {
            "root": output_root,
            "metrics": metrics_dir,
            "reports": reports_dir,
            "models": models_dir,
            "logs": logs_dir,
        },
    }

    original_subset = train_script.subset_bundle_for_weeks
    original_train_hybrid = train_script.train_hybrid_pso_model
    original_save_model = train_script.save_hybrid_pso_model

    def fake_subset_bundle_for_weeks(bundle, weeks):
        return {
            "tensor_bundle": object(),
            "compiled_parameter_layout": None,
        }

    def fake_save_hybrid_pso_model(model, path):
        Path(path).write_text("{}", encoding="utf-8")

    def fake_train_hybrid_pso_model(tensor_bundle, config, compiled_layout=None, progress_callback=None):
        assert progress_callback is not None
        progress_callback(
            {
                "iteration": 1,
                "iterations": 4,
                "phase_progress": 0.25,
                "best_score": 12.5,
                "mean_score": 20.0,
            }
        )
        snapshot = json.loads(status_path.read_text(encoding="utf-8"))
        assert snapshot["stage"] == "训练"
        assert snapshot["phase_name"] == "HYBRID_PSO_V040 训练"
        assert snapshot["phase_progress"] == 0.25
        assert 0.05 < snapshot["total_progress"] < 0.33
        assert "迭代 1/4" in snapshot["message"]

        progress_callback(
            {
                "iteration": 4,
                "iterations": 4,
                "phase_progress": 1.0,
                "best_score": 8.0,
                "mean_score": 9.5,
            }
        )
        return SimpleNamespace(
            model=SimpleNamespace(best_score=8.0),
            runtime_profile={
                "score_kernel_device": "cpu",
                "upper_particles": 8,
                "lower_particles": 6,
                "iterations": 4,
                "upper_dim": 6,
                "lower_dim": 4,
                "upper_dim_real": 6,
                "lower_dim_real": 4,
            },
            training_trace=pd.DataFrame(
                {
                    "iteration": [1, 2, 3, 4],
                    "best_score": [12.5, 10.0, 9.0, 8.0],
                    "mean_score": [20.0, 15.0, 12.0, 9.5],
                }
            ),
        )

    monkeypatch.setattr(train_script, "subset_bundle_for_weeks", fake_subset_bundle_for_weeks)
    monkeypatch.setattr(train_script, "train_hybrid_pso_model", fake_train_hybrid_pso_model)
    monkeypatch.setattr(train_script, "save_hybrid_pso_model", fake_save_hybrid_pso_model)
    try:
        result = train_script.run_train(context)
    finally:
        monkeypatch.setattr(train_script, "subset_bundle_for_weeks", original_subset)
        monkeypatch.setattr(train_script, "train_hybrid_pso_model", original_train_hybrid)
        monkeypatch.setattr(train_script, "save_hybrid_pso_model", original_save_model)

    final_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert result["device"] == "cpu"
    assert final_payload["stage"] == "训练"
    assert final_payload["phase_name"] == "HYBRID_PSO_V040 训练"
    assert final_payload["phase_progress"] == 1.0
    assert final_payload["total_progress"] == 0.33
    assert final_payload["message"] == "训练完成"
