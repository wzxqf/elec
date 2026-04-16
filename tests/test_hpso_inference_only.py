from __future__ import annotations

import inspect

import src.scripts.backtest as backtest_script
import src.scripts.evaluate as evaluate_script


def test_evaluate_has_param_policy_inference_path() -> None:
    source = inspect.getsource(evaluate_script.run_evaluate)

    assert "HPSO_PARAM_POLICY" in source
    assert "simulate_param_policy_strategy" in source


def test_backtest_has_param_policy_inference_path() -> None:
    source = inspect.getsource(backtest_script.run_backtest)

    assert "HPSO_PARAM_POLICY" in source
    assert "simulate_param_policy_strategy" in source
