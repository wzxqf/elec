from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.reporting import build_excess_return_validation_summary, build_market_mechanism_analysis, build_module1_summary


def test_build_module1_summary_mentions_v036_outputs() -> None:
    text = build_module1_summary(
        contract_value_path=Path("outputs/v0.36/metrics/contract_value_weekly.csv"),
        risk_factor_path=Path("outputs/v0.36/metrics/risk_factor_manifest.csv"),
        contract_value_weekly=pd.DataFrame({"contract_value_w": [325.0], "stability_score_w": [0.8]}),
        risk_factor_manifest=pd.DataFrame({"factor_category": ["spot_price_volatility"]}),
    )

    assert "contract_value_weekly.csv" in text
    assert "risk_factor_manifest.csv" in text


def test_build_market_mechanism_analysis_mentions_rule_mapping() -> None:
    text = build_market_mechanism_analysis(
        rule_table=pd.DataFrame({"rule_id": ["r1"], "state_name": ["renewable_mechanism_active"]}),
        constraints=pd.DataFrame({"constraint_id": ["c1"], "model_mapping": ["renewable_mechanism_active -> policy_state_trace"]}),
    )

    assert "制度规则 -> 模型字段 -> 结果输出" in text


def test_build_excess_return_validation_summary_mentions_conclusion() -> None:
    text = build_excess_return_validation_summary(
        policy_metrics=pd.DataFrame(
            {
                "policy_risk_adjusted_excess_return_w": [3.0, 1.0],
                "excess_profit_w": [4.0, 2.0],
                "policy_risk_penalty_w": [1.0, 1.0],
            }
        ),
        rolling_metrics=pd.DataFrame(
            {
                "window_name": ["window_01"],
                "window_policy_risk_adjusted_sharpe": [1.2],
                "active_excess_return_persistent": [True],
            }
        ),
    )

    assert "政策风险调整后夏普" in text
    assert "持续跑赢" in text
