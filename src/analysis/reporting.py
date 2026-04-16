from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_module1_summary(
    *,
    contract_value_path: Path,
    risk_factor_path: Path,
    contract_value_weekly: pd.DataFrame,
    risk_factor_manifest: pd.DataFrame,
) -> str:
    avg_contract_value = float(contract_value_weekly.get("contract_value_w", pd.Series(dtype="float64")).mean() or 0.0)
    avg_stability = float(contract_value_weekly.get("stability_score_w", pd.Series(dtype="float64")).mean() or 0.0)
    factor_count = int(risk_factor_manifest.get("factor_category", pd.Series(dtype="object")).nunique())
    return "\n".join(
        [
            "# 模块1摘要",
            "",
            f"- 合约价值结果表: {contract_value_path}",
            f"- 风险因子清单: {risk_factor_path}",
            f"- 周度合约价值均值: {avg_contract_value:.4f}",
            f"- 周度稳定性得分均值: {avg_stability:.4f}",
            f"- 风险因子类别数: {factor_count}",
            "",
        ]
    )


def build_market_mechanism_analysis(rule_table: pd.DataFrame, constraints: pd.DataFrame) -> str:
    rule_count = int(len(rule_table))
    constraint_count = int(len(constraints))
    sample_mappings = constraints.get("model_mapping", pd.Series(dtype="object")).head(3).tolist()
    return "\n".join(
        [
            "# 市场机制分析",
            "",
            "## 制度规则 -> 模型字段 -> 结果输出",
            "",
            f"- 正式规则数: {rule_count}",
            f"- 市场约束映射数: {constraint_count}",
            f"- 映射示例: {'; '.join(str(item) for item in sample_mappings) if sample_mappings else '无'}",
            "- 中长期价格通过结算口径影响采购成本与超额收益比较。",
            "- 现货价格与偏差结算通过 15 分钟代理结算进入回测层。",
            "- 制度状态、生效区间和模型映射字段保持同一时点口径。",
            "",
        ]
    )


def build_excess_return_validation_summary(policy_metrics: pd.DataFrame, rolling_metrics: pd.DataFrame | None = None) -> str:
    rolling_metrics = rolling_metrics if rolling_metrics is not None else pd.DataFrame()
    mean_penalty = float(policy_metrics.get("policy_risk_penalty_w", pd.Series(dtype="float64")).mean() or 0.0)
    mean_adjusted = float(policy_metrics.get("policy_risk_adjusted_excess_return_w", pd.Series(dtype="float64")).mean() or 0.0)
    mean_excess = float(policy_metrics.get("excess_profit_w", pd.Series(dtype="float64")).mean() or 0.0)
    if not rolling_metrics.empty and "active_excess_return_persistent" in rolling_metrics.columns:
        persistent = bool(rolling_metrics["active_excess_return_persistent"].astype(bool).any())
        avg_sharpe = float(rolling_metrics.get("window_policy_risk_adjusted_sharpe", pd.Series(dtype="float64")).mean() or 0.0)
    else:
        persistent = mean_adjusted > 0.0
        avg_sharpe = 0.0
    conclusion = "持续跑赢 dynamic_lock_only" if persistent else "未持续跑赢 dynamic_lock_only"
    return "\n".join(
        [
            "# 超额收益验证摘要",
            "",
            f"- 周度超额收益均值: {mean_excess:.4f}",
            f"- 周度政策风险惩罚均值: {mean_penalty:.4f}",
            f"- 周度政策风险调整后超额收益均值: {mean_adjusted:.4f}",
            f"- 政策风险调整后夏普: {avg_sharpe:.4f}",
            f"- 结论: {conclusion}",
            "",
        ]
    )
