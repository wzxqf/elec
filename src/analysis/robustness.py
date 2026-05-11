from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pandas as pd
import torch

from src.analysis.report_contracts import build_summary_scope_lines, infer_date_range
from src.backtest.materialize import materialize_particle_pair
from src.model_layout.schema import CompiledParameterLayout
from src.training.tensor_bundle import TrainingTensorBundle


def _rank_robustness_rows(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return result
    result = result.copy()
    result["robustness_rank"] = (
        result["total_profit"].rank(method="dense", ascending=False) + result["mean_cvar99"].rank(method="dense", ascending=True)
    )
    return result.sort_values(["robustness_rank", "scenario_name"]).reset_index(drop=True)


def _summarize_materialized_scenario(
    *,
    scenario_group: str,
    scenario_name: str,
    scenario_value: float | str,
    result,
    cutoff: pd.Timestamp | None = None,
) -> dict[str, object]:
    weekly = result.weekly_results.copy()
    settlement = result.settlement_results.copy()
    if cutoff is not None:
        weekly = weekly.loc[pd.to_datetime(weekly["week_start"]) >= cutoff]
        settlement = settlement.loc[pd.to_datetime(settlement["week_start"]) >= cutoff]
    return {
        "scenario_group": scenario_group,
        "scenario_name": scenario_name,
        "scenario_value": scenario_value,
        "scenario_settlement_method": "rerun_materialize_particle_pair",
        "total_profit": float(weekly["profit_w"].sum()) if not weekly.empty else 0.0,
        "total_procurement_cost": float(weekly["procurement_cost_w"].sum()) if not weekly.empty else 0.0,
        "mean_cvar99": float(weekly["cvar99_w"].mean()) if not weekly.empty else 0.0,
        "mean_hedge_error": float(weekly["hedge_error_w"].mean()) if not weekly.empty else 0.0,
        "week_count": int(weekly["week_start"].nunique()) if "week_start" in weekly else 0,
        "settlement_record_count": int(len(settlement)),
    }


def _scale_actual_settlement_load(tensor_bundle: TrainingTensorBundle, scale: float) -> TrainingTensorBundle:
    scale_value = max(float(scale), 0.0)
    actual_weekly_load = torch.clamp_min(tensor_bundle.actual_weekly_load * scale_value, 0.0)
    quarter_tensor = tensor_bundle.quarter_price_tensor.clone()
    if "net_load_id_mwh" in tensor_bundle.quarter_feature_columns:
        actual_idx = tensor_bundle.quarter_feature_columns.index("net_load_id_mwh")
        valid_mask = tensor_bundle.quarter_valid_mask.to(quarter_tensor.device).float()
        quarter_tensor[..., actual_idx] = torch.clamp_min(quarter_tensor[..., actual_idx] * scale_value, 0.0) * valid_mask
    return replace(tensor_bundle, actual_weekly_load=actual_weekly_load, quarter_price_tensor=quarter_tensor)


def _rerun_materialized_scenarios(
    *,
    config: dict[str, Any],
    tensor_bundle: TrainingTensorBundle,
    upper_particle: list[float] | torch.Tensor,
    lower_particle: list[float] | torch.Tensor,
    compiled_layout: CompiledParameterLayout | None,
    window_name: str | None,
) -> pd.DataFrame:
    robustness_cfg = config.get("robustness", {})
    rows: list[dict[str, object]] = []
    base_contract_ratio = float(config.get("score_kernel", {}).get("contract_position_base_ratio", 0.60))

    for shift in robustness_cfg.get("contract_ratio_shift", []):
        scenario_config = deepcopy(config)
        score_cfg = scenario_config.setdefault("score_kernel", {})
        score_cfg["contract_position_base_ratio"] = min(max(base_contract_ratio + float(shift), 0.0), 1.0)
        scenario_result = materialize_particle_pair(
            tensor_bundle=tensor_bundle,
            upper_particle=upper_particle,
            lower_particle=lower_particle,
            strategy_name=f"robustness_contract_ratio_shift_{float(shift):+.2f}",
            config=scenario_config,
            compiled_layout=compiled_layout,
        )
        rows.append(
            _summarize_materialized_scenario(
                scenario_group="contract_ratio_shift",
                scenario_name=f"contract_ratio_shift_{float(shift):+.2f}",
                scenario_value=float(shift),
                result=scenario_result,
            )
        )

    for scale in robustness_cfg.get("forecast_error_scale", []):
        scenario_bundle = _scale_actual_settlement_load(tensor_bundle, float(scale))
        scenario_result = materialize_particle_pair(
            tensor_bundle=scenario_bundle,
            upper_particle=upper_particle,
            lower_particle=lower_particle,
            strategy_name=f"robustness_forecast_error_scale_{float(scale):.2f}",
            config=config,
            compiled_layout=compiled_layout,
        )
        rows.append(
            _summarize_materialized_scenario(
                scenario_group="forecast_error_scale",
                scenario_name=f"forecast_error_scale_{float(scale):.2f}",
                scenario_value=float(scale),
                result=scenario_result,
            )
        )

    for cutoff in robustness_cfg.get("policy_cutoffs", []):
        cutoff_ts = pd.Timestamp(cutoff)
        scenario_result = materialize_particle_pair(
            tensor_bundle=tensor_bundle,
            upper_particle=upper_particle,
            lower_particle=lower_particle,
            strategy_name=f"robustness_policy_cutoff_{cutoff_ts.date()}",
            config=config,
            compiled_layout=compiled_layout,
        )
        rows.append(
            _summarize_materialized_scenario(
                scenario_group="policy_cutoff",
                scenario_name=f"policy_cutoff_{cutoff_ts.date()}",
                scenario_value=str(cutoff_ts.date()),
                result=scenario_result,
                cutoff=cutoff_ts,
            )
        )

    result = pd.DataFrame(rows)
    if window_name and not result.empty:
        result["window_name"] = window_name
    return _rank_robustness_rows(result)


def run_robustness_analysis(
    *,
    weekly_results: pd.DataFrame,
    config: dict,
    tensor_bundle: TrainingTensorBundle | None = None,
    upper_particle: list[float] | torch.Tensor | None = None,
    lower_particle: list[float] | torch.Tensor | None = None,
    compiled_layout: CompiledParameterLayout | None = None,
    window_name: str | None = None,
) -> pd.DataFrame:
    if tensor_bundle is not None and upper_particle is not None and lower_particle is not None:
        return _rerun_materialized_scenarios(
            config=config,
            tensor_bundle=tensor_bundle,
            upper_particle=upper_particle,
            lower_particle=lower_particle,
            compiled_layout=compiled_layout,
            window_name=window_name,
        )

    robustness_cfg = config.get("robustness", {})
    frame = weekly_results.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    rows: list[dict[str, object]] = []

    for shift in robustness_cfg.get("contract_ratio_shift", []):
        shifted_profit = frame["profit_w"] - frame["procurement_cost_w"] * abs(float(shift)) * 0.05
        shifted_cvar = frame["cvar99_w"] * (1.0 + abs(float(shift)) * 0.5)
        rows.append(
            {
                "scenario_group": "contract_ratio_shift",
                "scenario_name": f"contract_ratio_shift_{float(shift):+.2f}",
                "scenario_value": float(shift),
                "scenario_settlement_method": "post_hoc_sensitivity_fallback",
                "total_profit": float(shifted_profit.sum()),
                "total_procurement_cost": float(frame["procurement_cost_w"].sum()),
                "mean_cvar99": float(shifted_cvar.mean()),
                "mean_hedge_error": float(frame["hedge_error_w"].mean()) if "hedge_error_w" in frame else 0.0,
                "week_count": int(frame["week_start"].nunique()),
                "settlement_record_count": 0,
            }
        )

    for scale in robustness_cfg.get("forecast_error_scale", []):
        scaled_profit = frame["profit_w"] - frame["procurement_cost_w"] * abs(float(scale) - 1.0) * 0.10
        scaled_cvar = frame["cvar99_w"] * float(scale)
        rows.append(
            {
                "scenario_group": "forecast_error_scale",
                "scenario_name": f"forecast_error_scale_{float(scale):.2f}",
                "scenario_value": float(scale),
                "scenario_settlement_method": "post_hoc_sensitivity_fallback",
                "total_profit": float(scaled_profit.sum()),
                "total_procurement_cost": float(frame["procurement_cost_w"].sum()),
                "mean_cvar99": float(scaled_cvar.mean()),
                "mean_hedge_error": float(frame["hedge_error_w"].mean()) if "hedge_error_w" in frame else 0.0,
                "week_count": int(frame["week_start"].nunique()),
                "settlement_record_count": 0,
            }
        )

    for cutoff in robustness_cfg.get("policy_cutoffs", []):
        cutoff_ts = pd.Timestamp(cutoff)
        subset = frame.loc[frame["week_start"] >= cutoff_ts]
        rows.append(
            {
                "scenario_group": "policy_cutoff",
                "scenario_name": f"policy_cutoff_{cutoff_ts.date()}",
                "scenario_value": str(cutoff_ts.date()),
                "scenario_settlement_method": "post_hoc_sensitivity_fallback",
                "total_profit": float(subset["profit_w"].sum()) if not subset.empty else 0.0,
                "total_procurement_cost": float(subset["procurement_cost_w"].sum()) if not subset.empty else 0.0,
                "mean_cvar99": float(subset["cvar99_w"].mean()) if not subset.empty else 0.0,
                "mean_hedge_error": float(subset["hedge_error_w"].mean()) if "hedge_error_w" in subset and not subset.empty else 0.0,
                "week_count": int(subset["week_start"].nunique()) if not subset.empty else 0,
                "settlement_record_count": 0,
            }
        )

    result = pd.DataFrame(rows)
    return _rank_robustness_rows(result)


def aggregate_robustness_scenario_rows(scenario_rows: pd.DataFrame) -> pd.DataFrame:
    if scenario_rows.empty:
        return scenario_rows
    group_columns = ["scenario_group", "scenario_name", "scenario_value", "scenario_settlement_method"]
    aggregation = {
        "total_profit": "sum",
        "total_procurement_cost": "sum",
        "mean_cvar99": "mean",
        "mean_hedge_error": "mean",
        "week_count": "sum",
        "settlement_record_count": "sum",
    }
    result = scenario_rows.groupby(group_columns, as_index=False).agg(aggregation)
    return _rank_robustness_rows(result)


def build_robustness_summary_markdown(
    robustness_metrics: pd.DataFrame,
    *,
    sample_scope: str = "rolling_backtest_windows",
    week_count: int | None = None,
    aggregation_method: str = "scenario_rerun_settlement",
    date_range: str | None = None,
) -> str:
    if robustness_metrics.empty:
        lines = ["# 稳健性总结", "", *build_summary_scope_lines(sample_scope=sample_scope, week_count=week_count, aggregation_method=aggregation_method, date_range=date_range or "n/a"), "- 无可用情景。", ""]
        return "\n".join(lines)

    date_range = date_range or infer_date_range(robustness_metrics)
    week_count = week_count if week_count is not None else int(len(robustness_metrics))
    lines = [
        "# 稳健性总结",
        "",
    ]
    lines.extend(
        build_summary_scope_lines(
            sample_scope=sample_scope,
            week_count=week_count,
            aggregation_method=aggregation_method,
            date_range=date_range,
        )
    )
    for row in robustness_metrics.itertuples(index=False):
        lines.append(
            f"- {row.scenario_name}: group={row.scenario_group}, method={getattr(row, 'scenario_settlement_method', 'n/a')}, total_profit={float(row.total_profit):.2f}, mean_cvar99={float(row.mean_cvar99):.2f}, robustness_rank={float(row.robustness_rank):.2f}"
        )
    lines.append("")
    return "\n".join(lines)
