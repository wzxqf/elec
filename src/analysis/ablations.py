from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.report_contracts import build_summary_scope_lines, infer_date_range
from src.backtest.materialize import materialize_particle_pair
from src.training.tensor_bundle import compile_training_tensor_bundle


def _bundle_copy(bundle: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in bundle.items():
        if isinstance(value, pd.DataFrame):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def _zero_state_columns(bundle: dict[str, Any]) -> dict[str, Any]:
    copied = _bundle_copy(bundle)
    weekly_patterns = ("extreme_", "da_id_cross_corr", "business_hour", "peak_hour", "valley_hour", "price_spread_abs", "renewable_dev_abs")
    hourly_patterns = ("business_hour", "peak_hour", "valley_hour", "_abs", "_sign")
    weekly = copied["weekly_features"].copy()
    for column in weekly.columns:
        if any(pattern in column for pattern in weekly_patterns):
            weekly[column] = 0.0
    copied["weekly_features"] = weekly
    hourly = copied["hourly"].copy()
    for column in hourly.columns:
        if any(pattern in column for pattern in hourly_patterns):
            hourly[column] = 0.0
    copied["hourly"] = hourly
    copied["tensor_bundle"] = compile_training_tensor_bundle(copied, device=bundle["tensor_bundle"].device)
    return copied


def evaluate_ablation_variants(bundle: dict[str, Any], model: Any, config: dict[str, Any]) -> pd.DataFrame:
    full = materialize_particle_pair(
        tensor_bundle=bundle["tensor_bundle"],
        upper_particle=model.upper_best,
        lower_particle=model.lower_best,
        strategy_name="full_model",
        config=config,
        compiled_layout=bundle.get("compiled_parameter_layout"),
    )
    no_policy_bundle = _bundle_copy(bundle)
    no_policy_bundle.pop("feasible_domain", None)
    no_policy_bundle["tensor_bundle"] = compile_training_tensor_bundle(no_policy_bundle, device=bundle["tensor_bundle"].device)
    no_policy = materialize_particle_pair(
        tensor_bundle=no_policy_bundle["tensor_bundle"],
        upper_particle=model.upper_best,
        lower_particle=model.lower_best,
        strategy_name="no_policy_projection",
        config=config,
        compiled_layout=bundle.get("compiled_parameter_layout"),
    )
    no_state_bundle = _zero_state_columns(bundle)
    no_state = materialize_particle_pair(
        tensor_bundle=no_state_bundle["tensor_bundle"],
        upper_particle=model.upper_best,
        lower_particle=model.lower_best,
        strategy_name="no_state_enhancement",
        config=config,
        compiled_layout=bundle.get("compiled_parameter_layout"),
    )
    no_layout = materialize_particle_pair(
        tensor_bundle=bundle["tensor_bundle"],
        upper_particle=model.upper_best,
        lower_particle=model.lower_best,
        strategy_name="no_parameter_layout_enhancement",
        config=config,
        compiled_layout=None,
    )
    rows = []
    for variant_name, result in [
        ("full_model", full),
        ("no_policy_projection", no_policy),
        ("no_state_enhancement", no_state),
        ("no_parameter_layout_enhancement", no_layout),
    ]:
        rows.append({"variant_name": variant_name, **result.metrics})
    return pd.DataFrame(rows)


def build_ablation_summary_markdown(
    ablation_metrics: pd.DataFrame,
    *,
    sample_scope: str = "rolling_backtest_windows",
    week_count: int | None = None,
    aggregation_method: str = "aggregate_over_rolling_windows",
    date_range: str | None = None,
) -> str:
    full_profit = float(ablation_metrics.loc[ablation_metrics["variant_name"] == "full_model", "total_profit"].iloc[0]) if not ablation_metrics.empty else 0.0
    date_range = date_range or infer_date_range(ablation_metrics)
    week_count = week_count if week_count is not None else int(len(ablation_metrics))
    lines = [
        "# 消融总结",
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
    for row in ablation_metrics.itertuples(index=False):
        lines.append(
            f"- {row.variant_name}: total_profit={float(row.total_profit):.2f}, cvar99={float(row.cvar99):.2f}, profit_delta_vs_full={float(row.total_profit) - full_profit:.2f}"
        )
    lines.append("")
    return "\n".join(lines)
