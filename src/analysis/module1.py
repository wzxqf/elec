from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _policy_dependency_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.endswith("_active") or column.startswith("forward_")]


def enrich_weekly_results(
    weekly_results: pd.DataFrame,
    weekly_metadata: pd.DataFrame,
    weekly_features: pd.DataFrame | None = None,
    policy_state_trace: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = weekly_results.copy()
    enriched["week_start"] = pd.to_datetime(enriched["week_start"])
    join_frames = [weekly_metadata]
    if weekly_features is not None:
        join_frames.append(weekly_features)
    if policy_state_trace is not None:
        join_frames.append(policy_state_trace)
    for frame in join_frames:
        prepared = frame.copy()
        prepared["week_start"] = pd.to_datetime(prepared["week_start"])
        extra_columns = ["week_start"] + [column for column in prepared.columns if column != "week_start" and column not in enriched.columns]
        enriched = enriched.merge(prepared[extra_columns], on="week_start", how="left")
    return enriched


def build_contract_value_weekly(weekly_results: pd.DataFrame) -> pd.DataFrame:
    expected_spot_price = 0.6 * _series(weekly_results, "da_price_mean") + 0.4 * _series(weekly_results, "id_price_mean")
    liquidity_premium = (
        0.35 * _series(weekly_results, "spread_std")
        + 0.15 * _series(weekly_results, "da_price_std")
        + 0.15 * _series(weekly_results, "id_price_std")
        + 5.0 * _series(weekly_results, "extreme_price_spike_flag_w")
        + 3.0 * _series(weekly_results, "extreme_event_flag_w")
    )
    contract_value = expected_spot_price + liquidity_premium
    lock_ratio_proxy = _series(weekly_results, "lock_ratio_proxy_w")
    if "lock_ratio_proxy_w" not in weekly_results.columns:
        lock_ratio_proxy = _series(weekly_results, "contract_position_mwh") / _series(
            weekly_results, "forecast_weekly_net_demand_mwh", default=1.0
        ).clip(lower=1.0)
    curve_match_score = _series(weekly_results, "curve_match_score_w")
    if "curve_match_score_w" not in weekly_results.columns:
        curve_columns = [column for column in weekly_results.columns if column.startswith("contract_curve_h")]
        if curve_columns:
            curve_frame = weekly_results[curve_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            uniform_curve = np.full(len(curve_columns), 1.0 / len(curve_columns))
            curve_match_score = 1.0 - 0.5 * np.abs(curve_frame.to_numpy() - uniform_curve).sum(axis=1)
            curve_match_score = pd.Series(np.clip(curve_match_score, 0.0, 1.0), index=weekly_results.index)
        else:
            curve_match_score = pd.Series(0.0, index=weekly_results.index, dtype="float64")
    stability_score = _series(weekly_results, "stability_score_w")
    if "stability_score_w" not in weekly_results.columns:
        cvar_scale = _series(weekly_results, "cvar99_w")
        cvar_scale = cvar_scale / max(float(cvar_scale.mean()), 1.0e-6)
        stability_score = (
            0.35 * lock_ratio_proxy.clip(lower=0.0, upper=1.0)
            + 0.35 * curve_match_score.clip(lower=0.0, upper=1.0)
            + 0.15 * (1.0 - _series(weekly_results, "hedge_error_w").clip(lower=0.0, upper=1.0))
            + 0.15 * (1.0 / (1.0 + cvar_scale))
        ).clip(lower=0.0, upper=1.0)

    columns: dict[str, Iterable[object]] = {
        "week_start": pd.to_datetime(weekly_results["week_start"]),
        "strategy": weekly_results.get("strategy", pd.Series("unknown", index=weekly_results.index)),
        "expected_spot_price_w": expected_spot_price,
        "liquidity_premium_w": liquidity_premium,
        "contract_value_w": contract_value,
        "lock_ratio_proxy_w": lock_ratio_proxy,
        "curve_match_score_w": curve_match_score,
        "stability_score_w": stability_score,
    }
    return pd.DataFrame(columns)


def build_risk_factor_manifest(weekly_results: pd.DataFrame) -> pd.DataFrame:
    dependency_columns = _policy_dependency_columns(weekly_results)
    dependency_value = ",".join(dependency_columns)
    rows: list[dict[str, object]] = []
    for row in weekly_results.itertuples(index=False):
        week_start = pd.Timestamp(getattr(row, "week_start"))
        forecast = max(float(getattr(row, "forecast_weekly_net_demand_mwh", 0.0)), 1.0)
        actual = float(getattr(row, "actual_weekly_net_demand_mwh", forecast))
        rows.extend(
            [
                {
                    "week_start": week_start,
                    "risk_factor_id": "spot_price_volatility",
                    "risk_factor_name": "现货价格波动",
                    "factor_category": "spot_price_volatility",
                    "factor_value": float(
                        np.mean(
                            [
                                float(getattr(row, "da_price_std", 0.0)),
                                float(getattr(row, "id_price_std", 0.0)),
                                float(getattr(row, "spread_std", 0.0)),
                            ]
                        )
                    ),
                    "direction": "higher_is_riskier",
                    "source_fields": "da_price_std,id_price_std,spread_std",
                    "formula_note": "mean(da_price_std, id_price_std, spread_std)",
                    "policy_state_dependency": dependency_value,
                },
                {
                    "week_start": week_start,
                    "risk_factor_id": "da_id_cross_correlation",
                    "risk_factor_name": "日前-实时价格跨期相关",
                    "factor_category": "cross_market_correlation",
                    "factor_value": float(abs(getattr(row, "da_id_cross_corr_w", 0.0))),
                    "direction": "higher_is_riskier",
                    "source_fields": "da_id_cross_corr_w",
                    "formula_note": "abs(da_id_cross_corr_w)",
                    "policy_state_dependency": dependency_value,
                },
                {
                    "week_start": week_start,
                    "risk_factor_id": "load_forecast_bias",
                    "risk_factor_name": "负荷预测偏差",
                    "factor_category": "load_forecast_bias",
                    "factor_value": float(abs(actual - forecast) / forecast),
                    "direction": "higher_is_riskier",
                    "source_fields": "forecast_weekly_net_demand_mwh,actual_weekly_net_demand_mwh",
                    "formula_note": "abs(actual - forecast) / forecast",
                    "policy_state_dependency": dependency_value,
                },
                {
                    "week_start": week_start,
                    "risk_factor_id": "renewable_output_volatility",
                    "risk_factor_name": "新能源出力波动",
                    "factor_category": "renewable_output_volatility",
                    "factor_value": float(
                        0.7 * float(getattr(row, "renewable_dev_std", 0.0))
                        + 0.3 * abs(float(getattr(row, "prev_renewable_ratio_da_mean", 0.0)))
                    ),
                    "direction": "higher_is_riskier",
                    "source_fields": "renewable_dev_std,prev_renewable_ratio_da_mean",
                    "formula_note": "0.7 * renewable_dev_std + 0.3 * abs(prev_renewable_ratio_da_mean)",
                    "policy_state_dependency": dependency_value,
                },
                {
                    "week_start": week_start,
                    "risk_factor_id": "extreme_event_or_policy_disturbance",
                    "risk_factor_name": "极端事件或制度扰动",
                    "factor_category": "extreme_event_or_policy_disturbance",
                    "factor_value": float(
                        float(getattr(row, "extreme_event_flag_w", 0.0)) + float(getattr(row, "extreme_price_spike_flag_w", 0.0))
                    ),
                    "direction": "higher_is_riskier",
                    "source_fields": "extreme_event_flag_w,extreme_price_spike_flag_w",
                    "formula_note": "extreme_event_flag_w + extreme_price_spike_flag_w",
                    "policy_state_dependency": dependency_value,
                },
            ]
        )
    return pd.DataFrame(rows)
