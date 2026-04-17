from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from src.training.tensor_bundle import TrainingTensorBundle


def _evaluate_strategy(
    tensor_bundle: TrainingTensorBundle,
    *,
    strategy_name: str,
    contract_ratio: float,
    simple_hedge: bool,
    economics: dict[str, Any],
) -> dict[str, float | str]:
    forecast = tensor_bundle.forecast_weekly_load.detach().cpu()
    actual = tensor_bundle.actual_weekly_load.detach().cpu()
    lt_price = tensor_bundle.lt_weekly_price.detach().cpu()
    quarter = tensor_bundle.quarter_price_tensor.detach().cpu()
    quarter_mask = tensor_bundle.quarter_valid_mask.detach().cpu().float()
    hourly = tensor_bundle.hourly_tensor.detach().cpu()
    hour_mask = tensor_bundle.hourly_valid_mask.detach().cpu().float()

    retail_tariff = float(economics.get("retail_tariff_yuan_per_mwh", 430.0))
    friction_unit = float(economics.get("friction_cost_yuan_per_mwh", 1.2))
    profits: list[float] = []
    procurement_costs: list[float] = []
    cvars: list[float] = []

    for week_pos in range(len(tensor_bundle.week_index)):
        valid_intervals = max(int(quarter_mask[week_pos].sum().item()), 1)
        contract_position = float(forecast[week_pos].item()) * contract_ratio
        if simple_hedge:
            valid_hours = max(int(hour_mask[week_pos].sum().item()), 1)
            spread_signal = hourly[week_pos, :, 2] * hour_mask[week_pos]
            hedge_hourly = torch.sign(spread_signal) * (0.08 * float(forecast[week_pos].item()) / valid_hours)
            hedge_hourly = hedge_hourly * hour_mask[week_pos]
        else:
            hedge_hourly = torch.zeros_like(hour_mask[week_pos])
        spot_total = float(torch.abs(hedge_hourly).sum().item())

        scheduled_15m = (contract_position + spot_total) / valid_intervals
        actual_15m = float(actual[week_pos].item()) / valid_intervals
        da_price = quarter[week_pos, :, 0]
        id_price = quarter[week_pos, :, 1]
        valid_mask = quarter_mask[week_pos]
        scheduled_interval = scheduled_15m * valid_mask
        actual_interval = actual_15m * valid_mask
        interval_cost = scheduled_interval * (0.6 * float(lt_price[week_pos].item()) + 0.4 * da_price)
        interval_cost = interval_cost + torch.abs(actual_interval - scheduled_interval) * id_price
        interval_cost = interval_cost * valid_mask
        threshold = torch.quantile(interval_cost, q=0.99)
        tail = interval_cost[interval_cost >= threshold]
        cvar = float(tail.mean().item()) if tail.numel() > 0 else float(interval_cost.max().item())
        procurement_cost = float(interval_cost.sum().item())
        retail_revenue = float(actual[week_pos].item()) * retail_tariff
        friction_cost = spot_total * friction_unit
        profit = retail_revenue - procurement_cost - friction_cost
        procurement_costs.append(procurement_cost)
        profits.append(profit)
        cvars.append(cvar)

    total_profit = float(sum(profits))
    dynamic = {
        "strategy_name": strategy_name,
        "total_procurement_cost": float(sum(procurement_costs)),
        "total_profit": total_profit,
        "mean_profit": float(pd.Series(profits).mean() if profits else 0.0),
        "weekly_profit_volatility": float(pd.Series(profits).std(ddof=0) if len(profits) > 1 else 0.0),
        "cvar99": float(pd.Series(cvars).mean() if cvars else 0.0),
        "max_drawdown": float(_max_drawdown(pd.Series(profits))) if profits else 0.0,
    }
    return dynamic


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cumulative = series.cumsum()
    return float((cumulative.cummax() - cumulative).max())


def evaluate_benchmark_strategies(bundle: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    tensor_bundle: TrainingTensorBundle = bundle["tensor_bundle"]
    economics = config.get("economics", {})
    rows = [
        _evaluate_strategy(tensor_bundle, strategy_name="dynamic_lock_only", contract_ratio=0.55, simple_hedge=False, economics=economics),
        _evaluate_strategy(tensor_bundle, strategy_name="fixed_holding_60", contract_ratio=0.60, simple_hedge=False, economics=economics),
        _evaluate_strategy(tensor_bundle, strategy_name="static_no_spot_adjustment", contract_ratio=0.50, simple_hedge=False, economics=economics),
        _evaluate_strategy(tensor_bundle, strategy_name="simple_rolling_hedge", contract_ratio=0.55, simple_hedge=True, economics=economics),
    ]
    result = pd.DataFrame(rows)
    baseline_profit = float(result.loc[result["strategy_name"] == "dynamic_lock_only", "total_profit"].iloc[0])
    result["profit_delta_vs_dynamic_lock_only"] = result["total_profit"] - baseline_profit
    return result


def build_benchmark_summary_markdown(benchmark_metrics: pd.DataFrame) -> str:
    lines = [
        "# 基准策略比较",
        "",
    ]
    for row in benchmark_metrics.itertuples(index=False):
        lines.append(
            f"- {row.strategy_name}: total_profit={float(row.total_profit):.2f}, cvar99={float(row.cvar99):.2f}, delta_vs_dynamic_lock_only={float(getattr(row, 'profit_delta_vs_dynamic_lock_only', 0.0)):.2f}"
        )
    lines.append("")
    return "\n".join(lines)

