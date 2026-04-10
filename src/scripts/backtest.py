from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.train_ppo import evaluate_policy, load_model, train_model
from src.backtest.benchmarks import build_benchmark_actions
from src.backtest.simulator import simulate_strategy
from src.scripts.common import prepare_project_context
from src.utils.io import save_json, save_markdown
from src.utils.plotting import save_bar_plot, save_line_plot, save_multi_line_plot


def _ensure_main_model(context: dict[str, Any]):
    model_path = context["output_paths"]["models"] / "ppo_elec_env.zip"
    if model_path.exists():
        return load_model(model_path, context["config"])
    training = train_model(
        bundle=context["bundle"],
        train_months=context["train_sequence"],
        val_months=context["split"].val,
        config=context["config"],
        output_paths=context["output_paths"],
    )
    return training["model"]


def _benchmark_comparison_frame(results: dict[str, dict]) -> pd.DataFrame:
    metrics_frame = pd.DataFrame([payload["metrics"] for payload in results.values()]).sort_values(
        "cumulative_procurement_cost"
    )
    base_cost = float(metrics_frame.loc[metrics_frame["strategy"] == "fixed_lock", "cumulative_procurement_cost"].iloc[0])
    metrics_frame["cost_savings_vs_fixed_lock"] = base_cost - metrics_frame["cumulative_procurement_cost"]
    return metrics_frame


def _save_core_figures(results: dict[str, dict], output_paths: dict[str, Path]) -> None:
    monthly = pd.concat([payload["monthly_results"] for payload in results.values()], ignore_index=True)
    interval = pd.concat([payload["interval_results"] for payload in results.values()], ignore_index=True)

    monthly_cost = monthly.pivot(index="month", columns="strategy", values="procurement_cost_m").sort_index()
    save_bar_plot(
        monthly_cost,
        output_paths["figures"] / "monthly_cost_compare.png",
        title="Monthly Procurement Cost Comparison",
        xlabel="Month",
        ylabel="Cost",
    )

    equity_curve = (
        interval.pivot_table(index="datetime", columns="strategy", values="total_cost_with_penalties", aggfunc="sum")
        .sort_index()
        .cumsum()
    )
    save_multi_line_plot(
        equity_curve,
        output_paths["figures"] / "equity_curve.png",
        title="Cumulative Cost Curve",
        xlabel="Datetime",
        ylabel="Cumulative Cost",
    )

    hedge_error = monthly.pivot(index="month", columns="strategy", values="hedge_error_m").sort_index()
    save_multi_line_plot(
        hedge_error,
        output_paths["figures"] / "hedge_error_curve.png",
        title="Monthly Hedge Error",
        xlabel="Month",
        ylabel="Hedge Error",
    )


def run_sensitivity_analysis(
    context: dict[str, Any],
    ppo_actions: dict[pd.Timestamp, tuple[float, float]],
) -> pd.DataFrame:
    config = context["config"]
    months = context["split"].test
    rows = []

    for value in config["sensitivity"]["lambda_risk"]:
        config_variant = deepcopy(config)
        config_variant["cost"]["lambda_risk"] = float(value)
        result = simulate_strategy(context["bundle"], months, ppo_actions, config_variant, "ppo_sensitivity_lambda_risk")
        rows.append({"factor": "lambda_risk", "value": float(value), **result["metrics"]})

    for value in config["sensitivity"]["market_vol_scale"]:
        result = simulate_strategy(
            context["bundle"],
            months,
            ppo_actions,
            config,
            "ppo_sensitivity_market_vol",
            market_vol_scale=float(value),
        )
        rows.append({"factor": "market_vol_scale", "value": float(value), **result["metrics"]})

    for value in config["sensitivity"]["price_cap_multiplier"]:
        result = simulate_strategy(
            context["bundle"],
            months,
            ppo_actions,
            config,
            "ppo_sensitivity_price_cap",
            price_cap_multiplier=float(value),
        )
        rows.append({"factor": "price_cap_multiplier", "value": float(value), **result["metrics"]})

    frame = pd.DataFrame(rows)
    frame.to_csv(context["output_paths"]["metrics"] / "sensitivity_results.csv", index=False)

    for factor in frame["factor"].unique():
        subset = frame.loc[frame["factor"] == factor].sort_values("value")
        save_line_plot(
            subset["value"],
            subset["cumulative_procurement_cost"],
            context["output_paths"]["figures"] / f"sensitivity_{factor}.png",
            title=f"Sensitivity: {factor}",
            xlabel=factor,
            ylabel="Cumulative Procurement Cost",
        )
    return frame


def run_robustness_analysis(
    context: dict[str, Any],
    model,
    ppo_actions: dict[pd.Timestamp, tuple[float, float]],
) -> pd.DataFrame:
    config = context["config"]
    rows = []

    for shift in config["robustness"]["contract_ratio_shift"]:
        shifted_actions = {
            month: (float(np.clip(action[0] + shift, 0.0, 1.0)), action[1])
            for month, action in ppo_actions.items()
        }
        result = simulate_strategy(context["bundle"], context["split"].test, shifted_actions, config, "ppo_contract_shift")
        rows.append({"experiment": "contract_ratio_shift", "value": float(shift), **result["metrics"]})

    all_months = sorted(set(context["split"].train + context["split"].val + context["split"].test))
    all_month_actions = evaluate_policy(model, context["bundle"], all_months, config, "ppo_all_months")["actions"]
    for cutoff in config["robustness"]["policy_cutoffs"]:
        cutoff_ts = pd.Timestamp(cutoff)
        before = [month for month in all_months if month < cutoff_ts.to_period("M").to_timestamp()]
        after = [month for month in all_months if month >= cutoff_ts.to_period("M").to_timestamp()]
        if before:
            result_before = simulate_strategy(context["bundle"], before, all_month_actions, config, "ppo_policy_before")
            rows.append(
                {
                    "experiment": "policy_split_before",
                    "value": cutoff,
                    **result_before["metrics"],
                }
            )
        if after:
            result_after = simulate_strategy(context["bundle"], after, all_month_actions, config, "ppo_policy_after")
            rows.append(
                {
                    "experiment": "policy_split_after",
                    "value": cutoff,
                    **result_after["metrics"],
                }
            )

    for scale in config["robustness"]["forecast_error_scale"]:
        result = simulate_strategy(
            context["bundle"],
            context["split"].test,
            ppo_actions,
            config,
            "ppo_forecast_error",
            forecast_error_scale=float(scale),
        )
        rows.append({"experiment": "forecast_error_scale", "value": float(scale), **result["metrics"]})

    frame = pd.DataFrame(rows)
    frame.to_csv(context["output_paths"]["metrics"] / "robustness_results.csv", index=False)

    contract = frame.loc[frame["experiment"] == "contract_ratio_shift"].sort_values("value")
    save_line_plot(
        contract["value"],
        contract["cumulative_procurement_cost"],
        context["output_paths"]["figures"] / "robustness_contract_ratio_shift.png",
        title="Robustness: Contract Ratio Shift",
        xlabel="Lock Ratio Shift",
        ylabel="Cumulative Procurement Cost",
    )

    forecast = frame.loc[frame["experiment"] == "forecast_error_scale"].sort_values("value")
    save_line_plot(
        forecast["value"],
        forecast["cumulative_procurement_cost"],
        context["output_paths"]["figures"] / "robustness_forecast_error_scale.png",
        title="Robustness: Forecast Error Scale",
        xlabel="Forecast Error Scale",
        ylabel="Cumulative Procurement Cost",
    )

    policy = frame.loc[frame["experiment"].str.contains("policy_split")].copy()
    policy["label"] = policy["experiment"] + "_" + policy["value"].astype(str)
    policy = policy.set_index("label")[["cumulative_procurement_cost"]]
    save_bar_plot(
        policy,
        context["output_paths"]["figures"] / "robustness_policy_split.png",
        title="Robustness: Policy Split",
        xlabel="Segment",
        ylabel="Cumulative Procurement Cost",
    )
    return frame


def run_hparam_search(context: dict[str, Any]) -> pd.DataFrame:
    config = context["config"]
    search_cfg = config["hparam_search"]
    rng = np.random.default_rng(int(config["seed"]))
    candidates = list(search_cfg["manual_grid"])

    for _ in range(int(search_cfg["random_search"]["n_trials"])):
        candidates.append(
            {
                "learning_rate": float(rng.uniform(*search_cfg["random_search"]["learning_rate_range"])),
                "n_steps": 128,
                "batch_size": 32,
                "ent_coef": float(rng.uniform(*search_cfg["random_search"]["ent_coef_range"])),
                "lambda_risk": float(rng.uniform(*search_cfg["random_search"]["lambda_risk_range"])),
                "lambda_he": float(rng.uniform(*search_cfg["random_search"]["lambda_he_range"])),
                "hedge_intensity_scale": float(rng.uniform(*search_cfg["random_search"]["hedge_intensity_scale_range"])),
            }
        )

    rows = []
    for trial_id, candidate in enumerate(candidates, start=1):
        trial_config = deepcopy(config)
        trial_config["learning_rate"] = float(candidate["learning_rate"])
        trial_config["n_steps"] = int(candidate["n_steps"])
        trial_config["batch_size"] = int(candidate["batch_size"])
        trial_config["ent_coef"] = float(candidate["ent_coef"])
        trial_config["cost"]["lambda_risk"] = float(candidate["lambda_risk"])
        trial_config["cost"]["lambda_he"] = float(candidate["lambda_he"])

        training = train_model(
            bundle=context["bundle"],
            train_months=context["train_sequence"],
            val_months=context["split"].val,
            config=trial_config,
            output_paths=context["output_paths"],
            run_name=f"hparam_trial_{trial_id}",
            total_timesteps_override=int(search_cfg["total_timesteps"]),
            save_plots=False,
        )
        validation = evaluate_policy(
            model=training["model"],
            bundle=context["bundle"],
            months=context["split"].val,
            config=trial_config,
            strategy_name=f"hparam_trial_{trial_id}_validation",
        )
        validation = simulate_strategy(
            bundle=context["bundle"],
            months=context["split"].val,
            action_source=validation["actions"],
            config=trial_config,
            strategy_name=f"hparam_trial_{trial_id}_validation",
            hedge_intensity_scale=float(candidate["hedge_intensity_scale"]),
        )
        metrics = validation["metrics"]
        rows.append(
            {
                "trial_id": trial_id,
                **candidate,
                "val_cumulative_procurement_cost": metrics["cumulative_procurement_cost"],
                "val_cvar95": metrics["cvar95"],
                "val_mean_reward": metrics["mean_reward"],
            }
        )

    frame = pd.DataFrame(rows).sort_values("val_cumulative_procurement_cost").reset_index(drop=True)
    frame.to_csv(context["output_paths"]["metrics"] / "hparam_search_results.csv", index=False)

    top = frame.head(5)
    summary_lines = [
        "# Hyperparameter Search Summary",
        "",
        "## Top Trials",
        "",
    ]
    for _, row in top.iterrows():
        summary_lines.append(
            "- Trial {trial_id}: lr={learning_rate:.6f}, n_steps={n_steps}, batch_size={batch_size}, "
            "ent_coef={ent_coef:.4f}, lambda_risk={lambda_risk:.4f}, lambda_he={lambda_he:.2f}, "
            "hedge_intensity_scale={hedge_intensity_scale:.3f}, val_cost={val_cumulative_procurement_cost:.2f}".format(
                **row.to_dict()
            )
        )
    save_markdown("\n".join(summary_lines) + "\n", context["output_paths"]["reports"] / "hparam_search_summary.md")
    return frame


def run_backtest(context: dict[str, Any], model=None) -> dict[str, Any]:
    logger = context["logger"]
    logger.info("Start backtest module.")
    model = model or _ensure_main_model(context)

    ppo_result = evaluate_policy(
        model=model,
        bundle=context["bundle"],
        months=context["split"].test,
        config=context["config"],
        strategy_name="ppo",
    )
    benchmark_actions = build_benchmark_actions(
        months=context["split"].test,
        monthly_features=context["bundle"]["monthly_features"],
        config=context["config"],
    )
    results = {"ppo": ppo_result}
    for strategy, actions in benchmark_actions.items():
        results[strategy] = simulate_strategy(
            bundle=context["bundle"],
            months=context["split"].test,
            action_source=actions,
            config=context["config"],
            strategy_name=strategy,
        )

    metrics_frame = _benchmark_comparison_frame(results)
    metrics_frame.to_csv(context["output_paths"]["metrics"] / "benchmark_comparison.csv", index=False)
    save_json(
        {
            "main_strategy": results["ppo"]["metrics"],
            "benchmarks": metrics_frame.to_dict(orient="records"),
        },
        context["output_paths"]["metrics"] / "backtest_metrics.json",
    )
    pd.concat([payload["monthly_results"] for payload in results.values()], ignore_index=True).to_csv(
        context["output_paths"]["metrics"] / "backtest_monthly_results.csv",
        index=False,
    )
    _save_core_figures(results, context["output_paths"])

    sensitivity = run_sensitivity_analysis(context, ppo_result["actions"])
    robustness = run_robustness_analysis(context, model, ppo_result["actions"])
    hparam_search = run_hparam_search(context)

    logger.info("Backtest module complete.")
    return {
        "results": results,
        "metrics_frame": metrics_frame,
        "sensitivity": sensitivity,
        "robustness": robustness,
        "hparam_search": hparam_search,
    }


def main() -> dict[str, Any]:
    context = prepare_project_context("/Users/dk/py/elec")
    return run_backtest(context)


if __name__ == "__main__":
    main()
