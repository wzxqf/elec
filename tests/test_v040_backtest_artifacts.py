from __future__ import annotations

import pandas as pd

from src.analysis.ablations import build_ablation_summary_markdown
from src.analysis.benchmarks import build_benchmark_summary_markdown
from src.analysis.constraint_reporting import build_constraint_activation_report_markdown
from src.analysis.robustness import build_robustness_summary_markdown


def test_v040_reporting_builders_emit_required_sections() -> None:
    benchmark_metrics = pd.DataFrame(
        {
            "strategy_name": ["hybrid_pso", "dynamic_lock_only"],
            "total_profit": [120.0, 100.0],
            "cvar99": [60.0, 65.0],
            "profit_delta_vs_dynamic_lock_only": [20.0, 0.0],
        }
    )
    ablation_metrics = pd.DataFrame(
        {
            "variant_name": ["full_model", "no_policy_projection", "no_state_enhancement", "no_parameter_layout_enhancement"],
            "total_profit": [120.0, 90.0, 95.0, 85.0],
            "cvar99": [60.0, 80.0, 75.0, 78.0],
        }
    )
    weekly_results = pd.DataFrame(
        {
            "bound_reason_code": ["lt_price_linked", "default", "ancillary_tight"],
            "feasible_domain_triggered_w": [1.0, 0.0, 1.0],
            "feasible_domain_clip_gap_w": [10.0, 0.0, 6.0],
            "projection_target_field": ["contract_adjustment_mwh", "", "exposure_band_mwh"],
            "projection_rule_name": ["policy_linked_ratio_cap", "", "policy_ancillary_tight_cap"],
            "projection_before": [100.0, 0.0, 80.0],
            "projection_after": [60.0, 0.0, 40.0],
        }
    )
    robustness_metrics = pd.DataFrame(
        {
            "scenario_group": ["policy_cutoff"],
            "scenario_name": ["policy_cutoff_2026-02-01"],
            "total_profit": [150.0],
            "mean_cvar99": [55.0],
            "robustness_rank": [1.0],
        }
    )

    benchmark_text = build_benchmark_summary_markdown(benchmark_metrics)
    ablation_text = build_ablation_summary_markdown(ablation_metrics)
    constraint_text = build_constraint_activation_report_markdown(weekly_results)
    robustness_text = build_robustness_summary_markdown(robustness_metrics)

    assert "dynamic_lock_only" in benchmark_text
    assert "sample_scope" in benchmark_text
    assert "aggregation_method" in benchmark_text
    assert "no_policy_projection" in ablation_text
    assert "sample_scope" in ablation_text
    assert "约束激活" in constraint_text
    assert "policy_tightening_trigger_count" in constraint_text
    assert "default_projection_trigger_count" in constraint_text
    assert "projection_target_field" in constraint_text
    assert "sample_scope" in robustness_text
