> version: v0.45-param-opt-balanced
> experiment_id: v0.45-param-opt-balanced-5c06ad8b
> config_hash: 5c06ad8b002c173cae2144d71091a54dcba18a6cf06c30babd131c748cc85a4b
> run_timestamp: 2026-04-20T16:29:07+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 可行域摘要

- summary_scope: static_policy_bound_inventory
- 可行域周数: 21
- policy_tightening_week_count: 0
- default_bound_week_count: 21
- projection_clip_mean: 0.0000
- projection_clip_max: 0.0000
- 结算模式: previous_week_da_proxy
- runtime_projection_reference: reports/constraint_activation_report.md, metrics/ablation_metrics.csv
- note: 本摘要仅统计编译期静态边界，不统计运行时投影/裁剪触发次数。
