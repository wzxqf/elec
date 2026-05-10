> version: v0.47
> experiment_id: v0.47-fdb00545
> config_hash: fdb00545c6611582b09096325a7dba3890362251cb1e482326ef83f8c8ee9b0a
> run_timestamp: 2026-05-10T20:01:08+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 可行域摘要

- summary_scope: static_policy_bound_inventory
- 可行域周数: 21
- policy_tightening_week_count: 21
- default_bound_week_count: 0
- projection_clip_mean: 0.0000
- projection_clip_max: 0.0000
- 结算模式: previous_week_da_proxy, linked_40_60
- runtime_projection_reference: reports/constraint_activation_report.md, metrics/ablation_metrics.csv
- note: 本摘要仅统计编译期静态边界，不统计运行时投影/裁剪触发次数。
