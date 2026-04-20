> version: v0.44-param-opt-balanced
> experiment_id: v0.44-param-opt-balanced-9d061299
> config_hash: 9d061299e06307f80adf10281732c86492b04a0af321a7c5fd1fed3cb31484a9
> run_timestamp: 2026-04-20T14:42:25+08:00
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
