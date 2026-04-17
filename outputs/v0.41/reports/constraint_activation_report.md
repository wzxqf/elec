> version: v0.41
> experiment_id: v0.41-d0f1cd29
> config_hash: d0f1cd29e630e124c05e6cf7194aae745f54b32a4507faee6bb7207d66a63e29
> run_timestamp: 2026-04-17T11:47:32+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 约束激活报告

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: window_level_projection_audit
- date_range: 2026-01-12 -> 2026-03-16

- policy_tightening_trigger_count: 0
- default_projection_trigger_count: 10
- projection_clip_mean: 279884.6820
- projection_clip_max: 743741.9375
- total_trigger_count: 10

## 原因码分布

- default_band_floor: 8
- default_band_cap: 2

## 触发明细

- week=2026-01-12 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=186096.4688
- week=2026-01-19 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_cap, projection_before=1879690.3750, projection_after=1174806.5000
- week=2026-01-26 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_cap, projection_before=1823189.6250, projection_after=1139493.5000
- week=2026-02-02 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=200693.6875
- week=2026-02-09 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=190450.1094
- week=2026-02-16 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=150434.1719
- week=2026-02-23 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=158275.6094
- week=2026-03-02 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=178910.3125
- week=2026-03-09 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=155605.2344
- week=2026-03-16 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=129217.8516
