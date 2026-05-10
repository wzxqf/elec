> version: v0.47
> experiment_id: v0.47-fdb00545
> config_hash: fdb00545c6611582b09096325a7dba3890362251cb1e482326ef83f8c8ee9b0a
> run_timestamp: 2026-05-10T11:04:42+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 约束激活报告

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: window_level_projection_audit
- date_range: 2026-01-12 -> 2026-03-16

- policy_tightening_trigger_count: 0
- default_projection_trigger_count: 10
- projection_clip_mean: 600145.6719
- projection_clip_max: 923190.9375
- total_trigger_count: 10

## 原因码分布

- default_band_floor: 10

## 触发明细

- week=2026-01-12 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=186096.4688
- week=2026-01-19 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=234961.2969
- week=2026-01-26 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=227898.7031
- week=2026-02-02 00:00:00, projection_target_field=contract_adjustment_mwh, projection_rule_name=default_band_floor, projection_before=-1204162.1250, projection_after=-481664.8438
- week=2026-02-09 00:00:00, projection_target_field=contract_adjustment_mwh, projection_rule_name=default_band_floor, projection_before=-1142700.6250, projection_after=-457080.2188
- week=2026-02-16 00:00:00, projection_target_field=contract_adjustment_mwh, projection_rule_name=default_band_floor, projection_before=-902605.0625, projection_after=-361042.0000
- week=2026-02-23 00:00:00, projection_target_field=contract_adjustment_mwh, projection_rule_name=default_band_floor, projection_before=-949653.6875, projection_after=-379861.4688
- week=2026-03-02 00:00:00, projection_target_field=contract_adjustment_mwh, projection_rule_name=default_band_floor, projection_before=-1073461.8750, projection_after=-429384.7500
- week=2026-03-09 00:00:00, projection_target_field=contract_adjustment_mwh, projection_rule_name=default_band_floor, projection_before=-933631.3750, projection_after=-373452.5312
- week=2026-03-16 00:00:00, projection_target_field=contract_adjustment_mwh, projection_rule_name=default_band_floor, projection_before=-775307.1250, projection_after=-310122.8438
