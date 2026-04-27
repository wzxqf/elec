> version: v0.46
> experiment_id: v0.46-f409eec7
> config_hash: f409eec73a94a3379423732b1bc82b83b800edf1fb58988abb7c513705d2ba1d
> run_timestamp: 2026-04-27T02:13:14+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 约束激活报告

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: window_level_projection_audit
- date_range: 2026-01-12 -> 2026-03-16

- policy_tightening_trigger_count: 0
- default_projection_trigger_count: 10
- projection_clip_mean: 181254.3445
- projection_clip_max: 234961.2969
- total_trigger_count: 10

## 原因码分布

- default_band_floor: 10

## 触发明细

- week=2026-01-12 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=186096.4688
- week=2026-01-19 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=234961.2969
- week=2026-01-26 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=227898.7031
- week=2026-02-02 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=200693.6875
- week=2026-02-09 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=190450.1094
- week=2026-02-16 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=150434.1719
- week=2026-02-23 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=158275.6094
- week=2026-03-02 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=178910.3125
- week=2026-03-09 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=155605.2344
- week=2026-03-16 00:00:00, projection_target_field=exposure_band_mwh, projection_rule_name=default_band_floor, projection_before=0.0000, projection_after=129217.8516
