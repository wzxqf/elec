> version: v0.47
> experiment_id: v0.47-fdb00545
> config_hash: fdb00545c6611582b09096325a7dba3890362251cb1e482326ef83f8c8ee9b0a
> run_timestamp: 2026-05-10T11:04:42+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 基准策略比较

- sample_scope: holdout_validation_test
- week_count: 7
- aggregation_method: holdout_week_sum_and_mean
- date_range: 2026-02-02 -> 2026-03-16

- dynamic_lock_only: total_profit=1022129162.00, cvar99=3777383.82, delta_vs_dynamic_lock_only=0.00, positive_week_count=7, negative_week_count=0, min_week_profit=75050531.50, max_week_profit=264505625.50
- fixed_holding_60: total_profit=949453066.00, cvar99=3739479.39, delta_vs_dynamic_lock_only=-72676096.00, positive_week_count=7, negative_week_count=0, min_week_profit=58657827.50, max_week_profit=247071769.50
- static_no_spot_adjustment: total_profit=1093601418.00, cvar99=3815782.96, delta_vs_dynamic_lock_only=71472256.00, positive_week_count=7, negative_week_count=0, min_week_profit=91443107.50, max_week_profit=281363161.50
- simple_rolling_hedge: total_profit=1063903672.99, cvar99=3794524.57, delta_vs_dynamic_lock_only=41774510.99, positive_week_count=7, negative_week_count=0, min_week_profit=83788856.92, max_week_profit=277624029.94

- 强基准结论: `dynamic_lock_only` 仍作为强基准保留，但当前口径下利润领先者为 `static_no_spot_adjustment`。
