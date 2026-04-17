> version: v0.41
> experiment_id: v0.41-d0f1cd29
> config_hash: d0f1cd29e630e124c05e6cf7194aae745f54b32a4507faee6bb7207d66a63e29
> run_timestamp: 2026-04-17T11:47:32+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 基准策略比较

- sample_scope: holdout_validation_test
- week_count: 7
- aggregation_method: holdout_week_sum_and_mean
- date_range: 2026-02-02 -> 2026-03-16

- dynamic_lock_only: total_profit=nan, cvar99=2576188.25, delta_vs_dynamic_lock_only=nan, positive_week_count=5, negative_week_count=0, min_week_profit=nan, max_week_profit=nan
- fixed_holding_60: total_profit=nan, cvar99=2533050.40, delta_vs_dynamic_lock_only=nan, positive_week_count=5, negative_week_count=0, min_week_profit=nan, max_week_profit=nan
- static_no_spot_adjustment: total_profit=nan, cvar99=2619326.10, delta_vs_dynamic_lock_only=nan, positive_week_count=5, negative_week_count=0, min_week_profit=nan, max_week_profit=nan
- simple_rolling_hedge: total_profit=nan, cvar99=2511542.25, delta_vs_dynamic_lock_only=nan, positive_week_count=5, negative_week_count=0, min_week_profit=nan, max_week_profit=nan
