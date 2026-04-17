> version: v0.43
> experiment_id: v0.43-9160c944
> config_hash: 9160c9444e38f58d848a36f934bdf1a6dc4d6bfd205585a2a2535fadb9a37445
> run_timestamp: 2026-04-17T16:21:57+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 基准策略比较

- sample_scope: holdout_validation_test
- week_count: 7
- aggregation_method: holdout_week_sum_and_mean
- date_range: 2026-02-02 -> 2026-03-16

- dynamic_lock_only: total_profit=1459720481.00, cvar99=2724291.18, delta_vs_dynamic_lock_only=0.00, positive_week_count=7, negative_week_count=0, min_week_profit=126533667.50, max_week_profit=318707952.50
- fixed_holding_60: total_profit=1389510561.00, cvar99=2680772.61, delta_vs_dynamic_lock_only=-70209920.00, positive_week_count=7, negative_week_count=0, min_week_profit=110141219.50, max_week_profit=302406128.50
- static_no_spot_adjustment: total_profit=1529931745.00, cvar99=2767809.75, delta_vs_dynamic_lock_only=70211264.00, positive_week_count=7, negative_week_count=0, min_week_profit=139038889.00, max_week_profit=335010032.50
- simple_rolling_hedge: total_profit=1362429263.99, cvar99=2661942.46, delta_vs_dynamic_lock_only=-97291217.01, positive_week_count=7, negative_week_count=0, min_week_profit=104672440.92, max_week_profit=294367540.94

- 强基准结论: `dynamic_lock_only` 仍作为强基准保留，但当前口径下利润领先者为 `static_no_spot_adjustment`。
