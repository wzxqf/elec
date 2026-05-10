> version: v0.46
> experiment_id: v0.46-f409eec7
> config_hash: f409eec73a94a3379423732b1bc82b83b800edf1fb58988abb7c513705d2ba1d
> run_timestamp: 2026-05-10T08:44:20+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 消融总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: aggregate_over_rolling_windows
- date_range: 2026-01-12 -> 2026-03-16

- full_model: total_profit=1053366718.00, cvar99=4776809.30, profit_delta_vs_full=0.00
- no_parameter_layout_enhancement: total_profit=792471217.50, cvar99=4465553.28, profit_delta_vs_full=-260895500.50
- no_policy_projection: total_profit=2068174336.00, cvar99=4293417.42, profit_delta_vs_full=1014807618.00
- no_state_enhancement: total_profit=1053366718.00, cvar99=4776809.30, profit_delta_vs_full=0.00
