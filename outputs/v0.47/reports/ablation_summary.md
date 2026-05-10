> version: v0.47
> experiment_id: v0.47-fdb00545
> config_hash: fdb00545c6611582b09096325a7dba3890362251cb1e482326ef83f8c8ee9b0a
> run_timestamp: 2026-05-10T11:04:42+00:00
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
