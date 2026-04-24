> version: v0.45
> experiment_id: v0.45-e4f66602
> config_hash: e4f666026cf6639105a65d36808440d9da9966b5def2d278316ff1da496bff38
> run_timestamp: 2026-04-24T10:29:39+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 消融总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: aggregate_over_rolling_windows
- date_range: 2026-01-12 -> 2026-03-16

- full_model: total_profit=1927150576.00, cvar99=3146760.30, profit_delta_vs_full=0.00
- no_parameter_layout_enhancement: total_profit=1927150576.00, cvar99=3146760.30, profit_delta_vs_full=0.00
- no_policy_projection: total_profit=1922992708.00, cvar99=3145743.52, profit_delta_vs_full=-4157868.00
- no_state_enhancement: total_profit=1926051936.00, cvar99=3146481.35, profit_delta_vs_full=-1098640.00
