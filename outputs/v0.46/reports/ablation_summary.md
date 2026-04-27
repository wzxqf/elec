> version: v0.46
> experiment_id: v0.46-f409eec7
> config_hash: f409eec73a94a3379423732b1bc82b83b800edf1fb58988abb7c513705d2ba1d
> run_timestamp: 2026-04-27T02:13:14+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 消融总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: aggregate_over_rolling_windows
- date_range: 2026-01-12 -> 2026-03-16

- full_model: total_profit=2547823488.00, cvar99=3434425.25, profit_delta_vs_full=0.00
- no_parameter_layout_enhancement: total_profit=2346331456.00, cvar99=3308931.65, profit_delta_vs_full=-201492032.00
- no_policy_projection: total_profit=2547823488.00, cvar99=3434425.25, profit_delta_vs_full=0.00
- no_state_enhancement: total_profit=2547823488.00, cvar99=3434425.25, profit_delta_vs_full=0.00
