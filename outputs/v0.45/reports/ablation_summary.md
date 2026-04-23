> version: v0.45
> experiment_id: v0.45-666870e5
> config_hash: 666870e5f5fce151b5e6a8ba43a8f5fce8fdc439c8c23a7155691ed6e82340ef
> run_timestamp: 2026-04-23T17:39:11+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 消融总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: aggregate_over_rolling_windows
- date_range: 2026-01-12 -> 2026-03-16

- full_model: total_profit=1916425568.00, cvar99=3143914.98, profit_delta_vs_full=0.00
- no_parameter_layout_enhancement: total_profit=1916425568.00, cvar99=3143914.98, profit_delta_vs_full=0.00
- no_policy_projection: total_profit=1896176238.00, cvar99=3148905.75, profit_delta_vs_full=-20249330.00
- no_state_enhancement: total_profit=1916858916.00, cvar99=3144023.75, profit_delta_vs_full=433348.00
