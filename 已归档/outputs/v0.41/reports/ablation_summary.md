> version: v0.41
> experiment_id: v0.41-d0f1cd29
> config_hash: d0f1cd29e630e124c05e6cf7194aae745f54b32a4507faee6bb7207d66a63e29
> run_timestamp: 2026-04-17T15:51:38+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 消融总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: aggregate_over_rolling_windows
- date_range: 2026-01-12 -> 2026-03-16

- full_model: total_profit=1719835452.00, cvar99=3139273.83, profit_delta_vs_full=0.00
- no_parameter_layout_enhancement: total_profit=1719835452.00, cvar99=3139273.83, profit_delta_vs_full=0.00
- no_policy_projection: total_profit=1672655798.12, cvar99=3149888.77, profit_delta_vs_full=-47179653.88
- no_state_enhancement: total_profit=1719247722.00, cvar99=3139288.20, profit_delta_vs_full=-587730.00
