> version: v0.4
> experiment_id: v0.4-840a5740
> config_hash: 840a574028870937a87c4ff725930c4682004c50d528119e8eb3ec14af126a64
> run_timestamp: 2026-04-17T01:12:11+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 消融总结

- full_model: total_profit=1416261816.00, cvar99=3108591.38, profit_delta_vs_full=0.00
- no_parameter_layout_enhancement: total_profit=1416261816.00, cvar99=3108591.38, profit_delta_vs_full=0.00
- no_policy_projection: total_profit=1371627752.00, cvar99=3117796.42, profit_delta_vs_full=-44634064.00
- no_state_enhancement: total_profit=1406265212.00, cvar99=3039427.85, profit_delta_vs_full=-9996604.00
