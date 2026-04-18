> version: v0.4
> experiment_id: v0.4-840a5740
> config_hash: 840a574028870937a87c4ff725930c4682004c50d528119e8eb3ec14af126a64
> run_timestamp: 2026-04-17T01:12:11+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 基准策略比较

- dynamic_lock_only: total_profit=1144549644.50, cvar99=2784274.57, delta_vs_dynamic_lock_only=0.00
- fixed_holding_60: total_profit=1107423116.50, cvar99=2743179.86, delta_vs_dynamic_lock_only=-37126528.00
- simple_rolling_hedge: total_profit=1087534103.94, cvar99=2721840.18, delta_vs_dynamic_lock_only=-57015540.56
- static_no_spot_adjustment: total_profit=1181676236.50, cvar99=2826937.36, delta_vs_dynamic_lock_only=37126592.00
