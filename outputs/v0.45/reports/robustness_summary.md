> version: v0.45
> experiment_id: v0.45-666870e5
> config_hash: 666870e5f5fce151b5e6a8ba43a8f5fce8fdc439c8c23a7155691ed6e82340ef
> run_timestamp: 2026-04-23T17:39:11+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 稳健性总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: scenario_sweep_over_rolling_results
- date_range: 2026-01-12 -> 2026-03-16

- contract_ratio_shift_+0.00: group=contract_ratio_shift, total_profit=1916425568.00, mean_cvar99=3143914.98, robustness_rank=4.00
- forecast_error_scale_1.00: group=forecast_error_scale, total_profit=1916425568.00, mean_cvar99=3143914.98, robustness_rank=4.00
- policy_cutoff_2026-01-01: group=policy_cutoff, total_profit=1916425568.00, mean_cvar99=3143914.98, robustness_rank=4.00
- forecast_error_scale_0.80: group=forecast_error_scale, total_profit=1641775996.16, mean_cvar99=2515131.98, robustness_rank=5.00
- policy_cutoff_2026-02-01: group=policy_cutoff, total_profit=1810774400.00, mean_cvar99=2942152.18, robustness_rank=5.00
- contract_ratio_shift_+0.10: group=contract_ratio_shift, total_profit=1847763175.04, mean_cvar99=3301110.72, robustness_rank=6.00
- contract_ratio_shift_-0.10: group=contract_ratio_shift, total_profit=1847763175.04, mean_cvar99=3301110.72, robustness_rank=6.00
- forecast_error_scale_1.20: group=forecast_error_scale, total_profit=1641775996.16, mean_cvar99=3772697.97, robustness_rank=9.00
