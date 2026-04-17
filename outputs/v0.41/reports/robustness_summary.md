> version: v0.41
> experiment_id: v0.41-d0f1cd29
> config_hash: d0f1cd29e630e124c05e6cf7194aae745f54b32a4507faee6bb7207d66a63e29
> run_timestamp: 2026-04-17T15:51:38+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 稳健性总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: scenario_sweep_over_rolling_results
- date_range: 2026-01-12 -> 2026-03-16

- policy_cutoff_2026-02-01: group=policy_cutoff, total_profit=1810774400.00, mean_cvar99=2942152.18, robustness_rank=3.00
- contract_ratio_shift_+0.00: group=contract_ratio_shift, total_profit=1719835452.00, mean_cvar99=3139273.83, robustness_rank=5.00
- forecast_error_scale_0.80: group=forecast_error_scale, total_profit=1441267183.20, mean_cvar99=2511419.06, robustness_rank=5.00
- forecast_error_scale_1.00: group=forecast_error_scale, total_profit=1719835452.00, mean_cvar99=3139273.83, robustness_rank=5.00
- policy_cutoff_2026-01-01: group=policy_cutoff, total_profit=1719835452.00, mean_cvar99=3139273.83, robustness_rank=5.00
- contract_ratio_shift_+0.10: group=contract_ratio_shift, total_profit=1650193384.80, mean_cvar99=3296237.52, robustness_rank=7.00
- contract_ratio_shift_-0.10: group=contract_ratio_shift, total_profit=1650193384.80, mean_cvar99=3296237.52, robustness_rank=7.00
- forecast_error_scale_1.20: group=forecast_error_scale, total_profit=1441267183.20, mean_cvar99=3767128.59, robustness_rank=9.00
