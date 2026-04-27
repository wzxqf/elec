> version: v0.46
> experiment_id: v0.46-f409eec7
> config_hash: f409eec73a94a3379423732b1bc82b83b800edf1fb58988abb7c513705d2ba1d
> run_timestamp: 2026-04-27T02:13:14+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 稳健性总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: scenario_sweep_over_rolling_results
- date_range: 2026-01-12 -> 2026-03-16

- contract_ratio_shift_+0.00: group=contract_ratio_shift, total_profit=2547823488.00, mean_cvar99=3434425.25, robustness_rank=4.00
- forecast_error_scale_0.80: group=forecast_error_scale, total_profit=2285786140.16, mean_cvar99=2747540.20, robustness_rank=4.00
- forecast_error_scale_1.00: group=forecast_error_scale, total_profit=2547823488.00, mean_cvar99=3434425.25, robustness_rank=4.00
- policy_cutoff_2026-01-01: group=policy_cutoff, total_profit=2547823488.00, mean_cvar99=3434425.25, robustness_rank=4.00
- contract_ratio_shift_+0.10: group=contract_ratio_shift, total_profit=2482314151.04, mean_cvar99=3606146.51, robustness_rank=6.00
- contract_ratio_shift_-0.10: group=contract_ratio_shift, total_profit=2482314151.04, mean_cvar99=3606146.51, robustness_rank=6.00
- policy_cutoff_2026-02-01: group=policy_cutoff, total_profit=1944353536.00, mean_cvar99=3068616.96, robustness_rank=6.00
- forecast_error_scale_1.20: group=forecast_error_scale, total_profit=2285786140.16, mean_cvar99=4121310.30, robustness_rank=8.00
