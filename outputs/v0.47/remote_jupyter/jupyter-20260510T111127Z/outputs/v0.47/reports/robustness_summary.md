> version: v0.47
> experiment_id: v0.47-fdb00545
> config_hash: fdb00545c6611582b09096325a7dba3890362251cb1e482326ef83f8c8ee9b0a
> run_timestamp: 2026-05-10T11:04:42+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 稳健性总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: scenario_sweep_over_rolling_results
- date_range: 2026-01-12 -> 2026-03-16

- contract_ratio_shift_+0.00: group=contract_ratio_shift, total_profit=1053366718.00, mean_cvar99=4776809.30, robustness_rank=4.00
- forecast_error_scale_0.80: group=forecast_error_scale, total_profit=761490506.80, mean_cvar99=3821447.44, robustness_rank=4.00
- forecast_error_scale_1.00: group=forecast_error_scale, total_profit=1053366718.00, mean_cvar99=4776809.30, robustness_rank=4.00
- policy_cutoff_2026-01-01: group=policy_cutoff, total_profit=1053366718.00, mean_cvar99=4776809.30, robustness_rank=4.00
- contract_ratio_shift_+0.10: group=contract_ratio_shift, total_profit=980397665.20, mean_cvar99=5015649.77, robustness_rank=6.00
- contract_ratio_shift_-0.10: group=contract_ratio_shift, total_profit=980397665.20, mean_cvar99=5015649.77, robustness_rank=6.00
- policy_cutoff_2026-02-01: group=policy_cutoff, total_profit=459940926.00, mean_cvar99=4713290.86, robustness_rank=6.00
- forecast_error_scale_1.20: group=forecast_error_scale, total_profit=761490506.80, mean_cvar99=5732171.16, robustness_rank=8.00
