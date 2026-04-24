> version: v0.45
> experiment_id: v0.45-e4f66602
> config_hash: e4f666026cf6639105a65d36808440d9da9966b5def2d278316ff1da496bff38
> run_timestamp: 2026-04-24T10:29:39+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 稳健性总结

- sample_scope: rolling_backtest_windows
- week_count: 10
- aggregation_method: scenario_sweep_over_rolling_results
- date_range: 2026-01-12 -> 2026-03-16

- contract_ratio_shift_+0.00: group=contract_ratio_shift, total_profit=1927150576.00, mean_cvar99=3146760.30, robustness_rank=4.00
- forecast_error_scale_1.00: group=forecast_error_scale, total_profit=1927150576.00, mean_cvar99=3146760.30, robustness_rank=4.00
- policy_cutoff_2026-01-01: group=policy_cutoff, total_profit=1927150576.00, mean_cvar99=3146760.30, robustness_rank=4.00
- forecast_error_scale_0.80: group=forecast_error_scale, total_profit=1652725467.52, mean_cvar99=2517408.24, robustness_rank=5.00
- policy_cutoff_2026-02-01: group=policy_cutoff, total_profit=1810774400.00, mean_cvar99=2942152.18, robustness_rank=5.00
- contract_ratio_shift_+0.10: group=contract_ratio_shift, total_profit=1858544298.88, mean_cvar99=3304098.32, robustness_rank=6.00
- contract_ratio_shift_-0.10: group=contract_ratio_shift, total_profit=1858544298.88, mean_cvar99=3304098.32, robustness_rank=6.00
- forecast_error_scale_1.20: group=forecast_error_scale, total_profit=1652725467.52, mean_cvar99=3776112.36, robustness_rank=9.00
