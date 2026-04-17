> version: v0.4
> experiment_id: v0.4-840a5740
> config_hash: 840a574028870937a87c4ff725930c4682004c50d528119e8eb3ec14af126a64
> run_timestamp: 2026-04-17T01:12:11+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 稳健性总结

- policy_cutoff_2026-02-01: group=policy_cutoff, total_profit=1495428560.00, mean_cvar99=2902099.93, robustness_rank=3.00
- contract_ratio_shift_+0.00: group=contract_ratio_shift, total_profit=1416261816.00, mean_cvar99=3108591.38, robustness_rank=5.00
- forecast_error_scale_0.80: group=forecast_error_scale, total_profit=1131632911.04, mean_cvar99=2486873.10, robustness_rank=5.00
- forecast_error_scale_1.00: group=forecast_error_scale, total_profit=1416261816.00, mean_cvar99=3108591.38, robustness_rank=5.00
- policy_cutoff_2026-01-01: group=policy_cutoff, total_profit=1416261816.00, mean_cvar99=3108591.38, robustness_rank=5.00
- contract_ratio_shift_+0.10: group=contract_ratio_shift, total_profit=1345104589.76, mean_cvar99=3264020.94, robustness_rank=7.00
- contract_ratio_shift_-0.10: group=contract_ratio_shift, total_profit=1345104589.76, mean_cvar99=3264020.94, robustness_rank=7.00
- forecast_error_scale_1.20: group=forecast_error_scale, total_profit=1131632911.04, mean_cvar99=3730309.65, robustness_rank=9.00
