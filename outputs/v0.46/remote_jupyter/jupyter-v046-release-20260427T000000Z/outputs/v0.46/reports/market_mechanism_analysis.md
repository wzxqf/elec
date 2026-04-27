> version: v0.46
> experiment_id: v0.46-f409eec7
> config_hash: f409eec73a94a3379423732b1bc82b83b800edf1fb58988abb7c513705d2ba1d
> run_timestamp: 2026-04-27T02:13:14+00:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 市场机制分析

## 制度规则 -> 模型字段 -> 结果输出

- 正式规则数: 34
- 市场约束映射数: 8
- 映射示例: lt_settlement_base, spot_marginal_exposure, lt_spot_coupling_state, q_lt_hourly, lt_energy_15m; _allocate_weekly_lt_to_hourly, q_lt_hourly; settlement_interval_hours=0.25, settle_week, procurement_cost_15m, imbalance_energy_15m
- 中长期价格通过结算口径影响采购成本与超额收益比较。
- 现货价格与偏差结算通过 15 分钟代理结算进入回测层。
- 制度状态、生效区间和模型映射字段保持同一时点口径。
