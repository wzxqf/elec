> version: v0.45
> experiment_id: v0.45-e4f66602
> config_hash: e4f666026cf6639105a65d36808440d9da9966b5def2d278316ff1da496bff38
> run_timestamp: 2026-04-24T10:29:39+08:00
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
