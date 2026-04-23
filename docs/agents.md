# agents.md

## v0.45 策略层说明

当前项目中的 `agent` 指 `HYBRID_PSO_V040` 的双层粒子角色：

- 上层粒子：搜索周度合约调整量、合约曲线参数和边际敞口带宽
- 下层粒子：搜索小时级现货修正响应参数

二者共同依赖：

- `src/training/tensor_bundle.py`
- `src/training/score_kernel.py`
- `src/agents/hybrid_pso.py`

## 参数语义

上层主输出：

- `contract_adjustment_mwh_raw`
- `contract_adjustment_mwh_exec`
- `contract_position_mwh`
- `exposure_band_mwh`
- `contract_curve_h1` 至 `contract_curve_h24`

下层主输出：

- `spot_hedge_mwh`
- `spot_hedge_limit_mwh`
- `spread_response`
- `load_deviation_response`
- `renewable_disturbance_response`

## 协同方式

- 上层先确定周度头寸与可承受边际敞口
- 下层在该边界内执行小时级现货修正
- 周度、小时级和 15 分钟结算结果统一由物化与报告链输出
