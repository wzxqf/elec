# ARCHITECTURE.md

## v0.46 当前架构索引

当前正式架构固定为：

`data -> policy -> feasible_domain -> tensor_bundle -> HYBRID_PSO_V040 -> materialize/evaluate/backtest -> reporting`

项目围绕湖南电力市场售电公司的购电组合与风险管理任务实现，正式主线固定为：

- 周度中长期头寸参数化搜索
- 小时级现货边际修正
- 15 分钟代理结算回测
- 政策文件清单、结构化规则表、制度状态轨迹与可行域审计

## 主模块

- `src/data/`：主数据读取、清洗、周度与小时级样本整理
- `src/policy/`：政策解析、制度状态构造、可行域编译与动作投影
- `src/training/tensor_bundle.py`：研究样本打包为训练张量
- `src/training/score_kernel.py`：批量评分、奖励与风险计算
- `src/agents/hybrid_pso.py`：`HYBRID_PSO_V040` 主实现
- `src/backtest/materialize.py`：周度、小时级、15 分钟结果物化
- `src/scripts/`：训练、评估、回测与一键流水线入口

## 运行链

1. 读取 `experiment_config.yaml`
2. 生成版本目录与参数快照
3. 读取市场数据与政策目录
4. 构造制度状态和前瞻制度状态
5. 编译可行域并生成张量包
6. 运行 `HYBRID_PSO_V040`
7. 物化周度、小时级和 15 分钟结果
8. 输出报告、指标、模型与清单

## 正式输出字段

周度主字段：

- `contract_adjustment_mwh_raw`
- `contract_adjustment_mwh_exec`
- `contract_position_mwh`
- `exposure_band_mwh`
- `profit_w`
- `reward_w`
- `cvar99_w`

小时级主字段：

- `spot_hedge_mwh`
- `spot_hedge_limit_mwh`
- `spread_response`
- `load_deviation_response`
- `renewable_disturbance_response`

## 正式入口与详细文档

- 正式远程验证入口：`python run_remote_jupyter.py`
- 服务器内部根入口：`python run_all.py`
- 流水线入口：`python -m src.scripts.run_pipeline`
- 训练入口：`python -m src.scripts.train`
- 评估入口：`python -m src.scripts.evaluate`
- 回测入口：`python -m src.scripts.backtest`

详细架构、实现机理、参数布局和报告链路见 `docs/v0.46_architecture_implementation.md`。
