# ARCHITECTURE.md

## v0.38 总体架构

当前唯一有效正式架构为：

`data -> policy_deep -> tensor_bundle -> parametric_hybrid_pso -> policy_projection -> rolling_retrain -> 15m_settlement -> reporting`

项目围绕湖南电力市场售电公司的购电组合与风险管理任务构建，正式主线固定为：

- 周度参数化中长期持仓策略
- 小时级现货滚动修正规则
- 15 分钟代理结算回测

正式主语义不再使用 `lock_ratio`、`delta_lock` 或动作裁剪链。上层与下层均在电量域中定义。

## 模块分层

### 1. 数据层

- `src/data/loader.py`：读取主数据源
- `src/data/preprocess.py`：清洗、缺失处理、频率校验
- `src/data/weekly_builder.py`：构造周度、小时级、15 分钟研究张量所需基础表

### 2. 制度深结构层

- `src/policy_deep/document_reader.py`：政策文件清单、元数据、失败清单
- `src/policy_deep/candidate_rules.py`：候选规则生成
- `src/policy_deep/llm_bridge.py`：可关闭、可缓存的 LLM 候选接口
- `src/policy_deep/rule_review.py`：候选规则审核与结构化落表
- `src/policy_deep/regime_builder.py`：当前制度状态与四组前瞻制度状态构造

### 3. 训练张量层

- `src/training/tensor_bundle.py`：将研究样本编译为 Torch 张量包
- `src/training/score_kernel.py`：参数化双层混合粒子群评分核

评分核正式输出以下对象：

- 周度合约原始调整量与投影后执行量
- 周度合约持仓量
- 现货边际风险带宽
- 小时级现货修正量
- 经济收益、训练奖励、`CVaR_99%`

### 4. 策略搜索层

- `src/agents/hybrid_pso.py`：参数化双层混合粒子群主实现

其中：

- 上层粒子搜索周度参数化持仓策略
- 下层粒子搜索小时级现货滚动修正规则

上层正式输出：

- `contract_adjustment_mwh_raw`
- `contract_adjustment_mwh_exec`
- `contract_position_mwh`
- `exposure_band_mwh`
- `contract_curve_h1` 至 `contract_curve_h24`
- `policy_gate`

下层正式输出：

- `spot_hedge_mwh`
- `spot_hedge_limit_mwh`
- `spread_response`
- `load_deviation_response`
- `renewable_disturbance_response`
- `policy_shrink_multiplier`

### 5. 约束投影层

政策规则投影器是唯一正式业务约束器。它负责：

- 将上层原始合约调整量投影到政策可行域
- 记录 `policy_projection_active`
- 记录 `policy_violation_penalty_w`
- 输出政策绑定痕迹

正式主链不再使用额外的经验性极端裁剪，不再通过 `clip`、`soft_clip`、`hard_clip` 对业务决策做二次截断。

### 6. 评估与报告层

- `src/backtest/rolling_pipeline.py`：滚动窗口生成、每窗口重训、窗口汇总
- `src/backtest/materialize.py`：评估与回测阶段物化周度、小时级和 15 分钟结果

正式周度输出至少包含：

- `contract_adjustment_mwh_raw`
- `contract_adjustment_mwh_exec`
- `contract_position_mwh`
- `exposure_band_mwh`
- `retail_revenue_w`
- `procurement_cost_w`
- `profit_w`
- `profit_baseline_w`
- `excess_profit_w`
- `cvar99_w`
- `hedge_error_w`
- `reward_w`

### 7. 入口层

唯一保留的正式入口如下：

- `src/scripts/train.py`
- `src/scripts/evaluate.py`
- `src/scripts/backtest.py`
- `src/scripts/run_pipeline.py`
- 根入口 `run_all.py`

## 训练主链

1. 读取配置与数据
2. 构造制度深结构状态
3. 生成张量包
4. 运行参数化双层混合粒子群搜索
5. 用 `score_kernel` 批量评估经济收益、训练奖励、`CVaR_99%` 和政策投影惩罚
6. 输出模型摘要、参数快照和训练指标

训练阶段不生成大量 pandas DataFrame，不使用 PPO、SB3、gym 环境，不保留旧静态参数推断主链。

## 验证与回测主链

1. 生成滚动或扩展窗口
2. 每个窗口单独再训练
3. 在窗口验证集和测试集执行参数化评分与 15 分钟结算物化
4. 汇总窗口级和总体级指标
5. 输出滚动计划、参数快照、窗口指标和总报告

主回测不接受“训练一次后整段静态推断”。

## 收益与奖励

正式项目同时保留两套定义：

- `profit`：经济收益函数，用于论文结果和回测报告
- `reward`：训练奖励函数，用于优化器目标

两者不得混用，不再以 `-procurement_cost` 代替正式奖励函数。

尾部风险默认口径统一为 `CVaR_99%`。

## GPU 设计原则

- 热路径集中在 `tensor_bundle + score_kernel + hybrid_pso`
- 训练阶段优先 Torch 张量批量计算
- CPU 保留给数据读取、政策解析和最终报告物化
- 如果 CUDA 可用，评分核应在 `cuda` 设备执行；否则明确回退到 CPU

## 输出结构

所有正式产物统一写入：

- `outputs/<version>/models/`
- `outputs/<version>/logs/`
- `outputs/<version>/metrics/`
- `outputs/<version>/figures/`
- `outputs/<version>/reports/`

其中 `<version>` 取自 `experiment_config.yaml` 的 `project.version`。
