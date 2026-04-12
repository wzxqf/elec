# ARCHITECTURE.md

## 1. 架构定位
本项目是围绕湖南电力市场售电公司采购策略实验构建的离线研究工程，当前已固化为“周度中长期底仓残差 PPO 决策 + 小时级显式规则现货修正 + 15 分钟代理结算回测”的两阶段架构，而不是端到端单层策略。程序目标不是搭建通用交易平台，而是为论文实证提供可复现的数据处理、训练、验证、回测与分析闭环。

---

## 2. 顶层结构
建议将整个项目理解为六层：

1. **配置层**：根目录 `experiment_config.yaml` + `configs/`
2. **数据与制度层**：`src/data/`、`src/policy/`
3. **决策环境层**：`src/envs/`
4. **策略训练层**：`src/agents/`
5. **规则执行与回测层**：`src/rules/`、`src/backtest/`
6. **分析与输出层**：`src/analysis/`、`src/scripts/`、`outputs/`

其中，`src/scripts/` 负责把各层串成完整流水线，`outputs/` 负责落地模型、指标、图表 CSV 和中文报告。

---

## 3. 目录职责固化

### 3.1 configs/
统一管理实验模板与参考配置，不允许在训练脚本和回测脚本中散落硬编码参数。

根目录 `experiment_config.yaml` 为正式实验主入口。

- `default.yaml`：项目根参数、输出路径、基础运行参数
- `data.yaml`：数据口径、样本区间、字段处理要求
- `ppo.yaml`：PPO 超参数与训练设置
- `rules.yaml`：小时级规则修正参数
- `backtest.yaml`：结算、约束、基准策略、回测设置
- `analysis.yaml`：敏感性、鲁棒性、超参数搜索设置

### 3.2 src/data/
负责原始数据进入模型前的全部处理，不承担训练和回测逻辑。

- `loader`：定位并读取 `total.csv`
- `preprocess`：清洗、去重、重采样、缺口检查、质量报告
- `weekly_builder`：从 15 分钟与小时数据构造周度元数据与特征包
- `scenario_generator`：训练/验证/回测周划分与 bootstrap 训练序列生成

### 3.3 src/policy/
负责政策文件解析与制度状态构造。

- `policy_parser`：解析政策目录中的 docx/md 文本与 xlsx 结果表，输出文件清单、元数据索引和解析失败清单
- `policy_regime`：生成周度当前制度状态轨迹与四组前瞻制度状态
- `policy_tables`：输出带 `state_group` 等字段的结构化规则表与摘要报告

该层只负责“制度信息结构化”，不直接做交易决策。

### 3.4 src/envs/
当前核心环境为 `src/envs/elec_env.py`。

环境职责：
- 读取周度特征构造 observation
- 接收二维动作 `(\Delta h_w^{raw}, b_w)`，分别表示相对 `dynamic_lock_only` 强基准的底仓残差原始动作与边际敞口带宽动作
- 调用周度仿真逻辑得到 reward 与诊断信息
- 维持序贯周度 episode

环境只承担**上层周度决策**，不直接处理 15 分钟市场清算细节。

### 3.5 src/agents/
负责强化学习训练与策略评估。

当前主实现：
- `train_ppo.py`

职责包括：
- 构建 `DummyVecEnv/Monitor`
- 按需启用 `VecNormalize`
- 使用 Stable-Baselines3 PPO 训练
- 保存 latest/best/checkpoint 模型
- 导出训练指标与验证指标
- 将 PPO 输出转换为周度动作序列

此层只产出**周度策略动作**，不负责基准策略与分析实验。

### 3.6 src/rules/
负责小时级显式规则修正。

这里不是独立 RL，而是把周度动作转成可执行的小时级现货修正轨迹。其职责是根据：
- 周度中长期最终目标持仓
- 小时级价格偏差、负荷偏差、新能源偏差
- 制度状态、辅助服务耦合约束、边际敞口带宽约束与平滑要求

生成小时级 `delta_q`、套保误差与裁剪统计，并真实支持 `soft_clip` 连续平滑压缩。

### 3.7 src/backtest/
负责把策略真正落到结算口径上。

建议视为三部分：
- `benchmarks`：固定锁定、规则对冲等基准动作生成
- `simulator`：周度动作 → 小时修正 → 15 分钟结算的主仿真链路
- `settlement/metrics`：代理结算与指标汇总

这是项目中最关键的执行层。训练环境中的 reward 计算也最终依赖这里的周度仿真口径。

### 3.8 src/analysis/
负责训练后扩展分析，不介入主模型训练。

- `sensitivity`：参数或市场扰动敏感性分析
- `robustness`：波动、价格上限、预测误差等鲁棒性实验

### 3.9 src/scripts/
面向使用者的统一入口层。

- `run_pipeline.py`：全流程总入口
- `train.py`：训练入口
- `evaluate.py`：验证入口
- `backtest.py`：回测、基准比较、敏感性与鲁棒性入口
- 其他脚本：诊断或辅助执行

原则上，外部只调用 `src/scripts/`，不直接调用底层模块。

### 3.10 outputs/
统一输出目录，禁止结果散落到源码目录。

- `outputs/models/`：模型文件
- `outputs/logs/`：日志
- `outputs/metrics/`：指标表、轨迹表、政策表
- `outputs/figures/`：图表对应 CSV
- `outputs/reports/`：中文摘要与详细报告

---

## 4. 主数据流

### 4.1 预处理流
`total.csv`  
→ `loader` 读取  
→ `preprocess` 清洗与频率校验  
→ `weekly_builder` 构造 quarter/hourly/weekly 三层数据包  
→ 输出 `bundle`

### 4.2 制度流
政策目录文件  
→ `policy_parser` 解析  
→ `policy_regime` 生成周度制度状态  
→ 回填到 `weekly_metadata` 和 `weekly_features`

### 4.3 训练流
`bundle + split + train_sequence + config`  
→ `ElecEnv`  
→ `PPO.train`  
→ `ppo_best.zip / ppo_latest.zip`  
→ 训练指标与验证指标

### 4.4 执行流
周度 PPO 动作  
→ `simulate_week`  
→ 小时级规则修正  
→ 15 分钟代理结算  
→ 周度结果 / 小时轨迹 / 结算轨迹

### 4.5 分析流
PPO 结果 + 基准策略结果  
→ 成本、CVaR、套保误差比较  
→ 敏感性分析  
→ 鲁棒性分析  
→ 中文摘要与详细报告

---

## 5. 核心对象与边界

### 5.1 bundle
`bundle` 是项目内部最重要的数据容器，至少应稳定包含：
- `quarter`
- `hourly`
- `weekly_metadata`
- `weekly_features`
- `feature_manifest`
- `policy_inventory`
- `policy_metadata_index`
- `policy_parse_failures`
- `policy_rule_table`
- `policy_state_trace`
- `reward_reference`
- `reward_robust_stats`
- `agent_feature_columns`

原则上，训练、验证、回测都从 `bundle` 取数，不再各自重建一套数据口径。

### 5.2 action
上层动作固定为二维：
- `delta_lock_ratio_raw`：相对 `dynamic_lock_only` 基准底仓的残差原始动作
- `b_w`：小时级现货可承担的边际敞口带宽

不再扩展成高维复杂动作空间，避免偏离当前论文规划。

### 5.3 reward
奖励不直接等于利润，而是围绕三层结构组织：
- 相对 `dynamic_lock_only` 强基准的成本改进主信号
- 尾部风险相对超额惩罚
- 执行质量项

奖励在环境层产生，但其经济含义由回测口径决定，最终以 `tanh` 做软压缩而不是硬裁剪。

---

## 6. 关键执行逻辑

### 6.1 两阶段而非端到端
项目明确采用：
- 上层：PPO 学习周度持仓动作
- 下层：规则法执行小时级现货修正

这样做的原因是：
- 周度决策与小时调节时间尺度不同
- 可解释性更强
- 更贴合当前论文与交付约束
- 避免下层单独 DRL 带来的复杂度膨胀

### 6.2 中长期与现货关系
中长期部分只负责形成基础持仓与结算基线，不应被实现成物理执行计划；现货修正才对应短期暴露管理。程序实现必须保持这一口径一致。

### 6.2.1 制度状态的双重进入
制度状态必须同时进入：
- 周度状态空间
- 15 分钟代理结算逻辑

其中新能源机制电价制度只进入状态、波动解释和结算说明，不直接写入奖励主项。

### 6.3 15 分钟结算为最终口径
虽然上层是周度、下层是小时级，但最终成本、风险与套保误差必须落到 15 分钟代理结算层。这一层不能省略，否则训练结果与论文结论会脱节。

---

## 7. 入口与调用关系

### 推荐调用顺序
1. `python -m src.scripts.run_pipeline`
2. 或拆分执行：
   - `python -m src.scripts.train`
   - `python -m src.scripts.evaluate`
   - `python -m src.scripts.backtest`

### 调用关系
- `run_pipeline` 串联 `prepare_project_context -> run_train -> run_evaluate -> run_backtest`
- `train` 仅负责训练与训练摘要
- `evaluate` 仅负责验证集检查与验证摘要
- `backtest` 负责正式比较、图表导出、敏感性、鲁棒性、滚动验证和超参数搜索

---

## 8. 当前应保持不变的设计约束

1. 不增加下层独立 DRL。
2. 不把需求响应、调频收益纳入主策略收益。
3. 不把 15 分钟实时层改成独立动作层。
4. 不在脚本中重新拼装数据口径，统一走 `prepare_project_context`。
5. 不把图表默认导出为图片，继续保持 CSV 导出。
6. 所有摘要、报告和日志继续保持中文输出。
7. 输出文件统一写入 `outputs/`，不污染源码目录。

---

## 9. 后续扩展原则
如需扩展，只能沿以下方向推进：

- 在 `src/data/` 中补充更稳健的特征工程
- 在 `src/policy/` 中提高政策状态提取精度
- 在 `src/rules/` 中细化小时修正规则
- 在 `src/backtest/` 中增强结算与指标解释
- 在 `src/analysis/` 中扩展对比实验

原则上不改顶层“两阶段决策 + 15分钟结算回测”主架构。
