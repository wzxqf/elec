# Codex 交付说明（周度中长期 + 小时级现货 + 15分钟结算版）

## 1. 任务目标

请在本地项目目录 `/Users/dk/py/elec` 下，重建一个**可运行、可训练、可验证、可回测、可输出日志与报告**的项目。该项目用于实现“售电公司周度中长期持仓与小时级现货风险对冲”实验，采用 **Gymnasium 自定义环境 + Stable-Baselines3 的 PPO**。不得使用 `rl-baselines3-zoo` 作为主训练框架，不得额外引入第二个独立 DRL 下层模型。

本项目必须坚持**真实本地数据优先**。核心主数据文件名为 `total.csv`。实现时必须先读取和清洗真实数据，再在真实数据基础上完成聚合、特征工程、环境构建、训练与回测。不得为了让代码跑通而擅自改用模拟主数据。只有在某些变量缺失且无法直接由现有列获得时，才允许使用**可追溯、可配置、可替换**的估算口径，并必须写入配置和日志。

---

## 2. 本次交付的硬性口径

### 2.1 项目、环境与框架
- 项目根目录：`/Users/dk/py/elec`
- Python / mamba 环境名：`elec_env`
- 如本地不存在该环境，则创建同名环境，Python 版本固定为 `3.11`
- 强化学习框架：**普通 Stable-Baselines3**
- Gym 环境：**Gymnasium**
- 设备原则：**优先确保 CPU 可运行**，若检测到 CUDA 可用，可自动切换到 GPU，但代码不得依赖 GPU 才能运行

### 2.2 本次模型结构
- 上层：**周度中长期持仓决策模型**
- 下层：**小时级现货滚动修正模型**
- 结算层：**15分钟代理实时结算层**
- 上层采用 **PPO**
- 下层采用 **显式规则机制**
- 15分钟层**不作为动作层**，只作为结算和回测层

### 2.3 本次明确禁止
以下内容不得进入本次主实现：
- `rl-baselines3-zoo`
- SAC
- A2C、DDPG、TD3、DQN 作为正式主模型
- 第二个独立 DRL 下层模型
- 在线学习
- LLM 政策评分
- 随机伪造主数据
- 与规划不一致的额外动作层
- 擅自把周度决策改回月度或改成日度

### 2.4 模块划分固定为四个模块
1. **基础环境模块**
2. **算法应用模块**
3. **运行结果与日志输出模块**
4. **检验模块**（包含验证、回测、敏感性分析、鲁棒性分析、参数搜索记录）

### 2.5 制度与样本口径
- 研究主样本期：`2025-11-01 00:00:00` 至 `2026-03-20 23:45:00`
- 政策事件变量按**生效时间**编码，不按发布时间编码
- 不得将后期政策口径前移到前期样本
- 至少保留两个政策事件哑变量：
  - `policy_event_20260101`
  - `policy_event_20260201`

---

## 3. 必须先做环境检查

### 3.1 先检查环境
先执行：
```bash
mamba env list
```

### 3.2 若无 `elec_env` 则创建
```bash
mamba create -n elec_env python=3.11 -y
```

### 3.3 激活与安装依赖
```bash
source "$(mamba info --base)/etc/profile.d/conda.sh"
mamba activate elec_env
pip install -U pip
pip install pandas numpy scipy scikit-learn matplotlib pyyaml tqdm tensorboard gymnasium stable-baselines3 torch openpyxl
```

### 3.4 依赖原则
- 依赖应尽量精简，只保留项目真正使用的包
- 必须生成 `requirements.txt`
- 不得加入与当前实现无关的大型依赖
- 不得要求用户手动安装一批未记录的隐含依赖

---

## 4. 推荐目录结构

```text
/Users/dk/py/elec
├── total.csv
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── data.yaml
│   ├── ppo.yaml
│   ├── rules.yaml
│   ├── backtest.yaml
│   └── analysis.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   ├── feature_engineering.py
│   │   ├── weekly_builder.py
│   │   └── scenario_generator.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── elec_env.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── train_ppo.py
│   ├── rules/
│   │   ├── __init__.py
│   │   └── hourly_hedge.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── simulator.py
│   │   ├── settlement.py
│   │   ├── metrics.py
│   │   └── benchmarks.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── sensitivity.py
│   │   └── robustness.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── logger.py
│   │   ├── plotting.py
│   │   └── seeds.py
│   └── scripts/
│       ├── run_pipeline.py
│       ├── train.py
│       ├── evaluate.py
│       ├── backtest.py
│       └── diagnostics.py
├── outputs/
│   ├── logs/
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   └── reports/
└── notebooks/
```

说明：
- Python 运行环境叫 `elec_env`
- 自定义 Gymnasium 环境类命名为 `ElecEnv`
- 环境注册 id 建议使用 `ElecEnv-v0`
- 不得将 Python 环境名与 Gym id 混淆
- 代码中路径尽量相对化，根目录使用统一的项目路径解析函数

---

## 5. 数据入口、检查与频率处理

## 5.1 主数据文件位置
按以下顺序寻找真实数据文件：
1. `/Users/dk/py/elec/total.csv`
2. `/Users/dk/py/elec/data/total.csv`
3. 在项目根目录下递归查找精确文件名 `total.csv`

若未找到，必须报错并停止。不得自动生成伪造主数据。

## 5.2 当前 `total.csv` 已知结构
当前主数据包含 19 列：
```text
datetime
省调负荷_日前
新能源负荷-总加_日前
新能源负荷-光伏_日前
新能源负荷-风电_日前
联络线总加_日前
非市场化机组出力_日前
水电出力_日前
全网总出力_日前
全网统一出清价格_日前
省调负荷_日内
新能源负荷-总加_日内
新能源负荷-光伏_日内
新能源负荷-风电_日内
联络线总加_日内
非市场化机组出力_日内
水电出力_日内
全网总出力_日内
全网统一出清价格_日内
```

## 5.3 必须执行的数据质量检查
在代码中自动完成以下检查，并写入报告：
1. 将 `datetime` 转为 `datetime64[ns]`
2. 全表按 `datetime` 升序排序
3. 检查重复时间戳
4. 检查缺失时间戳
5. 检查数值列缺失值
6. 输出 `outputs/reports/data_quality_report.md`

### 5.3.1 当前已知数据问题
- 数据频率为 **15 分钟**
- 样本覆盖从 `2025-11-01 00:00:00` 到 `2026-03-28 23:45:00`
- 存在 **1 个重复时间戳**：`2026-01-01 00:00:00`
- 排序后不存在缺失 15 分钟时点

### 5.3.2 处理规则
- 对重复时间戳，按 `datetime` 分组后对数值列取均值，仅保留一条记录
- 主样本仅保留 `2025-11-01 00:00:00` 至 `2026-03-20 23:45:00`
- `2026-03-21` 之后的数据可以作为缓冲数据读取，但不得进入正式报告指标

## 5.4 频率分层原则
- **15分钟层**：原始频率，作为结算与回测层
- **小时层**：由15分钟聚合得到，作为小时级规则修正层
- **周度层**：由小时级进一步聚合得到，作为 PPO 决策层

### 5.4.1 小时级聚合
对15分钟数据按自然小时聚合：
- 价格类：取4个15分钟点均值
- 功率、负荷、出力类：取4个15分钟点均值作为小时状态量
- 能量口径：平均功率 × 1小时，或4个15分钟能量求和，两者必须统一

### 5.4.2 周度聚合
- 周边界采用自然周口径：周一 `00:00:00` 至周日 `23:45:00`
- 首周和末周若不完整，可保留，但必须单独标识 `is_partial_week`
- 周度状态特征由该周内小时级序列聚合得到
- 周度目标需求由该周日前净负荷小时序列累计得到
- 不得继续使用“月度聚合 -> PPO 决策”的旧逻辑

---

## 6. 必须实现的派生变量

至少构造以下字段：
- `net_load_da = 省调负荷_日前 - 新能源负荷-总加_日前`
- `net_load_id = 省调负荷_日内 - 新能源负荷-总加_日内`
- `price_spread = 全网统一出清价格_日内 - 全网统一出清价格_日前`
- `load_dev = net_load_id - net_load_da`
- `wind_dev = 新能源负荷-风电_日内 - 新能源负荷-风电_日前`
- `solar_dev = 新能源负荷-光伏_日内 - 新能源负荷-光伏_日前`
- `renewable_ratio_da = 新能源负荷-总加_日前 / 省调负荷_日前`
- `renewable_ratio_id = 新能源负荷-总加_日内 / 省调负荷_日内`

要求：
- 分母为零时必须保护
- 保护方式写成统一函数
- 派生变量输出到清洗后的中间数据集中
- 所有新列必须在 `feature_manifest` 中登记，便于追踪

---

## 7. 中长期价格、零售价格、实时价格缺失时的处理

## 7.1 中长期价格
当前 `total.csv` 中未直接提供真实中长期合同价格列。因此需要实现**可替换估算口径**：

默认周度合约价格定义为：
```text
lt_price_w = 上一自然周 全网统一出清价格_日前 的小时均价
```

要求：
- 周度层先按小时聚合，再按周取均值
- 样本最前一周无上一周历史时，标为 warm-up 周
- warm-up 周可用于初始化，但不纳入正式比较结果
- 若后续补充真实中长期价格列，则代码必须优先读取真实列，自动覆盖估算列

## 7.2 零售侧收益
由于当前数据缺少零售价格，**主评价口径采用采购侧成本最小化**，而不是完整利润最大化。

即：
- 训练奖励围绕采购成本与风险惩罚构造
- 回测输出围绕策略采购成本、相对基准成本节约、风险调整后成本改进展开
- 不得伪造零售价格曲线去构造利润

## 7.3 实时结算口径
当前数据未单列真实实时价格，因此：
- 使用**日内15分钟价格序列**作为近实时代理结算价格
- 所有报告中必须明确标注“15分钟日内代理结算口径”
- 若后续补入真实实时价格列，代码需支持无缝切换

---

## 8. 数学模型必须完整纳入实现

以下模型必须落实到代码和交付说明中，不能只停留在文字层面。

## 8.1 上层周度中长期持仓模型

### 8.1.1 周度目标中长期净持仓
\[
Q_{w}^{LT,target}=h_w \cdot \hat{L}_w
\]

其中：
- \(Q_{w}^{LT,target}\)：第 \(w\) 周目标中长期净持仓
- \(h_w\)：第 \(w\) 周锁定比例，属于动作变量
- \(\hat{L}_w\)：第 \(w\) 周预测净需求总量

### 8.1.2 周度统一风险项
\[
Risk_w=\alpha \sigma_w+(1-\alpha)CVaR_w
\]

其中：
- \(\sigma_w\)：第 \(w\) 周成本或收益波动率
- \(CVaR_w\)：第 \(w\) 周尾部风险
- \(\alpha\)：风险加权参数

### 8.1.3 周度交易成本
\[
TransCost_w
=
c_1 \sum_{t \in w} |\Delta q_t|
+
c_2 \sum_{t \in w} (\Delta q_t)^2
+
c_3 |Q^{LT,target}_w-Q^{LT,prev}_w|
\]

### 8.1.4 周度奖励函数
\[
r_w=-Cost_w-\lambda_1 Risk_w-\lambda_2 TransCost_w-\lambda_3 HedgeError_w
\]

要求：
- 奖励函数最终在环境中返回标量
- 奖励各组成项必须在 `info` 中分项返回
- 报告中必须输出成本项、风险项、交易成本项和套保误差项的周度分解表

## 8.2 下层小时级现货滚动修正模型

### 8.2.1 小时级基准申报曲线
\[
q_{base,t}=\hat{l}_t-q_t^{LT}
\]

### 8.2.2 理论现货需求
\[
Q_{t}^{spot,need}=\max(\hat{l}_t-q_t^{LT},0)
\]

### 8.2.3 小时级修正电量
\[
\Delta q_t=a_t \cdot q_{base,t}
\]

### 8.2.4 修正后现货申报电量
\[
q_t^{spot}=q_{base,t}+\Delta q_t
\]

### 8.2.5 小时级套保误差
\[
HedgeError_w=\frac{1}{T_w}\sum_{t\in w}|Q_{t}^{spot,adj}-Q_{t}^{spot,need}|
\]

要求：
- `a_t` 由规则模块显式给出，不由第二个 RL 模型学习
- 规则必须可配置，至少支持价差、负荷偏差、新能源偏差三类信号
- 规则执行轨迹必须可记录，至少输出每小时：
  - `q_base`
  - `a_t`
  - `delta_q`
  - `q_spot`
  - `spot_need`
  - `signal_spread`
  - `signal_load_dev`
  - `signal_renewable_dev`

## 8.3 状态变量与决策映射

### 8.3.1 政策风险指标
\[
PolicyRisk_w=D_w^{policy}
\]

### 8.3.2 上层周度配置决策函数
\[
Q_{w}^{LT,target}, h_w=f(Spread_w, PolicyRisk_w, NewEnergyVol_w, Cost_w)
\]

### 8.3.3 下层小时级修正决策函数
\[
\Delta q_t=g(Spread_t, LoadGap_t, NewEnergyRisk_t)
\]

要求：
- `f` 由 PPO 策略网络逼近
- `g` 由显式规则函数实现
- 两者边界清楚，不得混合写成一个不可解释黑箱

## 8.4 约束条件

### 8.4.1 周度锁定比例约束
\[
h_{min}\leq h_w\leq h_{max}
\]

### 8.4.2 周度持仓调整幅度约束
\[
|h_w-h_{w-1}|\leq \Delta h_{max}
\]

或
\[
|Q_{w}^{LT,target}-Q_{w}^{LT,prev}|\leq \Delta Q_{LT,max}
\]

### 8.4.3 小时级修正系数约束
\[
a_{min}\leq a_t\leq a_{max}
\]

### 8.4.4 申报非负约束
\[
q_t^{spot}\geq 0
\]

### 8.4.5 相邻时段平滑约束
\[
|\Delta q_t-\Delta q_{t-1}|\leq \gamma_{max}
\]

### 8.4.6 周度 CVaR 预算约束
\[
CVaR_w\leq CVaR_{budget}
\]

要求：
- 这些约束要在环境或规则层明确实现
- 约束违反时必须采用一致的处理机制：裁剪、惩罚、或同时执行
- 处理方式写入配置和日志，不得隐式处理

## 8.5 基准策略模型

### 8.5.1 固定锁定比例策略
\[
h_w=h^{fix}
\]
\[
Q_{w}^{LT,target}=h^{fix}\hat{L}_w
\]
\[
a_t=0
\]

### 8.5.2 仅中长期持仓调整策略
\[
Q_{w}^{LT,target}=h_w\hat{L}_w
\]
\[
a_t=0
\]

### 8.5.3 规则型现货对冲策略
\[
a_t=
\begin{cases}
+\kappa, & Spread_t>\theta_1 \text{ or } Gap_t>\theta_2 \\
-\kappa, & Spread_t<-\theta_1 \text{ or } Gap_t<-\theta_2 \\
0, & \text{otherwise}
\end{cases}
\]

要求：
- 三类基准策略都必须能独立运行
- 回测时必须与 PPO 主策略使用同一批数据、同一结算口径、同一评价指标
- 不得只输出主策略结果而忽略基准策略

---

## 9. 基础环境模块要求

## 9.1 必做文件
- `src/data/loader.py`
- `src/data/preprocess.py`
- `src/data/feature_engineering.py`
- `src/data/weekly_builder.py`
- `src/data/scenario_generator.py`
- `src/envs/elec_env.py`

## 9.2 环境定义
- 环境类：`ElecEnv`
- Gym id：`ElecEnv-v0`
- 一个 `step` 对应一个**周度决策期**

## 9.3 观测空间最低要求
观测向量至少包含：
- 上一周日前价格均值、标准差、分位数
- 上一周日内价格均值、标准差、分位数
- 上一周 `price_spread` 均值、标准差
- 上一周 `net_load_da` / `net_load_id` 均值、峰值、波动率
- 上一周 `load_dev`、`wind_dev`、`solar_dev` 统计量
- 上一周新能源占比统计
- 上一周联络线、水电、非市场化机组出力统计
- 周序列特征（如 weekofyear 的 sin / cos）
- 政策事件哑变量：
  - `policy_event_20260101`
  - `policy_event_20260201`
- 上一期动作
- 上一期奖励（可选）

## 9.4 动作空间最低要求
建议使用：
```python
Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]), shape=(2,))
```

两个动作分别为：
1. `lock_ratio`：周度中长期锁定比例
2. `hedge_intensity`：小时级规则修正强度或风险预算系数

说明：
- 原则上维持两维动作即可
- 不得无依据扩维
- 若实际实现为一维动作，必须在日志中解释原因，并同步更新基准策略与规则模块

## 9.5 环境内部执行流程
给定某周动作后，必须执行以下流程：
1. 读取该周输入状态
2. 根据 `lock_ratio` 计算 `Q_lt_target_w`
3. 将周度中长期目标电量分解到小时层
4. 调用小时级规则修正模块
5. 将小时结果映射回15分钟代理结算层
6. 计算该周采购成本
7. 计算风险项、交易成本项、套保误差项
8. 生成周度奖励
9. 返回下一周状态、奖励、终止标志和 `info`

## 9.6 周度目标需求
采用：
```text
forecast_weekly_net_demand_w = Σ(该周小时前净负荷)
Q_lt_target_w = lock_ratio_w * forecast_weekly_net_demand_w
```

## 9.7 样本不足问题
由于真实样本周数有限，直接训练 PPO 容易出现 episode 太短的问题，因此训练阶段允许：
- 在训练集内部使用 rolling resampling
- 或 block bootstrap 生成训练 episode

要求：
- 只能在训练集内部重采样
- 不得污染验证集与测试集
- 不得引入外部虚假数据
- 重采样逻辑必须写入 `src/data/scenario_generator.py`
- 报告中必须写明训练期样本构造方式

---

## 10. 算法应用模块要求

## 10.1 必做文件
- `src/agents/train_ppo.py`
- `src/scripts/train.py`
- `configs/ppo.yaml`

## 10.2 PPO 设置建议
至少支持以下配置项：
- `policy = "MlpPolicy"`
- `learning_rate`
- `n_steps`
- `batch_size`
- `gamma`
- `gae_lambda`
- `clip_range`
- `ent_coef`
- `vf_coef`
- `max_grad_norm`
- `total_timesteps`
- `seed`
- `device`

## 10.3 训练输出
训练完成后至少输出：
- 模型权重：`outputs/models/ppo_latest.zip`
- 最优模型：`outputs/models/ppo_best.zip`
- 训练配置快照：`outputs/reports/train_config_snapshot.yaml`
- 训练日志：`outputs/logs/train.log`
- TensorBoard 日志目录
- 训练摘要：`outputs/reports/train_summary.md`

## 10.4 训练摘要必须包含
- 训练数据起止时间
- 训练周数与重采样方式
- 关键超参数
- 最终训练轮次
- 奖励走势概览
- 是否使用 GPU
- 模型保存路径
- 异常与警告记录

---

## 11. 小时级规则修正模块要求

## 11.1 必做文件
- `src/rules/hourly_hedge.py`
- `configs/rules.yaml`

## 11.2 规则输入
至少包含：
- `q_base_t`
- `price_spread_t`
- `load_dev_t`
- `wind_dev_t`
- `solar_dev_t`
- `hedge_intensity`
- 阈值参数
- 上一时段修正量（用于平滑约束）

## 11.3 规则输出
至少包含：
- `a_t`
- `delta_q_t`
- `q_spot_t`
- 是否触发上限/下限裁剪
- 是否触发平滑约束

## 11.4 规则必须支持的逻辑
- 基于价差方向的修正
- 基于负荷偏差方向的修正
- 基于新能源偏差方向的修正
- 平滑约束
- 修正系数裁剪
- 非负约束

## 11.5 规则实现原则
- 规则应写成清晰函数，不要写成难以追踪的杂糅逻辑
- 每类信号的作用方向必须在注释中说明
- 所有参数必须配置化，不得把阈值写死在代码里

---

## 12. 回测与结算模块要求

## 12.1 必做文件
- `src/backtest/simulator.py`
- `src/backtest/settlement.py`
- `src/backtest/metrics.py`
- `src/backtest/benchmarks.py`
- `src/scripts/backtest.py`
- `configs/backtest.yaml`

## 12.2 结算流程
回测必须按以下逻辑完成：
1. 使用周度动作生成该周 `Q_lt_target`
2. 将周度目标分解至小时层
3. 应用小时级规则修正
4. 将小时申报映射到15分钟代理结算层
5. 使用15分钟日内价格序列结算
6. 汇总到小时、周度、全样本层
7. 输出总成本、风险指标与基准策略比较结果

## 12.3 至少输出的指标
- 总采购成本
- 平均周度成本
- 周度成本波动率
- CVaR
- 套保误差
- 相对固定锁定比例策略的成本节约
- 相对规则型现货对冲策略的成本节约

## 12.4 至少输出的文件
- `outputs/metrics/backtest_metrics.csv`
- `outputs/reports/backtest_summary.md`
- `outputs/reports/run_summary.md`
- `outputs/figures/cost_curve.png`
- `outputs/figures/weekly_reward_curve.png`
- `outputs/figures/benchmark_compare.png`

## 12.5 回测摘要必须包含
- 使用的模型版本
- 回测时间范围
- warm-up 周说明
- 结算口径说明
- 主策略与基准策略对比表
- 关键异常、裁剪次数、规则触发统计

---

## 13. 检验模块要求

## 13.1 验证
必须实现 `src/scripts/evaluate.py`，至少完成：
- 模型加载是否成功
- 环境构造是否成功
- 单轮 episode 是否能跑通
- 奖励是否为有限值
- 是否存在 NaN / inf

## 13.2 敏感性分析
至少考察：
1. 风险厌恶系数变化
2. 市场波动率强度变化
3. 价格限值变化

输出：
- `outputs/reports/sensitivity_report.md`
- `outputs/metrics/sensitivity_metrics.csv`

## 13.3 鲁棒性分析
至少考察：
1. 合约与现货配置比例扰动
2. 按政策边界拆分样本
3. 不同负荷预测误差水平

输出：
- `outputs/reports/robustness_report.md`
- `outputs/metrics/robustness_metrics.csv`

---

## 14. 脚本入口的硬性要求

## 14.1 `train.py`
必须支持：
```bash
python -m src.scripts.train
```

功能：
- 读取配置
- 读取并预处理数据
- 构建周度环境
- 训练 PPO
- 保存模型与训练摘要

## 14.2 `evaluate.py`
必须支持：
```bash
python -m src.scripts.evaluate
```

功能：
- 加载训练好的模型
- 在验证样本跑通至少一个 episode
- 输出验证摘要

## 14.3 `backtest.py`
必须支持：
```bash
python -m src.scripts.backtest
```

功能：
- 读取模型
- 在正式样本期回测
- 同时运行三类基准策略
- 输出指标、图表与报告

## 14.4 `run_pipeline.py`
必须支持：
```bash
python -m src.scripts.run_pipeline
```

功能：
- 按顺序执行 train → evaluate → backtest
- 将关键结果汇总到 `run_summary.md`

---

## 15. 日志、报告与可追踪性要求

## 15.1 日志
至少生成：
- `outputs/logs/pipeline.log`
- `outputs/logs/train.log`
- `outputs/logs/evaluate.log`
- `outputs/logs/backtest.log`

日志中至少记录：
- 启动时间
- 结束时间
- 使用配置
- 数据形状
- 周数、小时数、15分钟记录数
- 约束裁剪次数
- 异常信息

## 15.2 报告
至少生成：
- `data_quality_report.md`
- `train_summary.md`
- `validation_summary.md`
- `backtest_summary.md`
- `run_summary.md`

## 15.3 可追踪性
必须在报告中明确：
- 中长期价格是否为估算值
- 是否使用 GPU
- 训练集 / 验证集 / 回测集边界
- warm-up 周范围
- 规则参数
- 基准策略参数
- 代码版本或 git commit（若仓库可用）

---

## 16. 禁止 Codex 自行发挥的事项

以下部分不得自行更改：
1. 上层决策尺度是**周度**，不是月度、日度或小时级。
2. 下层是**显式规则机制**，不是第二个 RL。
3. 15分钟层只负责**结算与回测**，不是动作层。
4. 主评价口径是**采购侧成本 + 风险惩罚**，不是虚构利润。
5. 政策变量按**生效时间**，不是发布时间。
6. 主样本截止到 **2026-03-20 23:45:00**。
7. 不能用模拟主数据替换 `total.csv`。
8. 不能跳过基准策略比较。
9. 不能只给代码、不输出日志和报告。
10. 不能为了让结果好看而擅自修改样本区间、指标定义或结算口径。

---

## 17. 最低交付清单

完成后，至少应当可以直接看到以下结果：
- 一套可运行的 Python 项目目录
- 一个可用的 `elec_env` 环境依赖列表
- 可运行的训练、验证、回测脚本
- 训练好的 PPO 模型文件
- 三类基准策略回测结果
- 成本、波动率、CVaR、套保误差指标文件
- 日志文件
- 摘要报告文件
- 图表文件

---

## 18. 执行顺序建议

按以下顺序实施，不要跳步：
1. 环境检查与依赖安装
2. 读取 `total.csv`
3. 数据质量检查与清洗
4. 小时级聚合
5. 周度构造
6. 特征工程
7. 周度环境实现
8. 小时级规则实现
9. PPO 训练
10. 验证
11. 回测
12. 基准策略比较
13. 敏感性与鲁棒性分析
14. 统一输出报告

---

## 19. 最终原则

本文件是本次项目重建的唯一交付依据。若旧版总规划、旧版交付说明、旧版训练口径与本文件冲突，一律以本文件为准。实现时应尽量减少自由裁量空间，凡是涉及数据口径、模型结构、奖励定义、约束方式、脚本行为、输出内容的地方，都应以本文件写明的要求为准，不得自行改写成另一套体系。
