# 当前项目版本说明

## 1. 文档目的

本文档用于说明 `/Users/dk/py/elec` 当前工作树对应的项目版本状态、相较历史版本的迭代内容、现阶段算法流程与最新产出结果，便于后续汇报、论文撰写和版本交接。

## 2. 当前有效版本结论

当前仓库 `main` 分支最新提交为 `a49e35f`，提交信息为 `Update reporting outputs and practice data exports`。从版本演进看，项目经历了以下三个关键阶段：

- `v0.1`：初始交付版本，完成数据清洗、月度 PPO 决策、小时级规则修正、15 分钟代理结算回测，以及基础指标与图表输出。
- `v0.1.1`：曾引入自动设备选择能力，支持按 `cuda -> mps -> cpu` 顺序选择训练设备。
- 当前工作树状态：根据 [CHANGELOG.md](/Users/dk/py/elec/CHANGELOG.md)，GPU / CUDA / MPS 相关改动已撤回到 `v0.1` 基线，训练设备恢复为 `cpu`，但保留了中文日志、中文报告、中文图表标题，以及更完整的报告与实践数据导出能力。

因此，当前项目更准确的描述不是“持续启用 GPU 的 v0.1.1”，而是：

**以 `v0.1` 训练逻辑为基线、补强中文报告和结果导出的当前迭代版。**

## 3. 当前版本的核心定位

当前项目是一个基于真实 `total.csv` 数据构建的售电采购策略实验工程，核心目标是用强化学习与规则策略结合的方式，完成月度中长期持仓决策与后续执行评估。现版本重点不在硬件加速，而在以下几点：

- 保持训练和回测流程可复现。
- 输出适合论文或汇报直接引用的中文报告和统计表。
- 将训练、验证、回测、基准比较、敏感性、鲁棒性、超参数搜索统一到一条流水线中。

## 4. 当前版本的系统组成

项目当前主要由以下模块构成：

- 数据模块：`src/data/`
  负责读取 `total.csv`、完成去重与样本清洗、构造月度特征与数据质量报告。
- 环境模块：`src/envs/elec_env.py`
  定义基于月度序列的 Gymnasium 自定义环境，供 PPO 进行策略学习。
- 策略训练模块：`src/agents/train_ppo.py`
  基于 Stable-Baselines3 的 `PPO` 和 `MlpPolicy` 完成训练、评估、模型保存与训练指标导出。
- 规则修正模块：`src/rules/hourly_hedge.py`
  负责小时级规则型现货修正逻辑。
- 回测模块：`src/backtest/`
  负责 15 分钟代理结算、基准策略构造、指标计算、敏感性分析和鲁棒性分析。
- 流水线脚本：`src/scripts/`
  提供训练、验证、回测以及一键运行入口。
- 输出模块：`src/utils/`
  负责日志、图表、Markdown 报告、CSV 和 JSON 的统一落盘。

## 5. 相比早期版本的主要迭代

结合代码、日志和现有产出，当前版本相较初始版本的主要增强点如下：

### 5.1 报告能力增强

- 新增运行总结报告，输出到 `outputs/reports/run_summary.md`。
- 新增详细运行报告，输出到 `outputs/reports/detailed_run_report.md`。
- 新增数据质量报告，输出到 `outputs/reports/data_quality_report.md`。
- 新增超参数搜索总结，输出到 `outputs/reports/hparam_search_summary.md`。

这意味着项目已经从“给出模型和图表”升级到“可直接形成文字化实验说明”的状态。

### 5.2 实践数据导出增强

当前版本不仅保存模型和日志，还系统化导出：

- 训练指标与验证指标 CSV。
- 回测月度结果与基准比较表。
- 敏感性分析结果与鲁棒性分析结果。
- 算法应用效果相关统计表。

这些内容集中落在 `outputs/metrics/` 下，便于直接接入论文表格、附录整理或二次分析。

### 5.3 中文化输出增强

当前版本保留了中文日志、中文图表标题和中文 Markdown 报告。这一点对非技术汇报、课程论文和业务沟通更友好，降低了后续人工翻译与整理成本。

### 5.4 设备策略回归稳定实现

虽然 `v0.1.1` 曾加入自动设备选择逻辑，但当前代码中的训练实现已经固定使用 `cpu`。从现有代码看：

- [src/agents/train_ppo.py](/Users/dk/py/elec/src/agents/train_ppo.py) 中训练设备固定为 `cpu`。
- [outputs/reports/run_summary.md](/Users/dk/py/elec/outputs/reports/run_summary.md) 和 [outputs/reports/detailed_run_report.md](/Users/dk/py/elec/outputs/reports/detailed_run_report.md) 中的训练设备记录也均为 `cpu`。

这表明当前迭代更偏向稳定、可复现的实验口径，而不是追求 GPU 加速。

## 6. 当前版本的完整运行链路

现版本已经形成比较完整的一键实验流程，入口见 [README.md](/Users/dk/py/elec/README.md)：

```bash
python -m src.scripts.run_pipeline
python -m src.scripts.train
python -m src.scripts.evaluate
python -m src.scripts.backtest
```

其中 `run_pipeline` 会串联完成以下步骤：

1. 加载配置与原始数据。
2. 清洗 `total.csv` 并生成数据质量报告。
3. 构造月度特征和训练、验证、回测月份切分。
4. 训练 PPO 模型并记录训练过程。
5. 在验证集上评估策略。
6. 在回测集上与多类基准策略对比。
7. 输出图表、指标表、Markdown 报告和模型文件。

## 7. 当前版本使用的数据与样本划分

根据最新运行摘要，当前版本使用的数据口径如下：

- 主数据文件：`/Users/dk/py/elec/total.csv`
- 主样本区间：`2025-11-01 00:00:00` 至 `2026-03-20 23:45:00`
- 原始覆盖区间：`2025-11-01 00:00:00` 至 `2026-03-28 23:45:00`
- 原始记录数：`14209`
- 去重后记录数：`14208`
- 主样本记录数：`13440`
- 重复时间戳个数：`1`
- 去重后缺失 15 分钟时点数：`0`

样本划分为：

- 预热期：`2025-11`
- 训练集：`2025-12`、`2026-01`
- 验证集：`2026-02`
- 回测集：`2026-03`

## 8. 当前版本的模型配置

当前主模型为：

- 算法：`PPO`
- 策略网络：`MlpPolicy`
- 训练设备：`cpu`
- 总训练步数：`4096`
- `learning_rate=0.0003`
- `n_steps=256`
- `batch_size=64`
- `n_epochs=10`
- `gamma=0.99`
- `ent_coef=0.01`
- `seed=42`

配置快照已保存于 [outputs/logs/config_snapshot.yaml](/Users/dk/py/elec/outputs/logs/config_snapshot.yaml)。

## 9. 当前版本的最新结果摘要

### 9.1 验证集结果

- 累计采购成本：`5795511288.79`
- CVaR(95%)：`4131545.29`
- 平均奖励：`-5795.759786`

### 9.2 回测集 PPO 结果

- 累计采购成本：`4031736315.64`
- 成本波动率：`0.00`
- CVaR(95%)：`3852182.79`
- 套保误差：`228.5880`
- 月均调整量：`0.0000`

### 9.3 与基准策略对比

根据 [outputs/reports/detailed_run_report.md](/Users/dk/py/elec/outputs/reports/detailed_run_report.md)：

- `fixed_lock` 累计采购成本为 `4022879473.31`
- `dynamic_lock_only` 累计采购成本为 `4024264506.39`
- `ppo` 累计采购成本为 `4031736315.64`
- `rule_only` 累计采购成本为 `4659092735.10`

从当前这轮实验结果看，PPO 已明显优于 `rule_only`，但尚未优于 `fixed_lock` 与 `dynamic_lock_only` 两个较强基准。这说明当前版本已具备完整实验能力，但策略表现仍有继续优化空间。

## 10. 当前版本新增或强化的输出物

当前版本运行后会在以下目录形成较完整的成果集：

- `outputs/logs/`：训练监控、评估日志、TensorBoard 事件、配置快照。
- `outputs/models/`：最终模型、最佳模型、训练检查点。
- `outputs/metrics/`：训练指标、验证指标、回测指标、月度结果、基准比较、敏感性与鲁棒性结果。
- `outputs/figures/`：奖励曲线、损失曲线、累计成本曲线、套保误差曲线、敏感性与鲁棒性图表。
- `outputs/reports/`：运行总结、详细运行报告、数据质量报告、超参数搜索总结。

与较早版本相比，当前版本最大的价值是：**结果不仅能跑出来，而且能较完整地沉淀为可复核、可引用、可展示的实验材料。**

## 11. 当前环境链路机理说明

这一部分说明当前项目从数据进入、状态构造、动作生成、执行修正到成本结算的完整机理。

### 11.1 总体链路

当前版本的环境链路可以概括为：

1. 原始 `15min` 电力与价格数据进入清洗流程。
2. 构造净负荷、价差、预测偏差、新能源偏差等派生变量。
3. 将 `15min` 数据聚合到小时级和月度级。
4. 用“上一个月的统计特征”构造“当前月”的状态向量。
5. PPO 在月度粒度输出两个动作：
   `lock_ratio` 和 `hedge_intensity`。
6. `lock_ratio` 决定本月中长期合约目标电量。
7. `hedge_intensity` 决定小时级规则修正的强弱。
8. 小时级修正再被分摊回 `15min` 结算颗粒度。
9. 按中长期、日前、日内、偏差惩罚、交易费用完成成本核算。
10. 基于成本、风险、交易成本和套保误差构造奖励函数，反馈给 PPO。

因此，当前系统本质上是一个“两层控制结构”：

- 第一层：月度强化学习决策。
- 第二层：小时级规则执行修正。

### 11.2 数据流结构

对应代码链路如下：

- 数据读取与清洗：[src/scripts/common.py](/Users/dk/py/elec/src/scripts/common.py)
- 特征工程：[src/data/feature_engineering.py](/Users/dk/py/elec/src/data/feature_engineering.py)
- 环境定义：[src/envs/elec_env.py](/Users/dk/py/elec/src/envs/elec_env.py)
- 月度仿真与结算：[src/backtest/simulator.py](/Users/dk/py/elec/src/backtest/simulator.py)
- 风险指标：[src/backtest/metrics.py](/Users/dk/py/elec/src/backtest/metrics.py)
- 小时级规则修正：[src/rules/hourly_hedge.py](/Users/dk/py/elec/src/rules/hourly_hedge.py)

## 12. 数学模型与公式

### 12.1 基础变量定义

在第 \(t\) 个 `15min` 时段，定义：

- 日前省调负荷：\(L^{da}_t\)
- 日内省调负荷：\(L^{id}_t\)
- 日前新能源负荷：\(R^{da}_t\)
- 日内新能源负荷：\(R^{id}_t\)
- 日前价格：\(P^{da}_t\)
- 日内价格：\(P^{id}_t\)

则净负荷定义为：

\[
N^{da}_t = L^{da}_t - R^{da}_t
\]

\[
N^{id}_t = L^{id}_t - R^{id}_t
\]

价差定义为：

\[
S_t = P^{id}_t - P^{da}_t
\]

负荷偏差定义为：

\[
\Delta N_t = N^{id}_t - N^{da}_t
\]

风电、光伏偏差分别记为：

\[
\Delta W_t = W^{id}_t - W^{da}_t
\]

\[
\Delta S_t^{solar} = S^{id,solar}_t - S^{da,solar}_t
\]

新能源总偏差定义为：

\[
\Delta R_t = \Delta W_t + \Delta S_t^{solar}
\]

以上变量对应实现见 [src/data/feature_engineering.py](/Users/dk/py/elec/src/data/feature_engineering.py)。

### 12.2 月度状态空间

当前环境是按“月”推进的。对目标月份 \(m\)，状态不是直接使用当月未来信息，而是使用“前一月”的统计特征来构造观测。

设 \(m-1\) 为目标月的上一自然月，则状态向量可写为：

\[
s_m = \Big[
\phi_{m-1}^{price},
\phi_{m-1}^{spread},
\phi_{m-1}^{load},
\phi_{m-1}^{renewable},
\phi_{m-1}^{system},
\sin(2\pi \cdot month/12),
\cos(2\pi \cdot month/12),
\text{policy\_events},
a_{m-1},
r_{m-1}
\Big]
\]

其中：

- \(\phi_{m-1}^{price}\) 包含上月日前、日内价格的均值、标准差、分位数、最大值。
- \(\phi_{m-1}^{spread}\) 包含上月价差统计量。
- \(\phi_{m-1}^{load}\) 包含上月净负荷统计量。
- \(\phi_{m-1}^{renewable}\) 包含风光偏差和新能源占比统计量。
- \(\phi_{m-1}^{system}\) 包含联络线、水电、非市场化机组等统计量。
- \(a_{m-1}\) 为上月动作，即前一月的 `lock_ratio` 与 `hedge_intensity`。
- \(r_{m-1}\) 为上月奖励。

代码中，最终 observation 由：

- 月度特征
- `prev_lock_ratio`
- `prev_hedge_intensity`
- `prev_reward`

共同拼接得到，见 [src/envs/elec_env.py](/Users/dk/py/elec/src/envs/elec_env.py)。

### 12.3 动作空间

当前 PPO 每个月输出二维动作：

\[
a_m = [\alpha_m,\ \beta_m]
\]

其中：

- \(\alpha_m \in [0,1]\)：月度中长期锁定比例，对应 `lock_ratio`
- \(\beta_m \in [0,1]\)：小时级规则修正强度，对应 `hedge_intensity`

动作空间为二维连续区间：

\[
\mathcal{A} = [0,1]^2
\]

### 12.4 中长期合约目标电量

设月份 \(m\) 的日前预测月度净需求电量为：

\[
Q^{forecast}_m = \sum_{t \in m} N^{da}_t \cdot \Delta t
\]

当前项目中 \(\Delta t = 0.25\) 小时。

则月度中长期目标持仓电量定义为：

\[
Q^{lt}_m = \alpha_m \cdot Q^{forecast}_m
\]

这对应代码中的：

- `forecast_monthly_net_demand_mwh`
- `q_lt_target_mwh`

### 12.5 中长期电量按小时分配

为使中长期持仓与负荷结构匹配，系统将 \(Q^{lt}_m\) 按小时正向预测负荷比例分配。

设小时 \(h\) 的预测净负荷电量为：

\[
q^{forecast}_{m,h}
\]

则小时中长期分配量为：

\[
q^{lt}_{m,h} =
\begin{cases}
Q^{lt}_m \cdot \dfrac{\max(q^{forecast}_{m,h},0)}{\sum_h \max(q^{forecast}_{m,h},0)}, & \text{若分母}>0 \\
0, & \text{否则}
\end{cases}
\]

即只按正的预测需求做比例分配，避免负值负荷导致分摊失真。

### 12.6 小时级剩余敞口

分配完中长期持仓后，小时级剩余敞口为：

\[
E^{res}_{m,h} = q^{forecast}_{m,h} - q^{lt}_{m,h}
\]

这个剩余敞口将作为小时级规则修正的基数。

### 12.7 小时级规则修正信号

小时级规则修正由三个标准化信号加权组成：

- 价差信号
- 负荷偏差信号
- 新能源偏差信号

对任意时间序列 \(x\)，其标准化定义为：

\[
z(x) = \frac{x - \mu(x)}{\sigma(x)}
\]

若标准差为 0，则直接记为 0。

于是三个信号分别为：

\[
z^{spread}_{m,h} = z(S_{m,h} \cdot \eta_{mv})
\]

\[
z^{load}_{m,h} = z(\Delta N_{m,h} \cdot \eta_{fe})
\]

\[
z^{ren}_{m,h} = z(\Delta R_{m,h} \cdot \eta_{fe})
\]

其中：

- \(\eta_{mv}\) 为市场波动缩放系数 `market_vol_scale`
- \(\eta_{fe}\) 为预测误差缩放系数 `forecast_error_scale`

综合原始信号为：

\[
u_{m,h} =
w_1 z^{spread}_{m,h}
 w_2 z^{load}_{m,h}
 w_3 z^{ren}_{m,h}
\]

其中：

- \(w_1 =\) `price_spread_weight`
- \(w_2 =\) `load_dev_weight`
- \(w_3 =\) `renewable_dev_weight`

然后做截断：

\[
\tilde{u}_{m,h} = \text{clip}(u_{m,h}, -1, 1)
\]

### 12.8 价格门控机制

为了避免在极端高价时段过度调节，项目加入价格门控。

设日内价格的高分位阈值为：

\[
\bar{P}^{id}_{q} = Q_q(P^{id}_{m,h})
\]

其中 \(q=\) `price_gate_quantile`。

则门控系数定义为：

\[
g_{m,h} =
\begin{cases}
\rho, & P^{id}_{m,h} > \bar{P}^{id}_{q} \cdot \eta_{pc} \\
1, & \text{否则}
\end{cases}
\]

其中：

- \(\rho =\) `price_gate_penalty`
- \(\eta_{pc} =\) `price_cap_multiplier`

### 12.9 小时级修正量公式

最终小时级修正量为：

\[
H_{m,h} =
\tilde{u}_{m,h}
\cdot \beta_m
\cdot \eta_{hi}
\cdot \kappa
\cdot E^{res}_{m,h}
\cdot g_{m,h}
\]

其中：

- \(\beta_m\) 为 PPO 输出的 `hedge_intensity`
- \(\eta_{hi}\) 为 `hedge_intensity_scale`
- \(\kappa\) 为 `max_adjust_share`

这就是 [src/rules/hourly_hedge.py](/Users/dk/py/elec/src/rules/hourly_hedge.py) 的核心公式。

### 12.10 从小时修正到 15 分钟结算

小时级修正量平均分摊到四个 `15min` 时段：

\[
H_{m,t}^{(15min)} = \frac{H_{m,h}}{4}, \quad t \in h
\]

中长期小时分配量则按该小时内各 `15min` 的正向预测负荷权重进一步分摊：

\[
Q^{lt}_{m,t} =
\begin{cases}
q^{lt}_{m,h} \cdot \dfrac{w_{m,t}}{\sum_{t \in h} w_{m,t}}, & \sum_{t \in h} w_{m,t} > 0 \\
0, & \text{否则}
\end{cases}
\]

其中：

\[
w_{m,t} = \max(q^{forecast}_{m,t}, 0)
\]

### 12.11 15 分钟结算量定义

对每个 `15min` 时段 \(t\)，定义：

- 预测电量：
\[
q^{forecast}_{m,t} = N^{da}_t \cdot 0.25
\]

- 实际电量：
\[
q^{actual}_{m,t} = N^{id}_t \cdot 0.25
\]

- 日前剩余电量：
\[
q^{da}_{m,t} = q^{forecast}_{m,t} - Q^{lt}_{m,t}
\]

- 偏差电量：
\[
q^{imb}_{m,t} = q^{actual}_{m,t} - q^{forecast}_{m,t} - H_{m,t}^{(15min)}
\]

其中 \(q^{imb}_{m,t}\) 表示在中长期和修正动作执行后，最终仍需承担偏差惩罚的量。

### 12.12 成本函数

当前版本的成本由五部分构成。

1. 中长期成本：

\[
C^{lt}_{m,t} = Q^{lt}_{m,t} \cdot P^{lt}_m
\]

其中 \(P^{lt}_m\) 是月度中长期价格。当前实现中，若无真实中长期列，则采用“上一个自然月日前均价”估算：

\[
P^{lt}_m = \overline{P^{da}_{m-1}}
\]

2. 日前采购成本：

\[
C^{da}_{m,t} = q^{da}_{m,t} \cdot P^{da}_t
\]

3. 日内修正成本：

\[
C^{id}_{m,t} = H^{(15min)}_{m,t} \cdot P^{id}_t
\]

4. 偏差惩罚成本：

\[
C^{pen}_{m,t} = |q^{imb}_{m,t}| \cdot P^{id}_t \cdot \lambda_{pen}
\]

其中 \(\lambda_{pen} =\) `penalty_multiplier`。

5. 交易费用：

\[
C^{tc}_{m,t} = |H^{(15min)}_{m,t}| \cdot c_{fee}
\]

其中 \(c_{fee} =\) `transaction_fee_rate`。

于是采购成本定义为：

\[
C^{proc}_{m,t} = C^{lt}_{m,t} + C^{da}_{m,t} + C^{id}_{m,t} + C^{pen}_{m,t}
\]

总成本定义为：

\[
C^{total}_{m,t} = C^{proc}_{m,t} + C^{tc}_{m,t}
\]

月度累计采购成本为：

\[
C^{proc}_m = \sum_{t \in m} C^{proc}_{m,t}
\]

月度交易成本为：

\[
C^{tc}_m = \sum_{t \in m} C^{tc}_{m,t}
\]

### 12.13 风险项

当前版本的风险项由“成本波动”和“尾部风险”线性组合而成。

1. 月内总成本标准差：

\[
R^{vol}_m = \sigma(C^{total}_{m,t})
\]

2. 月内总成本的 \(CVaR_{95\%}\)：

先定义 95% 分位点：

\[
\tau_{0.95} = Q_{0.95}(C^{total}_{m,t})
\]

则：

\[
R^{cvar}_m = \mathbb{E}[C^{total}_{m,t} \mid C^{total}_{m,t} \ge \tau_{0.95}]
\]

3. 综合风险项：

\[
R_m = \omega_{vol} R^{vol}_m + \omega_{cvar} R^{cvar}_m
\]

其中：

- \(\omega_{vol} =\) `risk_vol_weight`
- \(\omega_{cvar} =\) `risk_cvar_weight`

### 12.14 奖励函数

当前 PPO 的月度奖励函数为：

\[
r_m = - \frac{
C^{proc}_m
 \lambda_r R_m
 \lambda_{tc} C^{tc}_m
 \lambda_{he} E^{he}_m
}{K}
\]

其中：

- \(C^{proc}_m\)：月度累计采购成本
- \(R_m\)：风险项
- \(C^{tc}_m\)：月度交易成本
- \(E^{he}_m\)：套保误差
- \(K=\) `reward_scale`

当前实现中的套保误差定义为：

\[
E^{he}_m = \frac{1}{T_m}\sum_{t \in m} |q^{imb}_{m,t}|
\]

也就是 `15min` 偏差电量绝对值的月内平均。

这说明当前强化学习目标本质上是一个多目标加权最小化问题，即：

\[
\min \Big( \text{采购成本} + \text{风险} + \text{交易成本} + \text{偏差误差} \Big)
\]

### 12.15 MDP 视角下的环境定义

从强化学习角度，当前环境可以写成一个有限期 MDP：

- 状态：\(s_m\)
- 动作：\(a_m=[\alpha_m,\beta_m]\)
- 状态转移：由月份顺序推进
- 奖励：\(r_m\)
- 终止条件：月份序列遍历结束

即：

\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, T)
\]

由于训练时使用的是训练月份的 block bootstrap 序列，实际 episode 并不是简单按自然月顺序单次遍历，而是在训练月份集合内进行块重采样，以增加有限样本下的训练暴露多样性。

### 12.16 Block Bootstrap 训练序列

设训练月份集合为：

\[
\mathcal{M}_{train} = \{m_1, m_2, \dots, m_n\}
\]

系统以块长 \(b\) 为单位随机抽取起点，并拼接形成长度为 \(L\) 的训练月份序列：

\[
\tilde{\mathcal{M}}_{train} = [m_{i_1}, m_{i_1+1}, \dots, m_{i_1+b-1}, m_{i_2}, \dots]
\]

直到达到设定长度 `train_sequence_length`。这样可以在小样本月份集上引入一定的序列扰动，同时保留相邻月份的局部相关性。

## 13. 基准策略机理

当前项目不仅评估 PPO，也同时构造三种基准策略：

### 13.1 固定锁定比例策略 `fixed_lock`

\[
\alpha_m = \alpha^{fixed}, \quad \beta_m = 0
\]

其中 \(\alpha^{fixed}=\) `fixed_lock_ratio`。

### 13.2 动态锁定比例策略 `dynamic_lock_only`

该策略只决定锁定比例，不做小时级修正：

\[
\beta_m = 0
\]

锁定比例定义为：

\[
\alpha_m =
\text{clip}\Big(
\alpha_0
- \delta_1 \min(\sigma^{spread}_{m-1}/100, 1)
+ \delta_2 \min(\bar{r}^{ren}_{m-1}, 1)
+ \delta_3 \cdot I^{policy}_{m},
0.05, 0.95
\Big)
\]

其中：

- \(\alpha_0=\) `dynamic_lock_base`
- \(\delta_1=\) `dynamic_lock_spread_penalty`
- \(\delta_2=\) `dynamic_lock_renewable_bonus`
- \(\delta_3=\) `dynamic_lock_policy_bonus`
- \(I^{policy}_m\) 是政策事件哑变量

### 13.3 纯规则策略 `rule_only`

\[
\alpha_m = 0,\quad \beta_m = \beta^{rule}
\]

其中 \(\beta^{rule}=\) `rule_only_intensity`。

这三类基准策略提供了“静态锁仓”“启发式月度锁仓”“纯执行层规则”的对照组，有助于判断 PPO 的增量价值究竟来自月度决策、执行修正，还是二者结合。

## 14. 指标体系公式

在回测汇总层，项目输出的主要指标包括：

1. 累计采购成本：

\[
\text{CumCost} = \sum_m C^{proc}_m
\]

2. 成本波动率：

\[
\text{VolCost} = \sigma(C^{proc}_m)
\]

3. 回测区间 CVaR(95%)：

\[
\text{CVaR}_{95} = \mathbb{E}[C^{total}_t \mid C^{total}_t \ge Q_{0.95}(C^{total}_t)]
\]

4. 平均套保误差：

\[
\text{HedgeError} = \frac{1}{M}\sum_m E^{he}_m
\]

5. 平均调整量：

\[
\text{AvgAdjust} = \frac{1}{M}\sum_m \overline{|H^{(15min)}_{m,t}|}
\]

6. 最大回撤：

项目中先把成本序列取负并做累计和视为“权益曲线”：

\[
Eq_k = -\sum_{i=1}^{k} C^{proc}_{m_i}
\]

则最大回撤为：

\[
\text{MDD} = \min_k \Big(Eq_k - \max_{j \le k} Eq_j\Big)
\]

7. 平均奖励：

\[
\text{MeanReward} = \frac{1}{M}\sum_m r_m
\]

## 15. 当前版本的已知边界

结合当前报告与实现，现版本仍有以下边界需要明确：

- 训练设备当前固定为 CPU，未实际保留 `v0.1.1` 的自动 GPU 训练能力。
- 中长期价格仍采用上一自然月日前小时均价估算，属于代理口径。
- 实时结算使用日内价格作为代理结算口径，仍不是完整真实结算链条。
- 当前 PPO 在本轮回测中未超过最优基准策略，说明策略空间、奖励设计和特征工程还有迭代空间。
- 回测集月份较少，目前主回测窗口仅覆盖 `2026-03`，统计稳定性仍可继续增强。

## 16. 对当前版本的简要评价

如果从“是否形成一个可交付、可复现、可汇报的实验项目”来看，当前版本已经基本成型。它的优势不是训练性能最强，而是：

- 流程完整。
- 输出规范。
- 中文报告齐备。
- 结果可追溯。
- 便于继续做论文整理和后续策略优化。

如果后续继续迭代，优先建议方向是：

- 恢复并验证自动设备选择逻辑，但前提是保证结果一致性。
- 优化奖励函数与动作空间，避免 PPO 长期退化为极低锁定比例策略。
- 拉长训练与回测窗口，提升结论稳健性。
- 基于 `outputs/metrics/` 的现有表结构继续扩展论文附图与统计检验。

## 17. 一句话版本说明

当前项目版本可以概括为：

**一个以 CPU 稳定训练为主、具备完整中文报告与实验导出能力的售电采购 PPO 实验迭代版。**
