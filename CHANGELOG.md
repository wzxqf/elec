# 更新日志

## v0.35

- 正式主线继续保持为双层 `HYBRID_PSO_V033`，不恢复 PPO / SB3 / gym 或其他上层 DRL 训练栈，`run_all.py` 仍作为全量项目的一键入口。
- 新增 `src/analysis/module1.py`、`src/analysis/excess_return.py`、`src/analysis/reporting.py`，将 `v0.35` 的评估层和分析层从训练主链中独立出来，避免把论文表达指标回灌到正式优化目标。
- 周度数据层补齐 `da_id_cross_corr_w`、`extreme_event_flag_w`、`extreme_price_spike_flag_w` 等基础字段，并在物化层新增 `lock_ratio_proxy_w`、`curve_match_score_w`、`stability_score_w`，用于模块1正式输出。
- 新增 `contract_value_weekly.csv` 与 `risk_factor_manifest.csv`，正式输出“中长期合约价值评估 + 五类风险因子”结果，不再停留在零散代理字段层面。
- 新增 `policy_risk_adjusted_metrics.csv` 与 `rolling_excess_return_metrics.csv`，正式输出政策风险调整后超额收益、窗口胜率、窗口最大回撤和政策风险调整后夏普比率。
- 新增 `module1_summary.md`、`market_mechanism_analysis.md`、`excess_return_validation_summary.md`，并同步扩充 `validation_summary.md`、`backtest_summary.md`、`rolling_audit_summary.md` 与 `run_summary.md`。
- 修复 `v0.35` 首轮集成中 `curve_match_score_w` 计算对整周 168 小时负荷曲线与 24 点合约曲线直接做差导致验证阶段中断的问题，现改为按 `hour_index % 24` 聚合到 24 点日内形状后再比较。
- 新增 `tests/test_v035_weekly_builder.py`、`tests/test_v035_materialize.py`、`tests/test_v035_module1_analysis.py`、`tests/test_v035_excess_return_analysis.py`、`tests/test_v035_script_outputs.py`，并同步更新 `run_all` 相关回归测试。
- 根配置升级为 `project.version: v0.35`，正式产物写入 `outputs/v0.35/<真实输出>`。

## v0.31

- 新增市场规则约束归集模块 `src/policy/market_constraints.py`，将中长期结算口径、24 时段合约曲线、96 点现货结算、零售侧单一售电公司、辅助服务边界、新能源机制电价、2026 年中长期价格联动和计量可追踪性归结为可查阅约束。
- 新增 `market_rule_constraints.md` 与 `market_rule_constraints.csv` 输出，运行后写入 `outputs/v0.31/reports/` 和 `outputs/v0.31/metrics/`，用于查阅“市场规则 -> 模型层 -> 字段/函数”的映射关系。
- 修正周度制度状态判定口径：当前制度状态改为按 `week_start` 生效，避免 2026-02-01 价格机制提前进入 2026-01-26 周的当前状态；前瞻制度状态继续保留切换倒计时。
- 新增 `tests/test_market_constraints.py`，回归校验 2026-01-01 新能源机制电价执行期、2026-02-01 中长期 40%/60% 价格联动和报告内容。
- 根配置升级为 `project.version: v0.31`，正式产物写入 `outputs/v0.31/<真实输出>`。

## v0.3

- 主算法从“周度 PPO + 小时级显式规则修正”切换为“双层 HPSO + 15 分钟代理结算回测”。
- 新增 `src/agents/hpso.py`，实现标准 PSO 更新、退火扰动、停滞扰动、BP 局部精修、固定随机种子复现、CUDA 优先与受控 CPU 降级。
- 上层 HPSO 搜索不设硬限的中长期合约调整量与诊断性 `exposure_bandwidth`，输出保留 `lock_ratio_base`、`delta_lock_ratio_raw`、`delta_lock_ratio` 与 `lock_ratio_final` 诊断字段。
- 下层 HPSO 搜索不设硬限的小时级现货合约修正量 `delta_q`，生成可映射到 15 分钟结算的完整小时轨迹；非负、带宽、平滑和辅助服务字段仅保留为诊断，不再硬裁剪模型输出。
- `v0.3.md` 新增本版数学公式，后续每个版本说明必须同步写入当版实际使用公式。
- 训练、验证、回测入口按 `training.algorithm` 分支，`HPSO` 路径不再加载 PPO 模型文件，PPO 依赖改为延迟导入。
- 根配置升级为 `project.version: v0.3`，新增 `hpso` 配置节，输出写入 `outputs/v0.3/<真实输出>`。
- 回测输出新增 `hpso_upper_weekly_actions.csv`、`hpso_hourly_delta_q.csv`、`hpso_convergence_curve.csv`，并继续导出政策清单、规则表、解析失败清单、滚动验证摘要和基准比较表。
- README 同步更新为 v0.3 HPSO 主线说明。

## v0.25

- 版本定位调整为纯工程性能优化，不改变周度底仓决策、小时级现货修正、15 分钟代理结算和 `dynamic_lock_only` 强基准口径。
- 新增回测运行时周级缓存，覆盖 `quarter`、`hourly`、`weekly_metadata`、`weekly_features` 与 `reward_reference` 的按周查找，并在缩放场景中自动重建缓存，避免读到旧切片。
- `apply_hourly_hedge_rule` 与 `settle_week` 收敛为基于 `numpy` 的批量计算路径，减少规则层和结算层的重复 `DataFrame` 列赋值开销。
- 搜索态训练在 `persist_artifacts=False` 时不再创建评估环境、monitor CSV 和 tensorboard 路径，降低滚动搜索的无效 I/O 与初始化耗时。
- 文档统一补充周度切片缓存、lookup/index 优化、重复 DataFrame 访问收敛、回测尾部瓶颈修复等提速方向，明确默认训练设备保持 `cpu`。
- 补充 `analysis.worker_count` 与 `search.worker_count` 的可选并行设置，默认仍保持单 worker，不改变策略语义。
- 补充 `mps` 为可选加速路径的说明，并同步记录子代理/模块归属信息，便于并行实施和后续追踪。

## v0.24

- 新增根目录一键全流程脚本 `run_all.sh`，固定通过 `mamba run -n elec_env python -m src.scripts.run_pipeline` 启动训练、验证、回测与报告导出。
- 脚本增加最小运行保护：校验 `experiment_config.yaml` 与 `mamba` 是否存在，并在启动前打印项目根目录、实验版本和输出目录。
- 新增 `--dry-run` 模式，便于只检查本次版本号、结果目录和真实执行命令而不实际启动流水线。
- 补充 `tests/test_run_all_script.py`，回归校验根目录脚本存在且固定绑定 `elec_env` 主入口。
- README 同步补充根目录一键入口说明，明确 `run_all.sh` 为当前推荐的正式启动方式。

## v0.23

- 新增根目录独立参数文件 `experiment_config.yaml`，训练、验证、回测、敏感性分析、鲁棒性分析和搜索流程统一改为从该文件加载；`configs/*.yaml` 保留为模板，不再作为正式实验主入口。
- 输出目录统一升级为 `outputs/<version>/<真实输出>` 结构，版本号由 `project.version` 自动决定，避免不同版本结果混放。
- 上层 PPO 第一维动作从“绝对底仓比例”改为相对 `dynamic_lock_only` 强基准的残差动作，输出补充 `lock_ratio_base`、`delta_lock_ratio_raw`、`delta_lock_ratio` 与 `lock_ratio_final`。
- 小时级规则层真正支持 `soft_clip` 连续压缩，规则轨迹新增 `delta_q_target`、`delta_q_after_smoothing`、`smoothing_mode`、`soft_clipped`，并汇总 `soft_clip_count`。
- 奖励参数按 `v0.23` 新口径调整为“成本改进优先、执行惩罚适度放松”，保持三层结构与 `tanh` 软压缩不变。
- 新增滚动验证与两阶段超参数搜索输出，包括 `rolling_validation_summary.md`、`rolling_validation_metrics.csv`、`hparam_search_results.csv` 和 `hparam_search_summary.md`。
- 状态空间增加“进入 PPO”与“仅用于报告”区分，导出 `feature_manifest.json` 与 `feature_manifest.csv`，并在训练/验证/总报告中记录实际进入 PPO 的特征集合。

## v0.22

- 按 2026-04-12 交付说明完成 `v0.22` 首轮迭代，主链切换到“周度底仓动作 + 边际敞口带宽动作 + 小时级带宽约束修正 + 15 分钟制度化代理结算”口径。
- 重写政策解析与制度状态构造，规则表新增 `state_group`，输出补充 `policy_metadata_index.csv`、`policy_parse_failures.csv`，并将单一倒计时升级为四组前瞻制度状态。
- 环境、规则层、回测层统一将第二动作从旧的 `hedge_intensity` 改为 `exposure_bandwidth`，小时级修正改为在制度状态和边际敞口带宽约束下执行。
- 奖励函数重写为三层结构：相对 `dynamic_lock_only` 强基准的成本改进、尾部风险相对超额、执行质量项，并继续使用 `tanh` 软压缩。
- 结算逻辑改为真正消费制度状态：2026-02 前采用上一自然周日前均价代理，2026-02 起采用“40% 日前固定价 + 60% 日内联动价”组合代理，并在输出中记录制度口径。
- 已重新运行完整流水线，训练、验证、回测、敏感性分析、鲁棒性分析和超参数搜索均已在 `elec_env` 中通过。

## v0.21

- 接入 `政策环境` 目录解析链路，新增 `src/policy/` 模块，自动扫描政策文本与竞价结果表，生成结构化规则表、政策文件清单、失败清单和周度制度状态轨迹。
- 政策状态不再仅使用简单事件哑变量，改为同时进入周度状态空间与结算说明，补充市场衔接、辅助服务耦合、新能源机制电价和中长期价格联动等制度状态。
- 2026 年 2 月起的中长期价格口径更新为“40% 日前固定价 + 60% 日内联动价”混合代理；2026 年 2 月前仍采用上一自然周日前均价代理。
- 奖励函数改为以 `dynamic_lock_only` 为强基准的相对奖励，采用训练集 robust 标准化、CVaR 相对超额软惩罚和 `tanh` 软压缩，不再使用旧的绝对成本硬裁剪写法。
- 新增并更新政策相关输出，包括 `policy_rule_summary.md`、`policy_state_trace.csv`、`policy_rule_table.csv`、`policy_file_inventory.csv`、`reward_reference_dynamic_baseline.csv`、`weekly_results.csv` 与 `hourly_rule_trace.csv`。

## v0.2

- 按 2026-04-10 交付说明完成全面重构，主链切换为“周度 PPO + 小时级显式规则 + 15 分钟代理结算”。
- 新增周度数据构建、15 分钟结算层、敏感性分析、鲁棒性分析、诊断脚本与论文写作用详细运行报告。
- 保留中文日志、中文报告和图表同名 CSV 导出机制。
- 清理旧月度产物与旧模型命名，仅保留当前周度结果与 `ppo_latest.zip` / `ppo_best.zip`。

## 当前工作树

- 已按 `v0.1` 基线撤销 GPU / CUDA / MPS 相关改动。
- 默认训练设备恢复为 CPU。
- 保留中文输出相关改动，包括中文日志、中文报告和中文图表标题。

## v0.1.1

- 新增训练设备自动选择逻辑，优先顺序为 `cuda -> mps -> cpu`。
- 在 Apple Silicon 且 `torch.backends.mps.is_available()` 为真时，PPO 训练可直接使用 MPS GPU。
- 运行摘要新增设备偏好、设备解析结果与可用性信息，便于复现。
- 基于 GPU 能力检测结果重新运行流水线，更新模型、日志、指标与报告产物。
- 补充说明：当前策略为 SB3 `MlpPolicy`，虽然可以在 GPU 上运行，但实际吞吐提升不保证，部分场景可能仍以 CPU 更稳妥。

## v0.1

- 初始交付版本。
- 完成真实 `total.csv` 数据清洗、月度 PPO 环境、小时级规则对冲、15 分钟代理结算回测。
- 输出训练日志、模型、图表、回测指标、敏感性分析、鲁棒性分析与参数搜索结果。
