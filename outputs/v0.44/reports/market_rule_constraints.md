> version: v0.44
> experiment_id: v0.44-27074b97
> config_hash: 27074b97aa4356c7e77a229df8467731880d6d750a2f01c2121fbaa50ffffed7
> run_timestamp: 2026-04-20T14:42:01+08:00
> device: cuda
> data_range: 2025-11-01 00:00:00 -> 2026-03-20 23:45:00

# 市场规则约束与模型映射

- 项目周期: 2025-11-01 00:00:00 至 2026-03-20 23:45:00
- 政策来源文件数: 13
- 结构化规则数: 34
- 口径说明: 以下约束只覆盖本项目主线市场范围，即中长期市场、日前现货、日内/实时现货和15分钟代理结算；不把辅助服务收益、需求响应收益或虚拟投标并入主策略收益。

## 一、核心结论

- 中长期合约仅作为结算依据，不作为物理调度计划。
- 周度上层动作只能决定中长期底仓残差和诊断性现货边际敞口带宽。
- 小时级现货修正是内部近似，正式回测按96点15分钟轨迹结算。
- 新能源机制电价状态只能自 2026-01-01 起进入执行期。
- 年度中长期价格 40% 固定 + 60% 实时联动机制只能自 2026-02-01 起进入结算代理。

## 二、约束清单

| constraint_id | constraint_name | effective_start | constraint_type | model_layer | model_mapping | policy_source_files |
| --- | --- | --- | --- | --- | --- | --- |
| lt_contract_settlement_only | 中长期合约仅作为结算依据 | 2025-10-15 00:00:00 | 结算口径硬约束 | 上层周度中长期底仓 + 15分钟结算 | lt_settlement_base, spot_marginal_exposure, lt_spot_coupling_state, q_lt_hourly, lt_energy_15m | 政策环境/2025.10.15湖南省电力现货市场交易实施细则.docx |
| lt_24_period_contract_curve | 中长期交易按每日24时段组织 | 2025-10-15 00:00:00 | 时间粒度硬约束 | 周度底仓向小时曲线分解 | _allocate_weekly_lt_to_hourly, q_lt_hourly |  |
| spot_15min_96_settlement | 日前/实时现货按96个15分钟时段结算 | 2025-10-15 00:00:00 | 结算时间粒度硬约束 | 15分钟代理结算回测 | settlement_interval_hours=0.25, settle_week, procurement_cost_15m, imbalance_energy_15m | 政策环境/2025.10.15湖南省电力现货市场交易实施细则.docx |
| retail_single_supplier_package | 零售用户合同周期内单一售电公司与单一套餐 | 2025-10-15 00:00:00 | 收益端边界约束 | 售电公司收益端与基准策略解释 | retail package is treated as exogenous revenue-side boundary |  |
| ancillary_peak_shaving_pause_during_spot_trial | 现货试运行期间省内调峰辅助服务暂停 | 2025-10-14 00:00:00 | 市场边界约束 | 下层小时级现货滚动修正 | ancillary_peak_shaving_pause, ancillary_freq_reserve_tight, bandwidth_multiplier | 政策环境/2025.10.14关于进一步加强湖南电力辅助服务市场建设有关事项的通知.docx |
| renewable_mechanism_from_2026_01_01 | 新能源机制电价自2026-01-01进入执行期 | 2026-01-01 00:00:00 | 制度时点硬约束 | 状态空间 + 新能源扰动代理 | renewable_mechanism_active, mechanism_stage_label, mechanism_price_floor, mechanism_price_ceiling, mechanism_volume_ratio_max | 政策环境/2025.11.05 关于2025年度新能源机制电价竞价工作有关事项的通知.docx |
| lt_price_linkage_from_2026_02_01 | 2026年度中长期价格40%固定 + 60%实时联动自2026-02执行 | 2026-02-01 00:00:00 | 制度时点硬约束 | 中长期价格代理与15分钟结算 | lt_price_linked_active, fixed_price_ratio_max, linked_price_ratio_min, resolve_settlement_context | 政策环境/2026.01 关于完善2026年度电力中长期交易价格机制的通知.docx |
| metering_data_traceability | 计量数据作为市场结算依据且异常处理可追踪 | 2025-10-15 00:00:00 | 数据质量约束 | 数据预处理 + 15分钟结算 | data_quality_report.md, actual_need_15m, imbalance_energy_15m | 政策环境/2025.10.15湖南省电力市场计量实施规则.docx |

## 三、详细解释

| constraint_name | market_rule | implementation |
| --- | --- | --- |
| 中长期合约仅作为结算依据 | 湖南现货规则采用“中长期合约仅作为结算依据管理市场风险、现货交易全电量集中竞价”的模式；中长期交易结果不作为调度执行依据。 | 周度动作只形成结算底仓 q_lt；现货采购与偏差结算在 settle_week 中按 15 分钟轨迹计算，不把中长期合约作为物理调度计划。 |
| 中长期交易按每日24时段组织 | 中长期交易按每天24小时划分为24个时段，年度、月度、月内交易均需分时段申报和分解。 | 程序用小时级 q_lt_hourly 承接24时段合约曲线，再映射到15分钟代理结算；不得退化为月度单点合约。 |
| 日前/实时现货按96个15分钟时段结算 | 日前市场按日组织，每个运行日包含96个15分钟交易出清时段；实时市场决定未来15分钟调度资源分配状态和计划。 | 小时级现货修正只作为内部决策近似；正式成本、波动率、CVaR 和套保误差均基于15分钟结算轨迹汇总。 |
| 零售用户合同周期内单一售电公司与单一套餐 | 同一个合同周期内，零售用户只能向一家售电公司购电、签订零售合同；同一用户在同一结算周期内仅可生效一个零售套餐。 | 模型不得把零售套餐价格作为自由动作；售电公司策略仅优化批发采购和风险对冲侧。 |
| 现货试运行期间省内调峰辅助服务暂停 | 电力现货市场试运行期间，调峰辅助服务市场暂停运行，调峰辅助服务不予补偿；调频辅助服务与现货市场存在容量预留耦合。 | 辅助服务不进入主策略收益函数；只通过制度状态压缩或标注现货边际可调空间。 |
| 新能源机制电价自2026-01-01进入执行期 | 2025年投产且纳入机制的项目，机制电价从2026年1月1日起执行；2026年投产且纳入机制的项目从申报投产日期次月1日起执行。 | 2025-11 至 2025-12 只允许出现竞价准备状态；2026-01-01 起才可进入执行状态并影响新能源扰动代理。 |
| 2026年度中长期价格40%固定 + 60%实时联动自2026-02执行 | 年度中长期合同电量的40%可协商固定价格，60%执行反映实时供需的灵活价格；该价格机制自2026年2月份年度多月交易开始执行。 | 2026-02 前使用上一自然周日前均价代理；2026-02 起 resolve_settlement_context 才按40%日前固定价和60%日内联动价代理中长期价格。 |
| 计量数据作为市场结算依据且异常处理可追踪 | 市场结算用计量数据原则上由用电信息采集系统自动采集；计量异常需要按规则拟合补全。 | 程序保留数据质量报告、缺失时点统计和15分钟偏差结算轨迹，缺失和异常处理不得静默发生。 |

## 四、模型校验

- 未发现制度时点或核心映射校验问题。

## 五、实现落点

- 政策解析: `src/policy/policy_parser.py` 生成结构化规则表。
- 制度状态: `src/policy/policy_regime.py` 生成周度当前状态和四组前瞻状态。
- 价格口径: `src/backtest/settlement.py` 根据 `lt_price_linked_active` 切换中长期价格代理。
- 下层边际空间: `src/training/score_kernel.py` 与 `src/backtest/materialize.py` 输出参数化电量决策、政策投影结果和 15 分钟结算指标。
- 输出位置: `outputs/<version>/reports/market_rule_constraints.md`。
