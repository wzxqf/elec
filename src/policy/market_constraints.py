from __future__ import annotations

from typing import Any

import pandas as pd


CORE_MARKET_CONSTRAINTS: list[dict[str, Any]] = [
    {
        "constraint_id": "lt_contract_settlement_only",
        "constraint_name": "中长期合约仅作为结算依据",
        "market_rule": "湖南现货规则采用“中长期合约仅作为结算依据管理市场风险、现货交易全电量集中竞价”的模式；中长期交易结果不作为调度执行依据。",
        "publish_time": "2025-10-15",
        "effective_start": "2025-10-15",
        "effective_end": "",
        "source_keyword": "现货市场交易实施细则",
        "constraint_type": "结算口径硬约束",
        "model_layer": "上层周度中长期底仓 + 15分钟结算",
        "model_mapping": "lt_settlement_base, spot_marginal_exposure, lt_spot_coupling_state, q_lt_hourly, lt_energy_15m",
        "implementation": "周度动作只形成结算底仓 q_lt；现货采购与偏差结算在 settle_week 中按 15 分钟轨迹计算，不把中长期合约作为物理调度计划。",
    },
    {
        "constraint_id": "lt_24_period_contract_curve",
        "constraint_name": "中长期交易按每日24时段组织",
        "market_rule": "中长期交易按每天24小时划分为24个时段，年度、月度、月内交易均需分时段申报和分解。",
        "publish_time": "2025-10-15",
        "effective_start": "2025-10-15",
        "effective_end": "",
        "source_keyword": "电力中长期实施细则",
        "constraint_type": "时间粒度硬约束",
        "model_layer": "周度底仓向小时曲线分解",
        "model_mapping": "_allocate_weekly_lt_to_hourly, q_lt_hourly",
        "implementation": "程序用小时级 q_lt_hourly 承接24时段合约曲线，再映射到15分钟代理结算；不得退化为月度单点合约。",
    },
    {
        "constraint_id": "spot_15min_96_settlement",
        "constraint_name": "日前/实时现货按96个15分钟时段结算",
        "market_rule": "日前市场按日组织，每个运行日包含96个15分钟交易出清时段；实时市场决定未来15分钟调度资源分配状态和计划。",
        "publish_time": "2025-10-15",
        "effective_start": "2025-10-15",
        "effective_end": "",
        "source_keyword": "现货市场交易实施细则",
        "constraint_type": "结算时间粒度硬约束",
        "model_layer": "15分钟代理结算回测",
        "model_mapping": "settlement_interval_hours=0.25, settle_week, procurement_cost_15m, imbalance_energy_15m",
        "implementation": "小时级现货修正只作为内部决策近似；正式成本、波动率、CVaR 和套保误差均基于15分钟结算轨迹汇总。",
    },
    {
        "constraint_id": "retail_single_supplier_package",
        "constraint_name": "零售用户合同周期内单一售电公司与单一套餐",
        "market_rule": "同一个合同周期内，零售用户只能向一家售电公司购电、签订零售合同；同一用户在同一结算周期内仅可生效一个零售套餐。",
        "publish_time": "2025-10-15",
        "effective_start": "2025-10-15",
        "effective_end": "",
        "source_keyword": "零售市场交易规则",
        "constraint_type": "收益端边界约束",
        "model_layer": "售电公司收益端与基准策略解释",
        "model_mapping": "retail package is treated as exogenous revenue-side boundary",
        "implementation": "模型不得把零售套餐价格作为自由动作；售电公司策略仅优化批发采购和风险对冲侧。",
    },
    {
        "constraint_id": "ancillary_peak_shaving_pause_during_spot_trial",
        "constraint_name": "现货试运行期间省内调峰辅助服务暂停",
        "market_rule": "电力现货市场试运行期间，调峰辅助服务市场暂停运行，调峰辅助服务不予补偿；调频辅助服务与现货市场存在容量预留耦合。",
        "publish_time": "2025-10-14",
        "effective_start": "2025-10-14",
        "effective_end": "",
        "source_keyword": "辅助服务市场建设",
        "constraint_type": "市场边界约束",
        "model_layer": "下层小时级现货滚动修正",
        "model_mapping": "ancillary_peak_shaving_pause, ancillary_freq_reserve_tight, bandwidth_multiplier",
        "implementation": "辅助服务不进入主策略收益函数；只通过制度状态压缩或标注现货边际可调空间。",
    },
    {
        "constraint_id": "renewable_mechanism_from_2026_01_01",
        "constraint_name": "新能源机制电价自2026-01-01进入执行期",
        "market_rule": "2025年投产且纳入机制的项目，机制电价从2026年1月1日起执行；2026年投产且纳入机制的项目从申报投产日期次月1日起执行。",
        "publish_time": "2025-11-05",
        "effective_start": "2026-01-01",
        "effective_end": "",
        "source_keyword": "新能源机制电价竞价工作有关事项",
        "constraint_type": "制度时点硬约束",
        "model_layer": "状态空间 + 新能源扰动代理",
        "model_mapping": "renewable_mechanism_active, mechanism_stage_label, mechanism_price_floor, mechanism_price_ceiling, mechanism_volume_ratio_max",
        "implementation": "2025-11 至 2025-12 只允许出现竞价准备状态；2026-01-01 起才可进入执行状态并影响新能源扰动代理。",
    },
    {
        "constraint_id": "lt_price_linkage_from_2026_02_01",
        "constraint_name": "2026年度中长期价格40%固定 + 60%实时联动自2026-02执行",
        "market_rule": "年度中长期合同电量的40%可协商固定价格，60%执行反映实时供需的灵活价格；该价格机制自2026年2月份年度多月交易开始执行。",
        "publish_time": "2025-12-31",
        "effective_start": "2026-02-01",
        "effective_end": "",
        "source_keyword": "完善2026年度电力中长期交易价格机制",
        "constraint_type": "制度时点硬约束",
        "model_layer": "中长期价格代理与15分钟结算",
        "model_mapping": "lt_price_linked_active, fixed_price_ratio_max, linked_price_ratio_min, resolve_settlement_context",
        "implementation": "2026-02 前使用上一自然周日前均价代理；2026-02 起 resolve_settlement_context 才按40%日前固定价和60%日内联动价代理中长期价格。",
    },
    {
        "constraint_id": "metering_data_traceability",
        "constraint_name": "计量数据作为市场结算依据且异常处理可追踪",
        "market_rule": "市场结算用计量数据原则上由用电信息采集系统自动采集；计量异常需要按规则拟合补全。",
        "publish_time": "2025-10-15",
        "effective_start": "2025-10-15",
        "effective_end": "",
        "source_keyword": "计量实施规则",
        "constraint_type": "数据质量约束",
        "model_layer": "数据预处理 + 15分钟结算",
        "model_mapping": "data_quality_report.md, actual_need_15m, imbalance_energy_15m",
        "implementation": "程序保留数据质量报告、缺失时点统计和15分钟偏差结算轨迹，缺失和异常处理不得静默发生。",
    },
]


def _timestamp_or_empty(value: Any) -> pd.Timestamp | str:
    if value in (None, ""):
        return ""
    return pd.Timestamp(value)


def build_market_rule_constraints(config: dict[str, Any], rule_table: pd.DataFrame) -> pd.DataFrame:
    sample_start = pd.Timestamp(config["sample_start"])
    sample_end = pd.Timestamp(config["sample_end"])
    source_files = rule_table["source_file"].astype(str).tolist() if "source_file" in rule_table.columns else []
    rows: list[dict[str, Any]] = []

    for item in CORE_MARKET_CONSTRAINTS:
        row = dict(item)
        effective_start = pd.Timestamp(row["effective_start"])
        effective_end = _timestamp_or_empty(row.get("effective_end"))
        if isinstance(effective_end, pd.Timestamp):
            active_in_project = effective_end >= sample_start and effective_start <= sample_end
        else:
            active_in_project = effective_start <= sample_end
        matched_sources = sorted({path for path in source_files if str(row["source_keyword"]) in path})
        row["effective_start"] = effective_start
        row["effective_end"] = effective_end
        row["active_in_project"] = bool(active_in_project)
        row["policy_source_files"] = "; ".join(matched_sources)
        row["source_file_count"] = len(matched_sources)
        rows.append(row)

    return pd.DataFrame(rows)


def _require_rule(
    rule_table: pd.DataFrame,
    *,
    rule_type: str,
    state_name: str,
    effective_start: str,
    expected_value: Any | None = None,
) -> str | None:
    if rule_table.empty:
        return f"缺少规则表，无法校验 {state_name}。"
    mask = (rule_table["rule_type"] == rule_type) & (rule_table["state_name"] == state_name)
    mask &= pd.to_datetime(rule_table["effective_start"], errors="coerce") == pd.Timestamp(effective_start)
    subset = rule_table.loc[mask]
    if subset.empty:
        return f"缺少 {state_name} 在 {effective_start} 生效的结构化规则。"
    if expected_value is not None and str(expected_value) not in set(subset["state_value"].astype(str)):
        return f"{state_name} 在 {effective_start} 的取值不包含 {expected_value}。"
    return None


def validate_market_rule_alignment(
    config: dict[str, Any],
    rule_table: pd.DataFrame,
    policy_trace: pd.DataFrame,
) -> list[str]:
    violations: list[str] = []
    for error in [
        _require_rule(
            rule_table,
            rule_type="renewable_mechanism_execution",
            state_name="renewable_mechanism_active",
            effective_start="2026-01-01",
            expected_value=1.0,
        ),
        _require_rule(
            rule_table,
            rule_type="lt_price_linkage",
            state_name="lt_price_linked_active",
            effective_start="2026-02-01",
            expected_value=1.0,
        ),
        _require_rule(
            rule_table,
            rule_type="lt_price_linkage",
            state_name="fixed_price_ratio_max",
            effective_start="2026-02-01",
            expected_value=0.4,
        ),
        _require_rule(
            rule_table,
            rule_type="lt_price_linkage",
            state_name="linked_price_ratio_min",
            effective_start="2026-02-01",
            expected_value=0.6,
        ),
    ]:
        if error is not None:
            violations.append(error)

    if not policy_trace.empty:
        trace = policy_trace.copy()
        trace["week_start"] = pd.to_datetime(trace["week_start"])
        before_renewable = trace.loc[trace["week_start"] < pd.Timestamp("2026-01-01"), "renewable_mechanism_active"]
        if not before_renewable.empty and (pd.to_numeric(before_renewable, errors="coerce").fillna(0.0) > 0.0).any():
            violations.append("renewable_mechanism_active 被提前应用到 2026-01-01 之前。")
        before_linkage = trace.loc[trace["week_start"] < pd.Timestamp("2026-02-01"), "lt_price_linked_active"]
        if not before_linkage.empty and (pd.to_numeric(before_linkage, errors="coerce").fillna(0.0) > 0.0).any():
            violations.append("lt_price_linked_active 被提前应用到 2026-02-01 之前。")

    if float(config["settlement_interval_hours"]) != 0.25:
        violations.append("settlement_interval_hours 必须为 0.25，以匹配15分钟结算。")
    if str(config["week_period"]) != "W-SUN":
        violations.append("week_period 应保持 W-SUN，保证周度决策与项目设定一致。")
    return violations


def _frame_to_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_无数据_"
    printable = frame.loc[:, columns].copy().fillna("")
    headers = [str(column) for column in printable.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in printable.astype(object).values.tolist():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def build_market_rule_constraints_markdown(
    *,
    config: dict[str, Any],
    constraints: pd.DataFrame,
    rule_table: pd.DataFrame,
    violations: list[str],
) -> str:
    active = constraints.loc[constraints["active_in_project"]].copy()
    columns = [
        "constraint_id",
        "constraint_name",
        "effective_start",
        "constraint_type",
        "model_layer",
        "model_mapping",
        "policy_source_files",
    ]
    detail_columns = [
        "constraint_name",
        "market_rule",
        "implementation",
    ]
    violation_text = "\n".join(f"- {item}" for item in violations) if violations else "- 未发现制度时点或核心映射校验问题。"
    return "\n".join(
        [
            "# 市场规则约束与模型映射",
            "",
            f"- 项目周期: {config['sample_start']} 至 {config['sample_end']}",
            f"- 政策来源文件数: {len(set(rule_table['source_file'])) if not rule_table.empty and 'source_file' in rule_table.columns else 0}",
            f"- 结构化规则数: {len(rule_table)}",
            "- 口径说明: 以下约束只覆盖本项目主线市场范围，即中长期市场、日前现货、日内/实时现货和15分钟代理结算；不把辅助服务收益、需求响应收益或虚拟投标并入主策略收益。",
            "",
            "## 一、核心结论",
            "",
            "- 中长期合约仅作为结算依据，不作为物理调度计划。",
            "- 周度上层动作只能决定中长期底仓残差和诊断性现货边际敞口带宽。",
            "- 小时级现货修正是内部近似，正式回测按96点15分钟轨迹结算。",
            "- 新能源机制电价状态只能自 2026-01-01 起进入执行期。",
            "- 年度中长期价格 40% 固定 + 60% 实时联动机制只能自 2026-02-01 起进入结算代理。",
            "",
            "## 二、约束清单",
            "",
            _frame_to_markdown(active, columns),
            "",
            "## 三、详细解释",
            "",
            _frame_to_markdown(active, detail_columns),
            "",
            "## 四、模型校验",
            "",
            violation_text,
            "",
            "## 五、实现落点",
            "",
            "- 政策解析: `src/policy/policy_parser.py` 生成结构化规则表。",
            "- 制度状态: `src/policy/policy_regime.py` 生成周度当前状态和四组前瞻状态。",
            "- 价格口径: `src/backtest/settlement.py` 根据 `lt_price_linked_active` 切换中长期价格代理。",
            "- 下层边际空间: `src/rules/hourly_hedge.py` 和 `src/agents/hpso.py` 保留带宽、平滑、非负等诊断字段；HPSO v0.3 不再执行硬裁剪。",
            "- 输出位置: `outputs/<version>/reports/market_rule_constraints.md`。",
            "",
        ]
    )
