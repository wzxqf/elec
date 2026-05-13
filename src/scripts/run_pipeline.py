from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import time

import pandas as pd

from src.analysis.report_contracts import infer_date_range
from src.analysis.reporting import build_hourly_spot_activation_summary
from src.scripts.backtest import run_backtest
from src.scripts.common import prepare_project_context
from src.scripts.evaluate import run_evaluate
from src.scripts.train import run_train
from src.utils.experiment_manifest import (
    build_artifact_index_markdown,
    filter_existing_key_outputs,
    build_release_manifest,
    build_run_manifest,
    fallback_run_metadata,
    prepend_report_header,
    relativize_path,
)
from src.utils.io import save_json, save_markdown
from src.utils.runtime_status import RuntimeStatusTracker, build_training_phase_name


def _persist_manifest_updates(context: dict, extra_outputs: dict[str, str]) -> None:
    if "seed" not in context["config"] or "split" not in context["config"]:
        return
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    run_metadata = context.get("run_metadata", fallback_run_metadata(context["config"]))
    key_outputs = filter_existing_key_outputs(output_root, extra_outputs)
    save_json(build_release_manifest(run_metadata, context["config"], key_outputs), output_root / "release_manifest.json")
    save_json(build_run_manifest(run_metadata, context["config"], key_outputs), output_root / "run_manifest.json")
    save_markdown(build_artifact_index_markdown(run_metadata, key_outputs), output_root / "artifact_index.md")


def _format_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.{digits}f}"


def _records_as_bullets(frame: pd.DataFrame, columns: list[str], limit: int = 5) -> list[str]:
    if frame.empty:
        return ["- 无可用记录。"]
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return ["- 表已生成，但缺少摘要列。"]
    lines: list[str] = []
    for row in frame.head(limit).to_dict(orient="records"):
        parts = [f"{column}={row.get(column)}" for column in available_columns]
        lines.append("- " + "; ".join(parts))
    return lines


def _format_table_value(value: object) -> str:
    if isinstance(value, (list, tuple, dict)):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _frame_as_markdown(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> list[str]:
    if frame.empty:
        return ["无可用记录。", ""]
    selected = frame.copy()
    if columns is not None:
        available = [column for column in columns if column in selected.columns]
        if not available:
            return ["表已生成，但缺少可展示列。", ""]
        selected = selected[available]
    omitted = 0
    if limit is not None and len(selected) > limit:
        omitted = len(selected) - limit
        selected = selected.head(limit)
    header = "| " + " | ".join(str(column) for column in selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join(_format_table_value(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    if omitted:
        rows.append(f"| ... | 省略 {omitted} 行，完整数据见同名 raw CSV。 |")
    return [header, separator, *rows, ""]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _frame_from_result_or_csv(result: dict, key: str, path: Path) -> pd.DataFrame:
    frame = result.get(key, pd.DataFrame())
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return frame
    return _read_csv_if_exists(path)


def _file_hash_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _raw_artifact_inventory(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    raw_root = output_root / "raw"
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        row: dict[str, object] = {
            "artifact": relativize_path(path, output_root),
            "category": path.parent.relative_to(raw_root).parts[0] if path.parent != raw_root else "raw",
            "bytes": path.stat().st_size,
            "sha256_16": _file_hash_prefix(path),
        }
        if path.suffix.lower() == ".csv":
            frame = _read_csv_if_exists(path)
            row["rows"] = len(frame)
            row["columns"] = len(frame.columns)
        else:
            row["rows"] = ""
            row["columns"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _dataframe_records(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> list[dict[str, object]]:
    if frame.empty:
        return []
    selected = frame.copy()
    if columns is not None:
        selected = selected[[column for column in columns if column in selected.columns]]
    if limit is not None:
        selected = selected.head(limit)
    records = selected.where(pd.notna(selected), None).to_dict(orient="records")
    return [{key: _json_safe_value(value) for key, value in row.items()} for row in records]


def _json_safe_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_reports_dir(reports_dir: Path, keep_names: set[str]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for child in reports_dir.iterdir():
        if child.name in keep_names:
            continue
        _remove_report_path(child)


def _remove_report_path(path: Path) -> None:
    def _retry_remove(function, failing_path, _exc_info):
        try:
            Path(failing_path).chmod(stat.S_IWRITE)
        except OSError:
            pass
        function(failing_path)

    for attempt in range(5):
        try:
            if path.is_dir():
                shutil.rmtree(path, onerror=_retry_remove)
            else:
                path.chmod(stat.S_IWRITE)
                path.unlink()
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def _build_human_report(context: dict, training: dict, validation: dict, backtest: dict) -> str:
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    config = context["config"]
    version = config.get("version", "unknown")
    run_metadata = context.get("run_metadata", {})
    split_cfg = config.get("split", {})
    training_runtime = training.get("runtime_profile", {})
    rolling_summary = backtest.get("rolling_summary")
    rolling_aggregate = getattr(rolling_summary, "aggregate", {}) if rolling_summary is not None else {}
    rolling_excess = backtest.get("rolling_excess_return_metrics", pd.DataFrame())
    benchmark_metrics = backtest.get("benchmark_metrics", pd.DataFrame())
    ablation_metrics = backtest.get("ablation_metrics", pd.DataFrame())
    robustness_metrics = backtest.get("robustness_metrics", pd.DataFrame())
    rule_table = context.get("bundle", {}).get("policy_rule_table", pd.DataFrame())
    policy_inventory = context.get("bundle", {}).get("policy_inventory", pd.DataFrame())
    policy_failures = context.get("bundle", {}).get("policy_failures", pd.DataFrame())
    persistent_windows = int(rolling_excess.get("active_excess_return_persistent", pd.Series(dtype="bool")).astype(bool).sum())
    rolling_window_range = infer_date_range(backtest.get("rolling_weekly_results", pd.DataFrame()))
    validation_metrics = validation.get("metrics", {})
    metrics_dir = context["output_paths"]["metrics"]
    validation_weekly_results = _frame_from_result_or_csv(validation, "weekly_results", metrics_dir / "validation_weekly_results.csv")
    contract_value_weekly = _frame_from_result_or_csv(validation, "contract_value_weekly", metrics_dir / "contract_value_weekly.csv")
    policy_risk_metrics = _frame_from_result_or_csv(validation, "policy_risk_metrics", metrics_dir / "policy_risk_adjusted_metrics.csv")
    rolling_schedule = _read_csv_if_exists(metrics_dir / "rolling_window_schedule.csv")
    rolling_validation_metrics = _frame_from_result_or_csv(backtest, "validation_metrics", metrics_dir / "rolling_validation_metrics.csv")
    rolling_backtest_metrics = _frame_from_result_or_csv(backtest, "backtest_metrics", metrics_dir / "rolling_backtest_metrics.csv")
    rolling_weekly_results = _frame_from_result_or_csv(backtest, "rolling_weekly_results", metrics_dir / "rolling_weekly_results.csv")
    raw_inventory = _raw_artifact_inventory(output_root)
    reward_cfg = config.get("reward", {})
    score_cfg = config.get("score_kernel", {})
    economics_cfg = config.get("economics", {})
    hpso_cfg = config.get("hybrid_pso", {})
    feasible_cfg = config.get("policy_feasible_domain", {})
    lines = [
        f"# {version} 人工阅读报告",
        "",
        "## 报告用途",
        "",
        "- 目标读者: 论文作者、项目审计者和需要快速复核实现结论的人。",
        "- 生成时点: 每次执行 `src.scripts.run_pipeline` 全流程后生成。",
        "- 口径: 整合项目实现逻辑、完整数学模型、全部数学结果摘要和 raw 产物索引；分模块摘要、CSV、模型、日志统一放入 `raw/`。",
        f"- 配套 AI 结构化报告: `reports/{version}_ai_structured_report.json`。",
        "",
        "## 项目定位",
        "",
        "- 研究主体: 湖南电力市场售电公司。",
        "- 决策框架: 周度中长期底仓决策、小时级现货边际风险修正、15 分钟代理结算回测。",
        "- 核心基准: `dynamic_lock_only` 强基准保留为重点比较对象，同时报告固定持仓、静态无现货修正、简单滚动对冲等基准。",
        "- 价格口径: 2026-02 后中长期有效价使用 `lt_price_w_effective`，体现年度多月交易 40% 固定价格与 60% 灵活价格的生效时点。",
        "",
        "## 完整实现逻辑",
        "",
        "### 实现链路",
        "",
        "- `experiment_config.yaml` 是唯一人工配置入口，版本号决定 `outputs/<version>/` 正式产物目录。",
        "- `prepare_project_context()` 读取原始 CSV，完成 15 分钟清洗、小时聚合、周度特征、政策文件解析、制度状态轨迹、可行域编译和训练张量打包。",
        "- `compile_parameter_layout()` 根据真实特征列生成上层和下层参数块；训练和评分均以 compiled layout 的真实维度为准。",
        "- `run_train()` 调用 `HYBRID_PSO_V040`，在训练周上用上层粒子和下层粒子的组合搜索最小目标函数。",
        "- `run_evaluate()` 在正式验证集物化最优粒子，输出周度、小时级、15 分钟结算和政策风险调整结果。",
        "- `run_backtest()` 按 expanding rolling plan 每个窗口重新训练并测试，同时输出强基准比较、消融和稳健性情景重算。",
        "- `materialize_particle_pair()` 是训练评分和报告明细共同使用的物化口径，保证报告结果与评分核一致。",
        "- `release_manifest.json`、`run_manifest.json`、`artifact_index.md` 作为索引入口，正式阅读入口只保留本报告和 AI 结构化报告。",
        "",
        "### raw 产物分类",
        "",
        "- `raw/metrics/`: CSV 数学结果、政策表、特征表、滚动窗口明细、15 分钟结算明细。",
        "- `raw/metadata/`: 配置快照、参数布局和训练运行元数据。",
        "- `raw/models/`: 训练得到的 HPSO 模型参数。",
        "- `raw/logs/`: pipeline、train、evaluate、backtest 和运行器日志。",
        "- `reports/`: 只保留一份面向人的报告和一份面向 AI 的结构化报告。",
        "",
        "## 数学模型与公式",
        "",
        "本节给出当前项目的完整数学模型，覆盖周度底仓、小时级带符号净修正量、15 分钟代理结算、风险指标、约束投影和 HPSO 目标函数。",
        "",
        "## 完整数学模型",
        "",
        "### 符号和状态",
        "",
        "- 周索引 `w`，小时索引 `h`，15 分钟结算点索引 `t`。",
        "- 周度状态 `s_w = [x_w, p_w, f_w, a_{w-1}, r_{w-1}]`，其中 `x_w` 为市场特征，`p_w` 为当前制度状态，`f_w` 为四组前瞻制度状态。",
        "- 上层动作 `a_w = (delta_h_raw_w, b_raw_w)`，分别表示相对强基准底仓的原始残差和现货边际敞口带宽。",
        "- 下层动作 `u_{w,h}` 是带符号小时级现货滚动修正量。",
        "",
        "### 周度中长期底仓模型",
        "",
        "- 周度信号: `g_w = x_w @ theta_feature + p_w @ theta_policy + theta_bias`。",
        "- 原始底仓残差: `delta_h_raw_w = rho_delta * tanh(k_x * x_w @ theta_feature + k_p * p_w @ theta_policy + theta_contract_bias) * Lhat_w`。",
        "- 约束投影: `delta_h_exec_w = clip(delta_h_raw_w, r_min_w * Lhat_w, r_max_w * Lhat_w)`。",
        "- 最终底仓: `H_w = max(r_base * Lhat_w + delta_h_exec_w, 0)`。",
        "- 原始敞口带宽: `B_raw_w = relu(r_band * Lhat_w * (1 + tanh(k_b * g_w + theta_exposure_bias)))`。",
        "- 执行敞口带宽: `B_w = clip(B_raw_w, b_min_w * Lhat_w, b_max_w * Lhat_w)`。",
        "- 24 小时合约曲线: `pi_{w,1:24} = softmax(theta_curve @ Basis_24h)`。",
        "",
        "### 小时级现货滚动修正模型",
        "",
        "- 小时信号: `m_{w,h} = beta_spread * spread_{w,h-1} + beta_load * load_dev_{w,h-1} + beta_ren * renewable_dev_{w,h-1} + beta_abs * |spread| * session_h + beta_ren_abs * |renewable_dev| * session_h`。",
        "- 门控: `gate_{w,h} = sigmoid((|m_{w,h}| - deadband) / temperature)`，低于 deadband 的信号置零。",
        "- 原始修正: `u_raw_{w,h} = gate_{w,h} * tanh(m_{w,h}) * B_w / N_h * (limit_base + limit_shrink * shrink_policy)`。",
        "- 约束投影: `u_{w,h} = Proj_hour(u_raw_{w,h}; B_w, share_cap_{w,h}, ramp_cap_{w,h})`。",
        "- 单点上限: `|u_{w,h}| <= B_w / N_h * share_cap_{w,h}`。",
        "- 平滑上限: `|u_{w,h} - u_{w,h-1}| <= B_w / N_h * share_cap_{w,h} * ramp_cap_{w,h}`。",
        "",
        "### 15 分钟代理结算模型",
        "",
        "- 合约小时量: `q_contract_{w,h} = H_w * pi_{w,h}`，再均分到 15 分钟点 `q_contract_{w,t}`。",
        "- 现货小时修正 `u_{w,h}` 均分到 `q_spot_net_{w,t}`，也就是带符号净修正量。",
        "- 计划电量: `q_sched_{w,t} = max(q_contract_{w,t} + q_spot_net_{w,t}, 0)`。",
        "- 实际需求: `q_real_{w,t} = max(net_load_id_mwh_{w,t}, 0)`。",
        "- 结算成本: `C_{w,t} = q_sched_{w,t} * (omega_lt * p_lt_eff_w + omega_da * p_da_{w,t}) + |q_real_{w,t} - q_sched_{w,t}| * p_id_{w,t} * eta_imbalance`。",
        "- 周采购成本: `C_w = sum_t C_{w,t}`。",
        "- 零售收入: `R_w = sum_t q_real_{w,t} * tariff_retail`。",
        "- 利润: `P_w = R_w - C_w - |delta_h_raw_w - delta_h_exec_w| * c_adjust - sum_h |u_{w,h}| * c_friction`。",
        "",
        "### 风险、奖励和优化目标",
        "",
        "- `CVaR_alpha(w)` 为 15 分钟成本 `C_{w,t}` 的最坏 `(1 - alpha)` 尾部均值。",
        "- 套保误差: `E_w = |sum_t q_real_{w,t} - sum_t q_sched_{w,t}| / max(sum_t q_real_{w,t}, 1)`。",
        "- 强基准利润: `P_base_w = R_w - min_b C_base(w, b)`，`b` 来自 `reward.baseline_position_ratios`。",
        "- 超额收益: `X_w = P_w - P_base_w`。",
        "- 违规惩罚: `V_w = (|delta_h_raw_w - delta_h_exec_w| + |B_raw_w - B_w| + sum_h |u_raw_{w,h} - u_{w,h}|) * lambda_projection`。",
        "- 周奖励: `r_w = X_w - lambda_tail * CVaR_alpha(w) - lambda_hedge * E_w - lambda_trade * F_w - lambda_violate * V_w`。",
        "- 粒子群目标函数: `J(theta_u, theta_l) = - sum_w r_w`。",
        "- PSO 更新: `v <- inertia*v + cognitive*rand()*(pbest - position) + social*rand()*(gbest - position)`，`position <- clip(position + v, -position_clip_abs, position_clip_abs)`。",
        "",
        "## 运行配置",
        "",
        f"- 版本: {config.get('version', 'unknown')}",
        f"- 实验编号: {run_metadata.get('experiment_id', 'unknown')}",
        f"- 配置哈希: {run_metadata.get('config_hash', 'unknown')}",
        f"- 运行时间: {run_metadata.get('run_timestamp', 'unknown')}",
        f"- 样本范围: {config.get('sample_start', '')} -> {config.get('sample_end', '')}",
        f"- 训练集: {split_cfg.get('train_start_week', 'n/a')} -> {split_cfg.get('train_end_week', 'n/a')}",
        f"- 正式验证集: {split_cfg.get('val_start_week', 'n/a')} -> {split_cfg.get('val_end_week', 'n/a')}",
        f"- 正式测试集: {split_cfg.get('test_start_week', 'n/a')} -> {split_cfg.get('test_end_week', 'n/a')}",
        f"- 滚动窗口覆盖: {rolling_window_range}",
        f"- 算法: {config.get('training', {}).get('algorithm', 'unknown')}",
        f"- 设备: {training.get('device', config.get('training', {}).get('device', 'unknown'))}",
        f"- 上层配置维度 / 真实维度: {config.get('hybrid_pso', {}).get('upper', {}).get('dimension', training_runtime.get('upper_dim', 'n/a'))} / {training_runtime.get('upper_dim_real', training_runtime.get('upper_dim', 'n/a'))}",
        f"- 下层配置维度 / 真实维度: {config.get('hybrid_pso', {}).get('lower', {}).get('dimension', training_runtime.get('lower_dim', 'n/a'))} / {training_runtime.get('lower_dim_real', training_runtime.get('lower_dim', 'n/a'))}",
        f"- 政策来源文件数: {len(policy_inventory)}",
        f"- 结构化规则数: {len(rule_table)}",
        f"- 政策解析失败文件数: {len(policy_failures)}",
        "",
    ]
    lines.extend(
        _frame_as_markdown(
            pd.DataFrame(
                [
                    {"parameter": "reward.baseline_strategy", "value": reward_cfg.get("baseline_strategy", "dynamic_lock_only")},
                    {"parameter": "reward.baseline_position_ratios", "value": reward_cfg.get("baseline_position_ratios", [])},
                    {"parameter": "reward.cvar_alpha", "value": reward_cfg.get("cvar_alpha", "n/a")},
                    {"parameter": "reward.lambda_tail", "value": reward_cfg.get("lambda_tail", "n/a")},
                    {"parameter": "reward.lambda_hedge", "value": reward_cfg.get("lambda_hedge", "n/a")},
                    {"parameter": "reward.lambda_trade", "value": reward_cfg.get("lambda_trade", "n/a")},
                    {"parameter": "reward.lambda_violate", "value": reward_cfg.get("lambda_violate", "n/a")},
                    {"parameter": "score_kernel.lt_settlement_weight", "value": score_cfg.get("lt_settlement_weight", "n/a")},
                    {"parameter": "score_kernel.da_settlement_weight", "value": score_cfg.get("da_settlement_weight", "n/a")},
                    {"parameter": "economics.retail_tariff_yuan_per_mwh", "value": economics_cfg.get("retail_tariff_yuan_per_mwh", "n/a")},
                    {"parameter": "hybrid_pso.upper.particles", "value": hpso_cfg.get("upper", {}).get("particles", "n/a")},
                    {"parameter": "hybrid_pso.lower.particles", "value": hpso_cfg.get("lower", {}).get("particles", "n/a")},
                    {"parameter": "hybrid_pso.upper.iterations", "value": hpso_cfg.get("upper", {}).get("iterations", "n/a")},
                    {"parameter": "hybrid_pso.optimizer.inertia", "value": hpso_cfg.get("optimizer", {}).get("inertia", "n/a")},
                    {"parameter": "policy_feasible_domain.contract_adjustment_ratio_limit_linked", "value": feasible_cfg.get("contract_adjustment_ratio_limit_linked", "n/a")},
                    {"parameter": "policy_feasible_domain.exposure_band_ratio_cap_ancillary_tight", "value": feasible_cfg.get("exposure_band_ratio_cap_ancillary_tight", "n/a")},
                ]
            )
        )
    )
    lines.extend(
        [
            "## 模型运行参数设置",
            "",
            "上方运行配置表和关键参数表构成本版本模型运行参数设置，完整配置快照见 `raw/metadata/train_config_snapshot.yaml`。",
            "",
            "## 模型运行效果",
            "",
            "下方给出正式验证、滚动回测、基准、消融和稳健性的全部数学结果摘要。",
            "",
            "## 全部数学结果",
            "",
            "### 运行结果摘要",
            "",
            f"- 验证集采购成本: {_format_number(validation_metrics.get('total_procurement_cost', 0.0))}",
            f"- 验证集累计收益: {_format_number(validation_metrics.get('total_profit', 0.0))}",
            f"- 验证集成本波动: {_format_number(validation_metrics.get('weekly_cost_volatility', 0.0))}",
            f"- 验证集 CVaR99: {_format_number(validation_metrics.get('cvar99', 0.0))}",
            f"- 验证集 hedge_error: {_format_number(validation_metrics.get('hedge_error', 0.0), digits=4)}",
            f"- 滚动窗口数: {_format_number(rolling_aggregate.get('window_count', 0.0), digits=0)}",
            f"- 滚动平均采购成本: {_format_number(rolling_aggregate.get('mean_total_procurement_cost', 0.0))}",
            f"- 滚动平均收益: {_format_number(rolling_aggregate.get('mean_total_profit', 0.0))}",
            f"- 滚动平均 CVaR99: {_format_number(rolling_aggregate.get('mean_cvar99', 0.0))}",
            f"- 跑赢 dynamic_lock_only 的滚动窗口数: {persistent_windows}",
            "",
            "### 训练运行结果",
            "",
        ]
    )
    lines.extend(_frame_as_markdown(pd.DataFrame([{"metric": key, "value": value} for key, value in training_runtime.items()])))
    lines.extend(["### 正式验证总体指标", ""])
    lines.extend(_frame_as_markdown(pd.DataFrame([{"metric": key, "value": value} for key, value in validation_metrics.items()])))
    lines.extend(["### 正式验证周度结果", ""])
    lines.extend(
        _frame_as_markdown(
            validation_weekly_results,
            [
                "week_start",
                "lock_ratio_base",
                "delta_lock_ratio",
                "lock_ratio_final",
                "contract_adjustment_mwh_raw",
                "contract_adjustment_mwh_exec",
                "contract_position_mwh",
                "exposure_band_mwh",
                "procurement_cost_w",
                "profit_w",
                "profit_baseline_w",
                "excess_profit_w",
                "cvar99_w",
                "hedge_error_w",
                "reward_w",
                "bound_reason_code",
            ],
        )
    )
    lines.extend(["### 合同价值周度结果", ""])
    lines.extend(
        _frame_as_markdown(
            contract_value_weekly,
            [
                "week_start",
                "expected_spot_price_w",
                "liquidity_premium_w",
                "contract_value_w",
                "lock_ratio_proxy_w",
                "curve_match_score_w",
                "stability_score_w",
            ],
        )
    )
    lines.extend(["### 政策风险调整结果", ""])
    lines.extend(
        _frame_as_markdown(
            policy_risk_metrics,
            [
                "week_start",
                "excess_profit_w",
                "policy_risk_penalty_w",
                "policy_risk_adjusted_excess_return_w",
                "active_excess_return_positive",
                "policy_risk_adjusted_conclusion",
            ],
        )
    )
    lines.extend(["### 滚动窗口调度", ""])
    lines.extend(_frame_as_markdown(rolling_schedule))
    lines.extend(["### 滚动验证窗口结果", ""])
    lines.extend(_frame_as_markdown(rolling_validation_metrics))
    lines.extend(["### 滚动回测窗口结果", ""])
    lines.extend(_frame_as_markdown(rolling_backtest_metrics))
    lines.extend(["### 滚动回测周度结果", ""])
    lines.extend(_frame_as_markdown(rolling_weekly_results, limit=20))
    lines.extend(["### 小时级现货修正激活摘要", ""])
    lines.extend([build_hourly_spot_activation_summary(rolling_weekly_results), ""])
    lines.extend(["### 滚动超额收益验证结果", ""])
    lines.extend(_frame_as_markdown(rolling_excess))
    lines.extend(["### 基准策略持出集结果", ""])
    lines.extend(_frame_as_markdown(benchmark_metrics))
    lines.extend(["### 消融结果", ""])
    lines.extend(_frame_as_markdown(ablation_metrics))
    lines.extend(["### 稳健性结果", ""])
    lines.extend(_frame_as_markdown(robustness_metrics))
    lines.extend(["### 完整数学结果产物索引", ""])
    lines.extend(_frame_as_markdown(raw_inventory))
    lines.extend(
        [
            "说明: 行级 15 分钟结算明细、小时现货修正明细、周度明细等大表以 raw CSV 作为完整数学结果载体；本节列出每个 raw 文件的类别、行列数和哈希。",
            "",
            "## 实现效果",
            "",
            "- 输出结构已收敛为两份正式报告和 raw 证据目录；旧分模块 Markdown 摘要不再作为产物落地。",
            "- 政策文件、结构化规则、制度状态、特征清单、训练轨迹、验证结果、滚动回测、基准、消融和稳健性结果全部进入 raw CSV/JSON/YAML。",
            "- 报告入口集中在 `reports/`，便于人工阅读和 AI 结构化复核。",
            "",
            "## 关键产物路径",
            "",
            f"- 配置快照: {relativize_path(context['output_paths']['metadata'] / 'train_config_snapshot.yaml', output_root)}",
            f"- 特征清单: {relativize_path(context['output_paths']['metrics'] / 'feature_manifest.csv', output_root)}",
            f"- 参数布局: {relativize_path(context['output_paths']['metadata'] / 'compiled_parameter_layout.json', output_root)}",
            f"- 基准比较: {relativize_path(context['output_paths']['metrics'] / 'benchmark_comparison.csv', output_root)}",
            f"- 稳健性结果: {relativize_path(context['output_paths']['metrics'] / 'robustness_metrics.csv', output_root)}",
            f"- 产物索引: {relativize_path(output_root / 'artifact_index.md', output_root)}",
            "",
            "## 复核提示",
            "",
            "- 本报告面向人工阅读，结论性数值以 raw CSV/JSON 为准。",
            "- 若远程验证与本地刷新结果不一致，应优先核对 `remote_jupyter` 拉回目录、manifest 哈希与产物时间戳。",
            "- 滚动窗口结果、正式验证集结果、正式测试集结果属于不同报告口径，不合并成单一胜负结论。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_ai_structured_report(context: dict, training: dict, validation: dict, backtest: dict, human_report_path: Path) -> dict:
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    config = context["config"]
    run_metadata = context.get("run_metadata", {})
    metrics_dir = context["output_paths"]["metrics"]
    validation_weekly_results = _frame_from_result_or_csv(validation, "weekly_results", metrics_dir / "validation_weekly_results.csv")
    contract_value_weekly = _frame_from_result_or_csv(validation, "contract_value_weekly", metrics_dir / "contract_value_weekly.csv")
    policy_risk_metrics = _frame_from_result_or_csv(validation, "policy_risk_metrics", metrics_dir / "policy_risk_adjusted_metrics.csv")
    rolling_schedule = _read_csv_if_exists(metrics_dir / "rolling_window_schedule.csv")
    rolling_validation_metrics = _frame_from_result_or_csv(backtest, "validation_metrics", metrics_dir / "rolling_validation_metrics.csv")
    rolling_backtest_metrics = _frame_from_result_or_csv(backtest, "backtest_metrics", metrics_dir / "rolling_backtest_metrics.csv")
    rolling_weekly_results = _frame_from_result_or_csv(backtest, "rolling_weekly_results", metrics_dir / "rolling_weekly_results.csv")
    rolling_excess = backtest.get("rolling_excess_return_metrics", pd.DataFrame())
    benchmark_metrics = backtest.get("benchmark_metrics", pd.DataFrame())
    ablation_metrics = backtest.get("ablation_metrics", pd.DataFrame())
    robustness_metrics = backtest.get("robustness_metrics", pd.DataFrame())
    raw_inventory = _raw_artifact_inventory(output_root)
    return {
        "report_type": "ai_structured_project_report",
        "version": config.get("version", "unknown"),
        "run_metadata": run_metadata,
        "output_layout": {
            "human_report_path": relativize_path(human_report_path, output_root),
            "ai_structured_report_path": f"reports/{config.get('version', 'unknown')}_ai_structured_report.json",
            "raw_root": "raw",
            "raw_categories": {
                "metrics": "raw/metrics",
                "metadata": "raw/metadata",
                "models": "raw/models",
                "logs": "raw/logs",
            },
            "reports_policy": "reports only contains the human report and this AI structured report",
        },
        "project_scope": {
            "subject": "湖南电力市场售电公司",
            "decision_framework": "周度中长期底仓决策 + 小时级现货边际风险修正 + 15分钟代理结算回测",
            "market_scope": ["中长期市场", "日前现货市场", "日内/实时现货市场"],
            "strong_baseline": "dynamic_lock_only",
        },
        "implementation_logic": [
            "experiment_config.yaml resolves versioned output paths",
            "prepare_project_context builds cleaned 15-minute data, hourly aggregates, weekly features, policy states, feasible domains and tensor bundles",
            "train_hybrid_pso_model searches upper and lower particles against batch_score_particles",
            "run_evaluate materializes the best particle pair on the formal validation set",
            "run_backtest retrains over expanding rolling windows and emits benchmark, ablation and robustness results",
            "final reports are generated once after all numerical outputs exist",
        ],
        "mathematical_model": {
            "weekly_state": "s_w = [x_w, p_w, f_w, a_{w-1}, r_{w-1}]",
            "weekly_action": "a_w = (delta_h_raw_w, b_raw_w)",
            "weekly_contract_position": "H_w = max(r_base * Lhat_w + clip(delta_h_raw_w, r_min_w*Lhat_w, r_max_w*Lhat_w), 0)",
            "exposure_band": "B_w = clip(B_raw_w, b_min_w*Lhat_w, b_max_w*Lhat_w)",
            "contract_curve": "pi_{w,1:24} = softmax(theta_curve @ Basis_24h)",
            "hourly_signal": "m_{w,h} combines lagged spread, load deviation, renewable deviation and session weights",
            "hourly_raw_hedge": "u_raw_{w,h} = gate_{w,h} * tanh(m_{w,h}) * B_w/N_h * limit_multiplier",
            "hourly_projection": "u_{w,h} = Proj_hour(u_raw_{w,h}; B_w, share_cap_{w,h}, ramp_cap_{w,h})",
            "settlement": "C_{w,t} = q_sched_{w,t}*(omega_lt*p_lt_eff_w + omega_da*p_da_{w,t}) + abs(q_real_{w,t}-q_sched_{w,t})*p_id_{w,t}*eta_imbalance",
            "cvar": "CVaR_alpha(w) is the mean of the worst (1-alpha) 15-minute costs",
            "reward": "r_w = X_w - lambda_tail*CVaR_alpha(w) - lambda_hedge*E_w - lambda_trade*F_w - lambda_violate*V_w",
            "objective": "J(theta_u, theta_l) = -sum_w r_w",
        },
        "configuration": {
            "split": config.get("split", {}),
            "reward": config.get("reward", {}),
            "score_kernel": config.get("score_kernel", {}),
            "hybrid_pso": config.get("hybrid_pso", {}),
            "policy_feasible_domain": config.get("policy_feasible_domain", {}),
        },
        "mathematical_results": {
            "training_runtime": {key: _json_safe_value(value) for key, value in training.get("runtime_profile", {}).items()},
            "validation_metrics": {key: _json_safe_value(value) for key, value in validation.get("metrics", {}).items()},
            "validation_weekly_results": _dataframe_records(validation_weekly_results),
            "contract_value_weekly": _dataframe_records(contract_value_weekly),
            "policy_risk_adjusted_metrics": _dataframe_records(policy_risk_metrics),
            "rolling_window_schedule": _dataframe_records(rolling_schedule),
            "rolling_validation_metrics": _dataframe_records(rolling_validation_metrics),
            "rolling_backtest_metrics": _dataframe_records(rolling_backtest_metrics),
            "rolling_weekly_results_sample": _dataframe_records(rolling_weekly_results, limit=25),
            "rolling_excess_return_metrics": _dataframe_records(rolling_excess),
            "benchmark_comparison": _dataframe_records(benchmark_metrics),
            "ablation_metrics": _dataframe_records(ablation_metrics),
            "robustness_metrics": _dataframe_records(robustness_metrics),
        },
        "raw_artifacts": _dataframe_records(raw_inventory),
    }


def main() -> dict:
    status_path = Path(os.environ.get("ELEC_RUNTIME_STATUS_PATH", Path.cwd() / ".cache" / "runtime_status.json"))
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_tracker = RuntimeStatusTracker(status_path)
    status_tracker.update(stage="初始化", phase_name="准备上下文", phase_progress=0.0, total_progress=0.0, message="加载配置与数据")
    context = prepare_project_context(Path.cwd(), logger_name="pipeline")
    context["runtime_status_path"] = status_path
    training_phase_name = build_training_phase_name(context["config"].get("training", {}).get("algorithm"))
    status_tracker.update(stage="训练", phase_name=training_phase_name, phase_progress=0.0, total_progress=0.05, message="开始训练")
    training = run_train(context)
    status_tracker.update(stage="验证", phase_name="验证模块", phase_progress=0.0, total_progress=0.33, message="开始验证")
    validation = run_evaluate(context, model=training["model"])
    status_tracker.update(stage="回测", phase_name="滚动回测", phase_progress=0.0, total_progress=0.66, message="开始回测")
    backtest = run_backtest(context, model=training["model"])
    output_root = context["output_paths"].get("root", context["output_paths"]["reports"].parent)
    run_metadata = context.get("run_metadata", fallback_run_metadata(context["config"]))
    version = context["config"]["version"]
    human_report_path = context["output_paths"]["reports"] / f"{version}_human_report.md"
    ai_structured_report_path = context["output_paths"]["reports"] / f"{version}_ai_structured_report.json"
    _clean_reports_dir(context["output_paths"]["reports"], {human_report_path.name, ai_structured_report_path.name})
    save_markdown(
        prepend_report_header(_build_human_report(context, training, validation, backtest), run_metadata, device=training["device"]),
        human_report_path,
    )
    save_json(
        _build_ai_structured_report(context, training, validation, backtest, human_report_path),
        ai_structured_report_path,
    )
    model_dir = context["output_paths"].get("models", output_root / "models")
    _persist_manifest_updates(
        context,
        {
            "compiled_layout_path": relativize_path(context["output_paths"]["metadata"] / "compiled_parameter_layout.json", output_root),
            "train_config_snapshot_path": relativize_path(context["output_paths"]["metadata"] / "train_config_snapshot.yaml", output_root),
            "feasible_domain_manifest_path": relativize_path(context["output_paths"]["metrics"] / "feasible_domain_manifest.csv", output_root),
            "policy_file_inventory_path": relativize_path(context["output_paths"]["metrics"] / "policy_file_inventory.csv", output_root),
            "policy_parse_failures_path": relativize_path(context["output_paths"]["metrics"] / "policy_parse_failures.csv", output_root),
            "policy_rule_table_path": relativize_path(context["output_paths"]["metrics"] / "policy_rule_table.csv", output_root),
            "policy_state_trace_path": relativize_path(context["output_paths"]["metrics"] / "policy_state_trace.csv", output_root),
            "market_rule_constraints_path": relativize_path(context["output_paths"]["metrics"] / "market_rule_constraints.csv", output_root),
            "feature_manifest_path": relativize_path(context["output_paths"]["metrics"] / "feature_manifest.csv", output_root),
            "weekly_metadata_path": relativize_path(context["output_paths"]["metrics"] / "weekly_metadata.csv", output_root),
            "weekly_features_path": relativize_path(context["output_paths"]["metrics"] / "weekly_features.csv", output_root),
            "model_path": relativize_path(model_dir / "hybrid_pso_model.json", output_root),
            "training_runtime_summary_path": relativize_path(context["output_paths"]["metadata"] / "training_runtime_summary.json", output_root),
            "training_trace_path": relativize_path(context["output_paths"]["metrics"] / "hybrid_pso_training_trace.csv", output_root),
            "validation_weekly_results_path": relativize_path(context["output_paths"]["metrics"] / "validation_weekly_results.csv", output_root),
            "validation_metrics_path": relativize_path(context["output_paths"]["metrics"] / "validation_metrics.csv", output_root),
            "contract_value_weekly_path": relativize_path(context["output_paths"]["metrics"] / "contract_value_weekly.csv", output_root),
            "risk_factor_manifest_path": relativize_path(context["output_paths"]["metrics"] / "risk_factor_manifest.csv", output_root),
            "policy_risk_adjusted_metrics_path": relativize_path(context["output_paths"]["metrics"] / "policy_risk_adjusted_metrics.csv", output_root),
            "human_report_path": relativize_path(human_report_path, output_root),
            "ai_structured_report_path": relativize_path(ai_structured_report_path, output_root),
            "rolling_window_schedule_path": relativize_path(context["output_paths"]["metrics"] / "rolling_window_schedule.csv", output_root),
            "rolling_parameter_snapshots_path": relativize_path(context["output_paths"]["metrics"] / "rolling_parameter_snapshots.csv", output_root),
            "rolling_validation_metrics_path": relativize_path(context["output_paths"]["metrics"] / "rolling_validation_metrics.csv", output_root),
            "rolling_backtest_metrics_path": relativize_path(context["output_paths"]["metrics"] / "rolling_backtest_metrics.csv", output_root),
            "rolling_weekly_results_path": relativize_path(context["output_paths"]["metrics"] / "rolling_weekly_results.csv", output_root),
            "rolling_hourly_results_path": relativize_path(context["output_paths"]["metrics"] / "rolling_hourly_results.csv", output_root),
            "rolling_settlement_results_path": relativize_path(context["output_paths"]["metrics"] / "rolling_settlement_results.csv", output_root),
            "rolling_excess_return_metrics_path": relativize_path(context["output_paths"]["metrics"] / "rolling_excess_return_metrics.csv", output_root),
            "backtest_metrics_path": relativize_path(context["output_paths"]["metrics"] / "backtest_metrics.csv", output_root),
            "benchmark_comparison_path": relativize_path(context["output_paths"]["metrics"] / "benchmark_comparison.csv", output_root),
            "ablation_metrics_path": relativize_path(context["output_paths"]["metrics"] / "ablation_metrics.csv", output_root),
            "robustness_metrics_path": relativize_path(context["output_paths"]["metrics"] / "robustness_metrics.csv", output_root),
        },
    )
    status_tracker.update(stage="完成", phase_name="全量流水线", phase_progress=1.0, total_progress=1.0, message="流水线执行完成")
    return {
        "training": training,
        "validation": validation,
        "backtest": backtest,
    }


if __name__ == "__main__":
    main()
