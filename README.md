# elec

面向湖南电力市场售电公司论文实验的多时间尺度采购策略工程。当前唯一正式口径是 `v0.45`，核心主线固定为：

- 周度中长期头寸参数化搜索
- 小时级现货边际修正
- 15 分钟代理结算回测
- 政策文件清单、结构化规则表、制度状态轨迹和可行域审计
- `dynamic_lock_only` 强基准、滚动验证、基准比较、消融和稳健性报告

## 当前正式入口

- 唯一人工配置入口：`experiment_config.yaml`
- 唯一正式版本号：`project.version: v0.45`
- 唯一正式训练主线：`training.algorithm: HYBRID_PSO_V040`
- 唯一正式输出目录：`outputs/v0.45/`

历史试验 YAML、旧版本结果和旧规划文档均已迁入 `已归档/`，不再作为当前运行依据。

## 运行前提

推荐使用 `torch311` 环境：

```bash
mamba activate torch311
pip install -r requirements.txt
```

## 运行入口

```bash
python run_all.py
bash run_all.sh
python -m src.scripts.run_pipeline
python -m src.scripts.train
python -m src.scripts.evaluate
python -m src.scripts.backtest
python -m src.scripts.diagnostics
```

说明：

- `python run_all.py` 是统一的一键入口，会解析根配置、选择 `torch311` Python 并执行全流程。
- `bash run_all.sh` 只是对 `run_all.py` 的薄封装。
- 只检查命令与输出目录时，可运行 `python run_all.py --dry-run`。
- 当前不再维护 `run_all.ps1` 或 `run_all.bat`。

## 输出结构

运行后正式产物写入：

- `outputs/v0.45/logs/`
- `outputs/v0.45/models/`
- `outputs/v0.45/metrics/`
- `outputs/v0.45/figures/`
- `outputs/v0.45/reports/`

关键产物包括：

- `release_manifest.json`
- `run_manifest.json`
- `artifact_index.md`
- `feasible_domain_manifest.csv`
- `parameter_layout_audit.md`
- `state_schema_snapshot.md`
- `tensor_bundle_audit.md`
- `benchmark_comparison.csv`
- `ablation_metrics.csv`
- `robustness_metrics.csv`
- `constraint_activation_report.md`
- `train_config_snapshot.yaml`
- `training_runtime_summary.json`
- `feature_manifest.json`
- `market_rule_constraints.md`
- `rolling_validation_metrics.csv`
- `v0.45报告.md`

## 测试规范

- `pytest` 缓存、临时目录和结果统一写入 `.cache/tests/pytest/`
- 当前版本测试文件统一命名为 `test_v045_<purpose>.py`
- `tests/` 主目录只保留当前版本有效测试脚本
- 历史测试与说明统一归档到 `已归档/tests/`

推荐入口：

```bash
python -m src.scripts.run_pytest tests/test_v045_repo_cleanup.py -q
```

## 文档入口

- 总体架构与实现机理：`docs/v0.45_architecture_implementation.md`
- 当前架构索引：`docs/ARCHITECTURE.md`
- 当前约束边界：`docs/CONSTRAINTS.md`
- 当前状态结构：`docs/STATE_SCHEMA.md`
- 当前版本发布说明：`v0.45.md`


