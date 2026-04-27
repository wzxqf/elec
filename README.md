# elec

面向湖南电力市场售电公司论文实验的多时间尺度采购策略工程。当前唯一正式口径是 `v0.46`，核心主线固定为：

- 周度中长期头寸参数化搜索
- 小时级现货边际修正
- 15 分钟代理结算回测
- 政策文件清单、结构化规则表、制度状态轨迹和可行域审计
- `dynamic_lock_only` 强基准、滚动验证、基准比较、消融和稳健性报告

## 当前正式入口

- 唯一人工配置入口：`experiment_config.yaml`
- 唯一正式版本号：`project.version: v0.46`
- 唯一正式训练主线：`training.algorithm: HYBRID_PSO_V040`
- 唯一正式输出目录：`outputs/v0.46/`

历史试验 YAML、旧版本结果和旧规划文档均已迁入 `已归档/`，不再作为当前运行依据。

## 运行前提

推荐使用 `torch311` 环境：

```bash
mamba activate torch311
pip install -r requirements.txt
```

远程验证依赖 Jupyter 的 HTTP API 与 kernel WebSocket，凭据只允许通过本机环境变量提供，不写入仓库文件：

```powershell
$env:ELEC_JUPYTER_URL="http://10.26.27.72:9007/"
$env:ELEC_JUPYTER_PASSWORD="<由老师提供的 Jupyter 密码>"
$env:ELEC_JUPYTER_KERNEL="python3"
$env:ELEC_REMOTE_ENV="torch311"
D:\miniforge\envs\torch311\python.exe run_remote_jupyter.py --probe
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

- `outputs/v0.46/logs/`
- `outputs/v0.46/models/`
- `outputs/v0.46/metrics/`
- `outputs/v0.46/figures/`
- `outputs/v0.46/reports/`

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
- `v0.46报告.md`

## 测试规范

- 正式版本迭代后的验证不再在本地直接跑 pytest，而是上传当前工作区到 Jupyter 后在服务器 `torch311` kernel 内执行。
- `pytest` 缓存、临时目录和结果统一写入 `.cache/tests/pytest/`
- 当前版本测试文件统一命名为 `test_v046_<purpose>.py`
- `tests/` 主目录只保留当前版本有效测试脚本
- 历史测试与说明统一归档到 `已归档/tests/`

正式远程验证入口：

```powershell
D:\miniforge\envs\torch311\python.exe run_remote_jupyter.py
```

该命令会打包上传当前仓库，默认在 Jupyter 侧解析 `ELEC_REMOTE_ENV=torch311` 的 Python 后，依次执行 `python run_all.py` 与 `python -m src.scripts.run_pytest tests -q`，把服务器结果拉回 `outputs/v0.46/remote_jupyter/<run_id>/`，并在完整远程 pipeline 返回码为 0 时用拉回的 `outputs/v0.46` 覆盖本地正式结果区；`remote_jupyter/` 运行记录会保留。当前服务器返回的 Jupyter kernel spec 为 `python3`；`--probe` 会打印 kernel Python 路径和项目运行 Python 路径，用于确认远程 `torch311` 环境是否可解析。当前已纳入自动发现的实测路径为 `/research/miniforge3/envs/torch311/bin/python`；如路径变化，可用 `--remote-python /path/to/torch311/bin/python` 覆盖自动发现。`src.scripts.run_pytest` 仍保留为服务器内部测试入口，本地只在调试远程运行器本身时使用。`--skip-pipeline` 不会覆盖本地正式结果；调试时也可加 `--no-sync-local-output` 禁止覆盖。

## 文档入口

- 总体架构与实现机理：`docs/v0.46_architecture_implementation.md`
- 当前架构索引：`docs/ARCHITECTURE.md`
- 当前约束边界：`docs/CONSTRAINTS.md`
- 当前状态结构：`docs/STATE_SCHEMA.md`
- 当前版本发布说明：`v0.46.md`


