# elec

面向湖南电力市场售电公司论文实验的多时间尺度采购策略工程。当前唯一正式口径是 `v0.51`，核心主线固定为：

- 周度中长期头寸参数化搜索
- 小时级现货边际修正
- 15 分钟代理结算回测
- 政策文件清单、结构化规则表、制度状态轨迹和可行域审计
- `dynamic_lock_only` 强基准、滚动验证、基准比较、消融和稳健性情景结算重跑报告

## 当前正式入口

- 唯一人工配置入口：`experiment_config.yaml`
- 唯一正式版本号：`project.version: v0.51`
- 唯一正式训练主线：`training.algorithm: HYBRID_PSO_V040`
- 唯一正式输出目录：`outputs/v0.51/`

历史试验 YAML、旧版本结果和旧规划文档均已迁入 `已归档/`，不再作为当前运行依据。

## 运行前提

推荐使用 `torch311` 环境：

```bash
mamba activate torch311
pip install -r requirements.txt
```

远程验证依赖 Jupyter 的 HTTP API 与 kernel WebSocket。当前允许在项目文档中记录 Jupyter 地址；密码或 token 只能通过本机环境变量提供，不写入仓库文件、配置文件、日志样例或运行产物。

```powershell
$env:ELEC_JUPYTER_PASSWORD="<本机环境变量提供的 Jupyter 密码>"
.\run_remote_jupyter.ps1
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

- `outputs/v0.51/reports/`
- `outputs/v0.51/raw/`

关键产物包括：

- `release_manifest.json`
- `run_manifest.json`
- `artifact_index.md`
- `reports/v0.51_human_report.md`
- `reports/v0.51_ai_structured_report.json`
- `raw/metadata/train_config_snapshot.yaml`
- `raw/metadata/compiled_parameter_layout.json`
- `raw/metadata/training_runtime_summary.json`
- `raw/models/hybrid_pso_model.json`
- `raw/metrics/feature_manifest.csv`
- `raw/metrics/feasible_domain_manifest.csv`
- `raw/metrics/benchmark_comparison.csv`
- `raw/metrics/ablation_metrics.csv`
- `raw/metrics/robustness_metrics.csv`
- `raw/metrics/rolling_settlement_results.csv`
- `raw/metrics/rolling_validation_metrics.csv`
- `raw/metrics/rolling_window_schedule.csv`

`reports/` 只保留两份入口报告：面向人的 Markdown 报告和面向 AI 复核的结构化 JSON 报告。CSV、模型、日志、配置快照和参数布局均归入 `raw/` 分类目录，不再生成分散的分模块 Markdown 中间报告。

## 参考文献处理范围

参考文献整理只在 `参考文献/` 目录内进行。该目录保存原始 `参考文献列表.docx`、重组任务计划、旧编号盘点矩阵、删减降级清单、PSO/CVaR 新增候选、重组版 Markdown 与重组版 DOCX。

原始文献表不直接覆盖；重组结果使用 `_重组版` 后缀另存。参考文献综述的当前口径应服务于市场机制与中长期-现货衔接、售电公司风险管理、HPSO/PSO 约束优化和 CVaR 尾部风险评价；智能决策与预测类文献仅保留为背景。政策文件不作为学术参考文献编号处理。

## 测试规范

- 正式版本迭代后的验证不再在本地直接跑 pytest，而是上传当前工作区到 Jupyter 后在服务器 `torch311` kernel 内执行。
- `pytest` 缓存、临时目录和结果统一写入 `.cache/tests/pytest/`
- 当前版本测试文件统一命名为 `test_v051_<purpose>.py`
- `tests/` 主目录只保留当前版本有效测试脚本
- 历史测试与说明统一归档到 `已归档/tests/`

正式远程验证入口：

```powershell
.\run_remote_jupyter.ps1
```

该命令会自动解析本机 `D:\miniforge\envs\torch311\python.exe` 等候选解释器，补齐 Jupyter URL、kernel 和远程环境的非敏感默认值，先执行 `--probe` 核对远程 `torch311` Python，再打包上传当前仓库。Jupyter 侧默认依次执行 `python run_all.py` 与 `python -m src.scripts.run_pytest tests -q`，把服务器结果拉回 `outputs/v0.51/raw/remote_jupyter/<run_id>/`，并在完整远程 pipeline 返回码为 0 时用拉回的 `outputs/v0.51` 覆盖本地正式结果区；`raw/remote_jupyter/` 运行记录会保留。密码或 token 仍只通过 `ELEC_JUPYTER_PASSWORD` 或 `ELEC_JUPYTER_TOKEN` 提供。当前服务器返回的 Jupyter kernel spec 为 `python3`；当前已纳入自动发现的远程实测路径为 `/research/miniforge3/envs/torch311/bin/python`；如路径变化，可在 PowerShell 入口中使用 `-RemotePython /path/to/torch311/bin/python` 覆盖自动发现。`src.scripts.run_pytest` 仍保留为服务器内部测试入口，本地只在调试远程运行器本身时使用。`-ProbeOnly` 只检查连接和解释器，`-SkipProbe` 跳过预检，`-DryRun` 只打印本地与远程运行计划。

## 文档入口

- 总体架构与实现机理：`docs/v0.51_architecture_implementation.md`
- 当前架构索引：`docs/ARCHITECTURE.md`
- 当前约束边界：`docs/CONSTRAINTS.md`
- 当前状态结构：`docs/STATE_SCHEMA.md`
- 当前版本发布说明：`v0.51.md`
