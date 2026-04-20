# elec

面向湖南电力市场售电公司论文实验的多时间尺度采购策略工程。当前正式版本已按 `v0.45` 收口并增强为：

- 上层周度 HPSO 搜索不设硬限的中长期合约调整量，并保留边际敞口带宽诊断字段
- 下层小时级 HPSO 搜索不设硬限的现货合约修正量 `delta_q`
- 15 分钟日内代理结算回测
- `dynamic_lock_only` 强基准对比、滚动验证、参数搜索与论文写作用详细报告
- 政策文件清单、元数据索引、结构化规则表、解析失败清单和制度状态轨迹
- 市场规则约束清单与“规则 -> 模型层 -> 字段/函数”映射报告
- 固定随机种子下的“PSO + 退火扰动 + BP 局部精修”可复现搜索
- 样本范围统一标注、`lt_price_source` 代理标签对齐、约束投影拆分报告、训练真实维度摘要与 `vxx报告.md` 正式总报告

## 配置入口

项目根目录的 `experiment_config.yaml` 仍是正式主线的默认入口。训练、验证、回测、敏感性分析、鲁棒性分析和搜索流程默认都从该文件读取参数；如需做不覆盖正式主线的参数试验，可显式传入 `--config configs/experiments/<name>.yaml` 运行独立测试配置。

## `v0.45` 发布边界

本版本在 `v0.4` 结构闭环基础上完成 `v0.45` 论文主版本收口，不改变正式策略边界：

- 正式主算法名升级为 `HYBRID_PSO_V040`
- `project.version`、运行日志、manifest 和模型元数据统一改为配置驱动
- `HYBRID_PSO_V040` 已进入根配置校验白名单
- 政策约束已进入 `feasible_domain -> projection` 正式主链
- 状态层新增 `STATE_SCHEMA` 快照与 `tensor_bundle` 审计输出，小时级状态补入营业时段/峰谷价差与偏差绝对值特征
- 政策风险调整指标扩展为稳定化指标组，并对 Sharpe 触发护栏
- 双层参数化 HPSO、小时级修正和 15 分钟代理结算口径保持不变
- `dynamic_lock_only`、固定持仓、无现货修正/规则对冲等基准继续输出并参与比较
- 回测正式补出基准比较、消融、稳健性和约束激活报告

## 子代理 / 模块分工

- `cache-runtime`：bundle 缓存、模拟器读取路径、基准读取路径。
- `device-reporting`：设备解析、GPU 标识、训练运行摘要。
- `analysis-search`：分析与搜索 worker-count 口径、回测尾部瓶颈说明。
- `docs-release`：`v0.25.md`、`CHANGELOG.md`、`README.md`、规格与计划文档同步。

## 运行前提

推荐环境为 `torch311`，`v0.4` HPSO 搜索默认面向 CUDA 加速计算：

```bash
mamba activate torch311
pip install -r requirements.txt
```

## 入口

```bash
python run_all.py
bash run_all.sh
python -m src.scripts.run_pipeline
python -m src.scripts.train
python -m src.scripts.evaluate
python -m src.scripts.backtest
python -m src.scripts.diagnostics
```

其中：

- `python run_all.py` 为统一的一键全流程入口，会自动解析 `project.version`、选择 `torch311` 环境中的 Python，并执行训练、验证、回测与报告导出。
- 如需运行独立测试配置，可使用 `python run_all.py --config configs/experiments/v0.45_param_opt_balanced.yaml`；这不会修改根配置，会按测试配置里的 `project.version` 单独写入 `outputs/<version>/`。
- `bash run_all.sh` 现在只是对 `run_all.py` 的薄封装，方便类 Unix 环境调用。
- 如需仅检查命令与输出目录而不真正执行，可运行 `python run_all.py --dry-run`。
- Windows 不再维护 `run_all.ps1` 或 `run_all.bat`，避免多入口漂移和 `cmd.exe`/PowerShell 钩子带来的额外故障面。

## 输出

运行后结果会写入：

- `outputs/<version>/logs/`
- `outputs/<version>/models/`
- `outputs/<version>/metrics/`
- `outputs/<version>/figures/`
- `outputs/<version>/reports/`

说明：

- `<version>` 由当前运行配置中的 `project.version` 自动决定；默认正式入口输出到 `outputs/v0.45/`，实验配置会写入各自独立版本目录。
- 图表不再直接输出图片，统一导出为与原图表同名的 CSV 文件。
- 日志、摘要、回测报告和详细运行报告均为中文输出。
- 中长期价格估算与 15 分钟代理结算口径会在日志和报告中明确标注。
- 训练设备摘要会明确标注默认 `cpu`，以及 `mps` 是否作为可选路径启用。
- 分析/搜索 worker-count 会在配置快照里保留，默认值为 `1`。
- 运行后会额外输出 `release_manifest.json`、`run_manifest.json`、`artifact_index.md`、`feasible_domain_manifest.csv`、`parameter_layout_audit.md`、`state_schema_snapshot.md`、`tensor_bundle_audit.md`、`benchmark_comparison.csv`、`ablation_metrics.csv`、`robustness_metrics.csv`、`constraint_activation_report.md`、`train_config_snapshot.yaml`、`training_runtime_summary.json`、`feature_manifest.json`、`market_rule_constraints.md`、`rolling_validation_metrics.csv` 和 `vxx报告.md`。

## 测试目录规范

- `pytest` 缓存、临时目录和测试结果不得落在项目根目录。
- 统一测试工作目录为 `.cache/tests/pytest/`。
- 凡是当前版本仍有效、并保留在 `tests/` 主目录参与默认 `pytest` 收集的测试脚本，一律命名为 `test_<currentversion>_<purpose>.py`。
- 文件名中的 `<currentversion>` 由 `project.version` 统一派生，规则为“去掉非字母数字字符后的版本标识”；例如 `v0.45 -> v045`。
- `tests/` 主目录只保留当前版本有效测试脚本，不放置 `.md`、`.txt`、`.rst` 等说明文档；归档说明统一放入 `已归档/tests/`。
- 推荐统一通过下面的入口运行测试：

```bash
python -m src.scripts.run_pytest tests/test_<normalized_project_version>_hpso_param_config.py -q
```

- 该入口会将：
  - cache 写到 `.cache/tests/pytest/cache`
  - basetemp 写到 `.cache/tests/pytest/tmp`
  - junit 结果写到 `.cache/tests/pytest/results/junit.xml`
- 测试全部通过后会自动删除 `.cache/tests/pytest/`；失败时保留该目录便于排查。


