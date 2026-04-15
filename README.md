# elec

基于 Gymnasium 自定义环境与 Stable-Baselines3 PPO 的售电采购策略实验工程。当前版本已按 `v0.25` 性能版维护为：

- 周度中长期底仓残差 PPO 决策
- 小时级 `soft_clip` 显式规则现货修正
- 15 分钟日内代理结算回测
- `dynamic_lock_only` 强基准对比、滚动验证、参数搜索与论文写作用详细报告
- 周度切片缓存、lookup/index 优化、重复 DataFrame 访问收敛与回测尾部瓶颈修复
- `analysis.worker_count` / `search.worker_count` 作为可选并行控制项
- 默认训练设备保持 `cpu`，`mps` 仅作为可选加速路径

## 配置入口

项目根目录的 `experiment_config.yaml` 是当前唯一人工修改入口。训练、验证、回测、敏感性分析、鲁棒性分析和搜索流程都从该文件读取参数；`configs/*.yaml` 仅保留为历史模板或参考，不再作为正式实验主入口。

## `v0.25` 发布边界

本版本只做工程实现提速，不改变策略语义、奖励含义、结算口径或输出字段。

- 环境侧优先引入周度预切片缓存和缓存命中读取。
- 相邻 lookup/index 逻辑统一收敛，减少重复 `set_index()`、过滤和拷贝。
- 回测尾部的单核拖尾问题通过同一条缓存与读取路径收敛处理。
- 默认训练设备保持 `cpu`，`mps` 与 `auto` 只作为可选路径。
- 分析与搜索流程提供可选 worker-count 配置，默认保持单 worker。
- 结果报告继续围绕 `dynamic_lock_only` 强基准展开。

## 子代理 / 模块分工

- `cache-runtime`：bundle 缓存、模拟器读取路径、基准读取路径。
- `device-reporting`：设备解析、GPU 标识、训练运行摘要。
- `analysis-search`：分析与搜索 worker-count 口径、回测尾部瓶颈说明。
- `docs-release`：`v0.25.md`、`CHANGELOG.md`、`README.md`、规格与计划文档同步。

## 运行前提

推荐环境为 `elec_env`：

```bash
mamba activate elec_env
pip install -r requirements.txt
```

## 入口

```bash
bash run_all.sh
python -m src.scripts.run_pipeline
python -m src.scripts.train
python -m src.scripts.evaluate
python -m src.scripts.backtest
python -m src.scripts.diagnostics
```

其中：

- `bash run_all.sh` 为根目录一键全流程入口，固定使用 `mamba run -n elec_env` 执行训练、验证、回测与报告导出。
- 如需仅检查命令与输出目录而不真正执行，可运行 `bash run_all.sh --dry-run`。

## 输出

运行后结果会写入：

- `outputs/<version>/logs/`
- `outputs/<version>/models/`
- `outputs/<version>/metrics/`
- `outputs/<version>/figures/`
- `outputs/<version>/reports/`

说明：

- `<version>` 由 `experiment_config.yaml` 中的 `project.version` 自动决定，例如当前版本输出到 `outputs/v0.25/`。
- 图表不再直接输出图片，统一导出为与原图表同名的 CSV 文件。
- 日志、摘要、回测报告和详细运行报告均为中文输出。
- 中长期价格估算与 15 分钟代理结算口径会在日志和报告中明确标注。
- 训练设备摘要会明确标注默认 `cpu`，以及 `mps` 是否作为可选路径启用。
- 分析/搜索 worker-count 会在配置快照里保留，默认值为 `1`。
- 运行后会额外输出 `outputs/<version>/reports/train_config_snapshot.yaml`、`outputs/<version>/reports/feature_manifest.json`、`outputs/<version>/reports/rolling_validation_summary.md` 和 `outputs/<version>/metrics/rolling_validation_metrics.csv`。
