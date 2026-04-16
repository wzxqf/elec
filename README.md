# elec

面向湖南电力市场售电公司论文实验的多时间尺度采购策略工程。当前版本已按 `v0.38` 版本收口版维护为：

- 上层周度 HPSO 搜索不设硬限的中长期合约调整量，并保留边际敞口带宽诊断字段
- 下层小时级 HPSO 搜索不设硬限的现货合约修正量 `delta_q`
- 15 分钟日内代理结算回测
- `dynamic_lock_only` 强基准对比、滚动验证、参数搜索与论文写作用详细报告
- 政策文件清单、元数据索引、结构化规则表、解析失败清单和制度状态轨迹
- 市场规则约束清单与“规则 -> 模型层 -> 字段/函数”映射报告
- 固定随机种子下的“PSO + 退火扰动 + BP 局部精修”可复现搜索

## 配置入口

项目根目录的 `experiment_config.yaml` 是当前唯一人工修改入口。训练、验证、回测、敏感性分析、鲁棒性分析和搜索流程都从该文件读取参数；`configs/*.yaml` 仅保留为历史模板或参考，不再作为正式实验主入口。

## `v0.38` 发布边界

本版本在 `v0.36` 参数编译主线基础上，完成发布前阻塞项收口，不改变正式策略边界：

- 正式主算法名升级为 `HYBRID_PSO_V038`
- `project.version`、运行日志和模型元数据统一改为配置驱动
- `HYBRID_PSO_V038` 已进入根配置校验白名单
- 政策风险调整后 Sharpe 对零波动滚动窗口执行去失真保护
- 双层参数化 HPSO、小时级修正和 15 分钟代理结算口径保持不变
- `dynamic_lock_only`、固定持仓、无现货修正/规则对冲等基准继续输出并参与比较

## 子代理 / 模块分工

- `cache-runtime`：bundle 缓存、模拟器读取路径、基准读取路径。
- `device-reporting`：设备解析、GPU 标识、训练运行摘要。
- `analysis-search`：分析与搜索 worker-count 口径、回测尾部瓶颈说明。
- `docs-release`：`v0.25.md`、`CHANGELOG.md`、`README.md`、规格与计划文档同步。

## 运行前提

推荐环境为 `torch311`，v0.38 HPSO 搜索默认面向 CUDA 加速计算：

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

- `<version>` 由 `experiment_config.yaml` 中的 `project.version` 自动决定，例如当前版本输出到 `outputs/v0.38/`。
- 图表不再直接输出图片，统一导出为与原图表同名的 CSV 文件。
- 日志、摘要、回测报告和详细运行报告均为中文输出。
- 中长期价格估算与 15 分钟代理结算口径会在日志和报告中明确标注。
- 训练设备摘要会明确标注默认 `cpu`，以及 `mps` 是否作为可选路径启用。
- 分析/搜索 worker-count 会在配置快照里保留，默认值为 `1`。
- 运行后会额外输出 `outputs/<version>/metrics/hpso_upper_weekly_actions.csv`、`outputs/<version>/metrics/hpso_hourly_delta_q.csv`、`outputs/<version>/metrics/hpso_convergence_curve.csv`、`outputs/<version>/reports/train_config_snapshot.yaml`、`outputs/<version>/reports/feature_manifest.json`、`outputs/<version>/reports/rolling_validation_summary.md`、`outputs/<version>/reports/market_rule_constraints.md` 和 `outputs/<version>/metrics/rolling_validation_metrics.csv`。
