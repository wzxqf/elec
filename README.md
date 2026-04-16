# elec

面向湖南电力市场售电公司论文实验的多时间尺度采购策略工程。当前版本已按 `v0.31` 制度约束归集版维护为：

- 上层周度 HPSO 搜索不设硬限的中长期合约调整量，并保留边际敞口带宽诊断字段
- 下层小时级 HPSO 搜索不设硬限的现货合约修正量 `delta_q`
- 15 分钟日内代理结算回测
- `dynamic_lock_only` 强基准对比、滚动验证、参数搜索与论文写作用详细报告
- 政策文件清单、元数据索引、结构化规则表、解析失败清单和制度状态轨迹
- 市场规则约束清单与“规则 -> 模型层 -> 字段/函数”映射报告
- 固定随机种子下的“PSO + 退火扰动 + BP 局部精修”可复现搜索

## 配置入口

项目根目录的 `experiment_config.yaml` 是当前唯一人工修改入口。训练、验证、回测、敏感性分析、鲁棒性分析和搜索流程都从该文件读取参数；`configs/*.yaml` 仅保留为历史模板或参考，不再作为正式实验主入口。

## `v0.31` 发布边界

本版本在 `v0.3` 双层 HPSO 主线基础上，补齐市场规则约束归集、制度时点校验和可查阅文本呈现。

- 上层主变量为相对 `dynamic_lock_only` 的中长期合约调整量，模型输出不再按 `delta_lock_cap`、`delta_h_max` 或锁定比例上下界硬裁剪。
- 下层不再以显式规则作为主决策机制，而是由 HPSO 搜索小时级 `delta_q`；非负、带宽、平滑和辅助服务字段保留为诊断，不再改写现货合约输出。
- 主目标不再使用 PPO 三层 reward 作为训练目标，改为综合成本风险目标。
- 当前制度状态按周初 `week_start` 判定，避免后期政策在周内生效时前移覆盖周初决策；前瞻制度状态继续保留。
- 新增 `market_rule_constraints.md`，集中呈现中长期、现货、零售、新能源机制电价、中长期价格联动、辅助服务和计量约束。
- 每个版本的 `v*.md` 必须记录当版实际使用的数学公式，至少覆盖 HPSO 更新式、上层目标、下层目标、结算式和已取消的硬约束。
- 15 分钟实时市场仍只作为结算与回测层，不扩展为第三层优化。
- `dynamic_lock_only`、固定持仓、无现货修正/规则对冲等基准继续输出并参与比较。
- 若配置请求 CUDA 但当前环境不可用，HPSO 会按 `hpso.allow_cpu` 控制是否降级，并在摘要中记录实际设备。

## 子代理 / 模块分工

- `cache-runtime`：bundle 缓存、模拟器读取路径、基准读取路径。
- `device-reporting`：设备解析、GPU 标识、训练运行摘要。
- `analysis-search`：分析与搜索 worker-count 口径、回测尾部瓶颈说明。
- `docs-release`：`v0.25.md`、`CHANGELOG.md`、`README.md`、规格与计划文档同步。

## 运行前提

推荐环境为 `torch311`，v0.31 HPSO 搜索默认面向 CUDA 加速计算：

```bash
mamba activate torch311
pip install -r requirements.txt
```

## 入口

```bash
bash run_all.sh
.\run_all.ps1
run_all.bat
python -m src.scripts.run_pipeline
python -m src.scripts.train
python -m src.scripts.evaluate
python -m src.scripts.backtest
python -m src.scripts.diagnostics
```

其中：

- `bash run_all.sh` 为根目录一键全流程入口，固定使用 `mamba run -n torch311` 执行训练、验证、回测与报告导出。
- Windows 环境推荐运行 `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`；也可直接双击 `run_all.bat`。
- 如需仅检查命令与输出目录而不真正执行，可运行 `bash run_all.sh --dry-run`。
- Windows 下对应调试命令为 `powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -DryRun`。
- 若运行 `run_all.bat` 时先打印“系统找不到指定的路径”，通常是本机 `cmd.exe` 的 AutoRun 里存在失效 conda/mamba hook；直接运行 `run_all.ps1` 可避开该问题。

## 输出

运行后结果会写入：

- `outputs/<version>/logs/`
- `outputs/<version>/models/`
- `outputs/<version>/metrics/`
- `outputs/<version>/figures/`
- `outputs/<version>/reports/`

说明：

- `<version>` 由 `experiment_config.yaml` 中的 `project.version` 自动决定，例如当前版本输出到 `outputs/v0.31/`。
- 图表不再直接输出图片，统一导出为与原图表同名的 CSV 文件。
- 日志、摘要、回测报告和详细运行报告均为中文输出。
- 中长期价格估算与 15 分钟代理结算口径会在日志和报告中明确标注。
- 训练设备摘要会明确标注默认 `cpu`，以及 `mps` 是否作为可选路径启用。
- 分析/搜索 worker-count 会在配置快照里保留，默认值为 `1`。
- 运行后会额外输出 `outputs/<version>/metrics/hpso_upper_weekly_actions.csv`、`outputs/<version>/metrics/hpso_hourly_delta_q.csv`、`outputs/<version>/metrics/hpso_convergence_curve.csv`、`outputs/<version>/reports/train_config_snapshot.yaml`、`outputs/<version>/reports/feature_manifest.json`、`outputs/<version>/reports/rolling_validation_summary.md`、`outputs/<version>/reports/market_rule_constraints.md` 和 `outputs/<version>/metrics/rolling_validation_metrics.csv`。
