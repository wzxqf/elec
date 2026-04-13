# elec

基于 Gymnasium 自定义环境与 Stable-Baselines3 PPO 的售电采购策略实验工程。当前版本已按 `v0.24` 说明维护为：

- 周度中长期底仓残差 PPO 决策
- 小时级 `soft_clip` 显式规则现货修正
- 15 分钟日内代理结算回测
- `dynamic_lock_only` 强基准对比、滚动验证、参数搜索与论文写作用详细报告

## 配置入口

项目根目录的 `experiment_config.yaml` 是当前唯一人工修改入口。训练、验证、回测、敏感性分析、鲁棒性分析和搜索流程都从该文件读取参数；`configs/*.yaml` 仅保留为历史模板或参考，不再作为正式实验主入口。

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

- `<version>` 由 `experiment_config.yaml` 中的 `project.version` 自动决定，例如当前版本输出到 `outputs/v0.24/`。
- 图表不再直接输出图片，统一导出为与原图表同名的 CSV 文件。
- 日志、摘要、回测报告和详细运行报告均为中文输出。
- 中长期价格估算与 15 分钟代理结算口径会在日志和报告中明确标注。
- 运行后会额外输出 `outputs/<version>/reports/train_config_snapshot.yaml`、`outputs/<version>/reports/feature_manifest.json`、`outputs/<version>/reports/rolling_validation_summary.md` 和 `outputs/<version>/metrics/rolling_validation_metrics.csv`。
