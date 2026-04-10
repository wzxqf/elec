# elec

基于 Gymnasium 自定义环境与 Stable-Baselines3 PPO 的售电采购策略实验工程。当前版本已按 2026-04-10 交付说明重构为：

- 周度中长期持仓 PPO 决策
- 小时级显式规则现货修正
- 15 分钟日内代理结算回测
- 基准策略、敏感性、鲁棒性、参数搜索与论文写作用详细报告

## 运行前提

推荐环境为 `elec_env`：

```bash
mamba activate elec_env
pip install -r requirements.txt
```

## 入口

```bash
python -m src.scripts.run_pipeline
python -m src.scripts.train
python -m src.scripts.evaluate
python -m src.scripts.backtest
python -m src.scripts.diagnostics
```

## 输出

运行后结果会写入：

- `outputs/logs/`
- `outputs/models/`
- `outputs/metrics/`
- `outputs/figures/`
- `outputs/reports/`

说明：

- 图表不再直接输出图片，统一导出为与原图表同名的 CSV 文件。
- 日志、摘要、回测报告和详细运行报告均为中文输出。
- 中长期价格估算与 15 分钟代理结算口径会在日志和报告中明确标注。
