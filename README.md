# elec

基于 Gymnasium 自定义环境与 Stable-Baselines3 PPO 的售电采购策略实验工程。项目使用本地真实 `total.csv` 数据，构建：

- 月度中长期持仓 PPO 决策
- 小时级规则型现货修正
- 15 分钟代理结算回测
- 基准策略、敏感性、鲁棒性与参数搜索输出

## 运行前提

推荐环境为 `elec_env`：

```bash
mamba activate elec_env
pip install -r requirements.txt
```

训练设备默认使用 `auto` 模式：

- 若存在 CUDA，则优先使用 CUDA GPU
- 若为 Apple Silicon 且 MPS 可用，则使用 MPS GPU
- 若以上均不可用，则回退到 CPU

## 入口

```bash
python -m src.scripts.run_pipeline
python -m src.scripts.train
python -m src.scripts.evaluate
python -m src.scripts.backtest
```

## 输出

运行后结果会写入：

- `outputs/logs/`
- `outputs/models/`
- `outputs/metrics/`
- `outputs/figures/`
- `outputs/reports/`

所有估算口径会在日志和报告中明确标注。
