# Run Summary

## Data

- 数据文件: /Users/dk/py/elec/total.csv
- 数据主样本区间: 2025-11-01 00:00:00 至 2026-03-20 23:45:00
- 原始覆盖区间: 2025-11-01 00:00:00 至 2026-03-28 23:45:00
- 重复时间戳个数: 1
- 去重后缺失 15 分钟时点数: 0

## Split

- warmup: 2025-11
- train: 2025-12, 2026-01
- val: 2026-02
- test: 2026-03

## PPO Hyperparameters

- learning_rate: 0.0003
- n_steps: 256
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- ent_coef: 0.01
- seed: 42

## Model

- 最优模型目录: /Users/dk/py/elec/outputs/models/ppo_elec_env_best/best_model.zip
- 最终模型路径: /Users/dk/py/elec/outputs/models/ppo_elec_env.zip
- 训练 timesteps: 4096
- 训练设备: cpu

## Validation

- 累计采购成本: 5795511288.79
- CVaR(95%): 4131545.29
- 平均奖励: -5795.759786

## Backtest

- 累计采购成本: 4031736315.64
- 成本波动率: 0.00
- CVaR(95%): 3852182.79
- 套保误差: 228.5880
- 月均调整量: 0.0000

## Estimation Notes

- 中长期价格: 使用上一自然月日前小时均价估算，若后续补充真实列可自动替换。
- 实时结算: 使用日内价格作为代理结算口径。
- 训练 episode: 仅在训练月内部进行 block bootstrap 重采样。
