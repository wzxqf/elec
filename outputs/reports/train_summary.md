# 训练摘要

## 数据与样本

- 训练数据起止: 2025-11-03 00:00:00 至 2026-01-26 00:00:00
- 训练周数: 13
- 重采样方式: block_bootstrap
- 重采样序列长度: 96
- 预热周: 2025-10-27

## 超参数

- policy: MlpPolicy
- learning_rate: 0.0003
- n_steps: 128
- batch_size: 32
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- vf_coef: 0.5
- max_grad_norm: 0.5

## 训练结果

- 最终训练轮次: 4096
- 训练设备: cpu
- 是否使用 GPU: 否
- 最新模型路径: /Users/dk/py/elec/outputs/models/ppo_latest.zip
- 最优模型路径: /Users/dk/py/elec/outputs/models/ppo_best.zip
- 训练指标文件: /Users/dk/py/elec/outputs/metrics/ppo_train_metrics.csv
- 评估指标文件: /Users/dk/py/elec/outputs/metrics/ppo_eval_metrics.csv
- 异常与警告记录: 详见 outputs/logs/train.log
