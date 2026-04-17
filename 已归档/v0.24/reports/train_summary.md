# 训练摘要

## 数据与样本

- 训练数据起止: 2025-11-03 00:00:00 至 2026-01-26 00:00:00
- 训练周数: 13
- 重采样方式: block_bootstrap
- 重采样序列长度: 96
- 预热周: 2025-10-27
- 政策来源文件数: 18
- 政策解析失败文件数: 0
- 根参数文件: /Users/dk/py/elec/experiment_config.yaml
- PPO 实际特征数: 110

## 超参数

- policy: MlpPolicy
- learning_rate: 0.00015
- n_steps: 256
- batch_size: 64
- gamma: 0.995
- gae_lambda: 0.97
- clip_range: 0.15
- ent_coef: 0.005
- vf_coef: 0.5
- max_grad_norm: 0.5

## 训练结果

- 最终训练轮次: 50000
- 训练设备: cpu
- 是否使用 GPU: 否
- 周度动作语义: 基准底仓残差 + 边际敞口带宽
- 最新模型路径: /Users/dk/py/elec/outputs/v0.24/models/ppo_latest.zip
- 最优模型路径: /Users/dk/py/elec/outputs/v0.24/models/ppo_best.zip
- 训练指标文件: /Users/dk/py/elec/outputs/v0.24/metrics/ppo_train_metrics.csv
- 评估指标文件: /Users/dk/py/elec/outputs/v0.24/metrics/ppo_eval_metrics.csv
- 奖励强基准: dynamic_lock_only
- 中长期价格口径: 2026-02 前采用上一自然周日前均价代理，2026-02 起采用 40% 日前固定价 + 60% 日内联动价混合代理
- 滚动验证窗口数: 3
- 特征清单: /Users/dk/py/elec/outputs/v0.24/reports/feature_manifest.json
- 异常与警告记录: 详见 outputs/logs/train.log
