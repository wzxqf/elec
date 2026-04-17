# 训练摘要

## 数据与样本

- 训练数据起止: 2025-11-03 00:00:00 至 2026-01-26 00:00:00
- 训练周数: 13
- 重采样方式: block_bootstrap
- 重采样序列长度: 96
- 预热周: 2025-10-27
- 政策来源文件数: 18
- 政策解析失败文件数: 0
- 根参数文件: D:\elec\experiment_config.yaml
- 主算法: HPSO_PARAM_POLICY
- 实际特征数: 110

## 超参数

- HPSO 参数维度: 64
- 粒子数/迭代数: 48 / 80
- bootstrap 序列长度/块大小: 96 / 3
- 验证与回测语义: 加载训练后 theta 推断，不重新优化目标周
- 下层修正语义: 因果滚动修正

## 训练结果

- 最终训练轮次: 80
- 训练设备: cuda
- 主算设备: cpu
- 并行 worker: 1
- 是否使用 GPU: 是
- 周度动作语义: HPSO 参数化策略 theta 推断底仓残差 + 边际敞口带宽 + 24小时曲线
- 最新模型路径: d:\elec\outputs\v0.32\models\hpso_param_policy.json
- 最优模型路径: d:\elec\outputs\v0.32\models\hpso_param_policy.json
- 训练指标文件: d:\elec\outputs\v0.32\metrics\hpso_weekly_practice_data.csv
- 评估指标文件: d:\elec\outputs\v0.32\metrics\hpso_convergence_curve.csv
- 奖励强基准: dynamic_lock_only
- 中长期价格口径: 2026-02 前采用上一自然周日前均价代理，2026-02 起采用 40% 日前固定价 + 60% 日内联动价混合代理
- 滚动验证窗口数: 3
- 特征清单: d:\elec\outputs\v0.32\reports\feature_manifest.json
- 异常与警告记录: 详见 d:\elec\outputs\v0.32\logs\train.log
