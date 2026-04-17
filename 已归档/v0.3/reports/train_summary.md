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
- 主算法: HPSO
- 实际特征数: 110

## 超参数

- 上层粒子数/迭代数: 32 / 40
- 下层粒子数/迭代数: 24 / 35
- 退火初温/降温率: 1.0 / 0.94
- 上层/下层 BP 精修步数: 4 / 4
- 合约与现货合约硬限制: 已取消，边界字段仅作诊断与初始搜索尺度
- CPU 降级允许: 是

## 训练结果

- 最终训练轮次: 40
- 训练设备: cuda
- 是否使用 GPU: 是
- 周度动作语义: HPSO 搜索不设硬限的中长期合约调整量 + 诊断性边际敞口带宽
- 最新模型路径: HPSO 无模型文件
- 最优模型路径: HPSO 无模型文件
- 训练指标文件: d:\elec\outputs\v0.3\metrics\hpso_weekly_practice_data.csv
- 评估指标文件: d:\elec\outputs\v0.3\metrics\hpso_convergence_curve.csv
- 奖励强基准: dynamic_lock_only
- 中长期价格口径: 2026-02 前采用上一自然周日前均价代理，2026-02 起采用 40% 日前固定价 + 60% 日内联动价混合代理
- 滚动验证窗口数: 3
- 特征清单: d:\elec\outputs\v0.3\reports\feature_manifest.json
- 异常与警告记录: 详见 d:\elec\outputs\v0.3\logs\train.log
