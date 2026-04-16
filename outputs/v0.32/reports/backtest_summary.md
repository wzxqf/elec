# 回测摘要

- 使用模型版本: d:\elec\outputs\v0.32\models\hpso_param_policy.json
- 主算法: HPSO_PARAM_POLICY
- 回测时间范围: 2026-03-02 00:00:00 至 2026-03-16 00:00:00
- 结算口径: 15分钟日内代理结算口径
- 中长期价格口径: 2026-02 前采用上一自然周日前均价代理，2026-02 起采用 40% 日前固定价 + 60% 日内联动价混合代理
- 奖励强基准: dynamic_lock_only
- 根参数文件: D:\elec\experiment_config.yaml
- HPSO_PARAM_POLICY 总采购成本: 3613162887.09
- HPSO_PARAM_POLICY 周度成本波动率: 167231601.03
- HPSO_PARAM_POLICY CVaR: 1426707698.29
- HPSO_PARAM_POLICY 套保误差: 1363.5954
- 强边界裁剪次数: 0
- 平滑压缩次数: 0
- soft_clip 触发次数: 0
- 非负裁剪次数: 0

## 主策略与基准策略比较

- hpso_param_policy: 总采购成本=3613162887.09, CVaR=1426707698.29, 相对 dynamic_lock_only 成本差=-124631285.99
- rule_only: 总采购成本=3712384171.65, CVaR=1438113394.59, 相对 dynamic_lock_only 成本差=-25410001.43
- fixed_lock: 总采购成本=3737794173.08, CVaR=1488543328.28, 相对 dynamic_lock_only 成本差=0.00
- dynamic_lock_only: 总采购成本=3737794173.08, CVaR=1488543328.28, 相对 dynamic_lock_only 成本差=0.00
