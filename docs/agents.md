# agents.md

## v0.33 策略层说明

当前项目的“agent”不再指 RL 智能体，而是：

- 上层混合粒子群：搜索周度底仓残差、边际敞口带宽、合约曲线参数
- 下层混合粒子群：搜索小时级现货修正响应参数

二者共同依赖：

- `src/training/tensor_bundle.py`
- `src/training/score_kernel.py`
- `src/agents/hybrid_pso.py`

## 动作语义

- 上层输出：周度锁定比例倾向和边际敞口带宽
- 下层输出：小时级修正响应强度

## 不再使用的旧组件

- 旧强化学习训练栈
- 旧环境封装层
- 旧单次训练后整段静态推断链
