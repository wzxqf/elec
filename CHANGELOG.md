# 更新日志

## v0.45

- 活跃说明文档已按当前实现重写，当前正式文档体系仅保留 `README.md`、`v0.45.md`、`docs/ARCHITECTURE.md`、`docs/CONSTRAINTS.md`、`docs/STATE_SCHEMA.md`、`docs/agents.md` 与 `docs/v0.45_architecture_implementation.md`。
- 新增 `docs/v0.45_architecture_implementation.md`，系统说明真实运行链、模块职责、参数布局、评分机理、回测流程与正式产物。
- `README.md`、`CHANGELOG.md`、`v0.45.md` 和当前架构文档已统一到 `v0.45 + HYBRID_PSO_V040 + outputs/v0.45/` 口径，不再把过渡期参数试验作为活跃主线表述。
- `src/scripts/run_pipeline.py` 的版本总报告公式说明已改为当前真实字段：周度合约调整、24 小时合约曲线、小时级现货修正、15 分钟代理结算与正式奖励结构。
- `src/config/load_config.py` 当前仅放行 `HYBRID_PSO_V040`，`src/agents/hybrid_pso.py` 的默认回退版本与算法标签也已收口到当前正式口径。
- 历史训练依赖与独立旧配置已清理：`requirements.txt` 不再保留过时训练栈依赖，原独立旧配置文件已移出活跃区。
- 新增仓库清理回归校验，确保活跃文档不再出现历史训练栈描述，且当前版本架构文档必须存在。

## 归档说明

- 历史试验 YAML 保留在 `已归档/configs/experiments/`。
- 历史版本结果保留在 `已归档/outputs/`。
- 历史测试与历史说明保留在 `已归档/tests/` 与 `已归档/`。
- 历史规划/设计文档保留在 `已归档/docs/`。

