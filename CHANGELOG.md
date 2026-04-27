# 更新日志

## v0.46

- 将当前正式版本固化为 `project.version: v0.46`，正式输出目录切换为 `outputs/v0.46/`，活跃测试命名同步迁移为 `test_v046_<purpose>.py`。
- 继承 `v0.45` 已验证的 `HYBRID_PSO_V040` 主线与 balanced 参数口径，不改变论文模型边界、制度约束和强基准奖励结构。
- 根目录 `run_remote_jupyter.py` 作为正式验证入口继续负责上传、远程运行 pipeline/pytest、拉回服务器产物，并在完整远程运行成功后覆盖本地 `outputs/v0.46/` 正式结果区。
- `v0.45` 当前说明、架构说明和正式产物转入归档区，根目录和活跃文档只保留 `v0.46` 作为当前口径。
- 发布流程同步清理已合并的 `codex-v045-spot-hedge-profit` worktree 与本地分支，减少历史迭代残留。

## v0.45

- 新增根目录 Jupyter 远程验证入口 `run_remote_jupyter.py`：本地打包上传当前工作区，在服务器 Jupyter kernel 内解析 `ELEC_REMOTE_ENV=torch311` 的 Python 后依次运行 `run_all.py` 与 `src.scripts.run_pytest`，再把服务器产物拉回 `outputs/v0.45/remote_jupyter/<run_id>/`；实测服务器 kernel spec 为 `python3`，项目解释器路径为 `/research/miniforge3/envs/torch311/bin/python`，通过 `--probe` 核对项目运行 Python 路径。
- 修复远程 Jupyter 复验兼容性：源码包保留 `已归档/outputs/` 下的空目录契约，`run_all.py` 同时支持 Windows `python.exe` 与 Linux `bin/python` 形式的 `torch311` 环境解析。
- 完整远程 pipeline 返回码为 0 后，默认使用拉回的服务器 `outputs/<version>` 覆盖本地正式结果区；`outputs/<version>/remote_jupyter/` 保留每次远程运行审计记录，`--skip-pipeline` 不触发覆盖，调试时可用 `--no-sync-local-output` 禁止覆盖。
- 正式测试口径改为远程 Jupyter 执行，本地不再直接运行版本迭代后的 pytest；Jupyter 密码或 token 仅允许通过环境变量提供，并通过 `.gitignore` 排除本地 `.env` 凭据文件。
- 修正小时级现货修正的结算语义：`spot_hedge_mwh` 以带符号净修正量进入计划电量，交易摩擦仍按绝对成交量计量。
- 新增小时级 no-trade gate，在信号未覆盖交易摩擦与预测不确定性前抑制现货修正。
- 训练奖励的强基准从单一 `dynamic_lock_only` 扩展为 `[0.50, 0.55, 0.60]` 合约比例基准族，避免模型只跑赢弱于当前持出集最优的基准。
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

