# 更新日志

## v0.51

- 将当前正式版本号更新为 `project.version: v0.51`，正式输出目录切换为 `outputs/v0.51/`，活跃测试命名同步迁移为 `test_v051_<purpose>.py`。
- 继承 `v0.50` 的小时级现货修正激活修复，继续保留 `HYBRID_PSO_V040` 主线、强基准比较和远程 Jupyter 正式验证口径。
- 新增 `v0.51.md`、`docs/v0.51_architecture_implementation.md` 和 `docs/v0.51_algorithm_implementation_logic.md` 作为当前活跃说明，README、约束、状态 Schema 和策略层文档同步更新到 `v0.51` 口径。
- 补齐根目录 `run_remote_jupyter.ps1` 一键入口，默认执行 `--probe` 后再上传运行，Jupyter 密码或 token 仍只从本机环境变量读取。
- 清理测试口径更新为适配当前 `.gitignore` 策略：活跃区继续禁止旧版本模块和旧输出，`已归档/` 作为 ignored 历史区不再要求参与默认 pytest 文件存在性校验。
- 远程 Jupyter 已刷新 `outputs/v0.51`：完整 pipeline 返回 0；补齐远程 `pymupdf` 后，全量 pytest 返回 `104 passed`；正式滚动小时级现货修正出现 1537 个非零小时。

## v0.50

- 将当前正式版本号更新为 `project.version: v0.50`，正式输出目录切换为 `outputs/v0.50/`，活跃测试命名同步迁移为 `test_v050_<purpose>.py`。
- 沿用 `HYBRID_PSO_V040` 主线和湖南售电公司研究边界，不引入新的市场主体、旧式月度决策或独立下层智能体训练。
- 正式报告链收口为两份入口文件：`reports/v0.50_human_report.md` 面向人工阅读，`reports/v0.50_ai_structured_report.json` 面向 AI 深度复核。
- CSV、模型、日志、配置快照、参数布局和远程拉回记录继续按 `raw/metrics`、`raw/models`、`raw/logs`、`raw/metadata`、`raw/remote_jupyter/<run_id>` 分类保存。
- `v0.50.md`、`docs/v0.50_architecture_implementation.md` 和 `docs/v0.50_algorithm_implementation_logic.md` 作为当前活跃说明，README、约束、状态 Schema 和策略层文档同步更新到 `v0.50` 口径。
- 远程正式验证仍通过根目录 `run_remote_jupyter.py` / `run_remote_jupyter.ps1` 执行完整 pipeline 与 pytest，Jupyter 密码或 token 仅从本机环境变量读取。
- 修复小时级现货修正限额基数：`raw_spot_hedge_mwh` 现在使用制度投影后的 `exposure_band_mwh` 生成原始小时交易空间，避免原始带宽被上层粒子压到 0 时屏蔽可行域 floor。
- 新增小时级现货修正激活诊断字段，滚动周度结果可直接记录 `spot_hedge_net_mwh_w`、`spot_hedge_abs_mwh_w`、`spot_hedge_nonzero_hours_w` 和 `spot_hedge_limit_mean_mwh_w`。
- 新增禁用状态的 `hourly_spot_experiment` 配置入口与 guardrail helper，用于后续扫描死区、温度、风险带宽、交易摩擦和 `lambda_trade`，并以强基准改善、CVaR99 和 hedge error 作为筛选约束。

## v0.48

- 将当前正式版本号更新为 `project.version: v0.48`，正式输出目录切换为 `outputs/v0.48/`，活跃测试命名同步迁移为 `test_v048_<purpose>.py`。
- 继承 `v0.47` 审计修复后的制度价格与远程 Jupyter 验证口径，不改变 `HYBRID_PSO_V040` 主线。
- 新增 `v0.48.md` 与 `docs/v0.48_architecture_implementation.md` 作为当前活跃说明，`v0.47` 说明、架构说明、算法实现说明和正式产物转入归档区。
- 文档统一到 `v0.48 + HYBRID_PSO_V040 + outputs/v0.48/` 口径，保留 Jupyter 地址说明且不写入密码或 token。
- 版本迭代后的正式验证继续以根目录 `run_remote_jupyter.py` / `run_remote_jupyter.ps1` 为入口，远程执行完整 pipeline 和 pytest 后拉回 `outputs/v0.48/raw/remote_jupyter/<run_id>/`。
- 抛弃旧式分散报告逻辑：`reports/` 只保留 `v0.48_human_report.md` 和 `v0.48_ai_structured_report.json`，CSV、模型、日志、配置快照和参数布局统一归入 `raw/` 分类目录，远程拉回记录迁移到 `raw/remote_jupyter/<run_id>/`。

## v0.47

- 将当前正式版本号更新为 `project.version: v0.47`，正式输出目录切换为 `outputs/v0.47/`，活跃测试命名同步迁移为 `test_v047_<purpose>.py`。
- 继承 `v0.46` 审计修复后的制度时点、可行域合并、15 分钟实际负荷基准结算和远程验证口径，不改变 `HYBRID_PSO_V040` 主线与论文模型边界。
- 新增 `v0.47.md` 与 `docs/v0.47_architecture_implementation.md` 作为当前活跃说明，`v0.46` 说明和架构文档转入归档区。
- 删除本轮审计过程中失败或中间状态的远程 Jupyter 产物，只保留成功验证记录和历史发布产物。
- 调整远程验证文档限制：项目文档允许记录 Jupyter 地址 `http://10.26.27.72:9007/`；密码或 token 仍只允许通过本机环境变量提供，禁止写入仓库文件、配置文件、日志样例、版本文档或运行产物。
- 修复 2026-02 后中长期联动价格口径：`lt_price_w_effective` 进入评分张量、物化回测和基准策略，`lt_price_w_prev_week_da_proxy` 保留为审计字段。
- 补全 `artifact_index.md`、`release_manifest.json`、`run_manifest.json` 的全流程产物索引，覆盖模型、训练、验证、滚动回测、基准、消融、稳健性和版本总报告。
- 新增 `reports/gpt_deep_research_run_detail.md`，每次完整 pipeline 运行后记录项目实现链路、配置口径、正式验证/测试结果、滚动窗口、基准、消融、稳健性和关键产物路径，并纳入三份 manifest。
- 明确滚动外推窗口、正式验证集和正式测试集的报告口径，稳健性改为滚动测试窗口情景结算重跑。
- 将上层配置维度统一到当前编译真实维度 `182`，并把文档中的主路径统一为 `score_kernel.py -> materialize.py`。
- 新增 PowerShell 远程验证一键入口 `run_remote_jupyter.ps1`：自动解析本机 `torch311` Python，补齐非敏感 Jupyter 默认参数，默认先 `--probe` 再执行完整远程 pipeline 和 pytest；Jupyter 密码或 token 仍只从环境变量读取。

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

