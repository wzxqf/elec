# V0.4 SPEC

## 定位

`v0.4` 是论文主版本的收口版，目标不是继续外扩算法，而是把版本口径、可行域约束、参数布局、审计产物和结果解释统一起来。

## 本轮已落地范围

- 根配置正式升级到 `project.version: v0.4` 与 `training.algorithm: HYBRID_PSO_V040`
- 新增 `src/policy/feasible_domain.py`
  - 将制度状态编译为周度/小时级可行域边界
- 新增 `src/policy/projection.py`
  - 对周度动作与小时级现货修正执行正式投影
- `tensor_bundle -> score_kernel -> materialize` 主链已接入可行域
  - 输出保留原始动作、投影后动作、裁剪幅度和触发标记
- `prepare_project_context()` 新增正式审计产物
  - `feasible_domain_manifest.csv`
  - `policy_feasible_domain_summary.md`
  - `parameter_layout_audit.md`
  - `release_manifest.json`
  - `run_manifest.json`
  - `artifact_index.md`
- 回测分析层新增指标护栏
  - 中位数、分位数、CVaR95/CVaR99、Sharpe 护栏和正文/附录分级
- 正式 Markdown 报告统一输出运行页眉

## 暂未在本轮全面展开的内容

- `weekly_builder.py` 的大规模状态变量外扩
- 独立的消融实验与稳健性报告全量编排
- 论文图表接口的系统性重构

这些内容保留到后续 `v0.4.x` 或下一轮专题迭代继续推进。
