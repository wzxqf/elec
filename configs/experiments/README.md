# 实验配置说明

本目录用于放置未来新的、不会覆盖正式根配置 `experiment_config.yaml` 的独立试验配置。

当前状态：

- 根目录 `experiment_config.yaml` 为唯一正式入口。
- `v0.46` 正式参数已并入根配置，成为当前唯一正式参数入口。
- `explore` 仅以根配置注释保留，不再作为本目录下的活跃入口。

归档说明：

- 本轮参数试验 YAML `v0.45_param_opt_balanced.yaml` 与 `v0.45_param_opt_explore.yaml` 已迁入 `已归档/configs/experiments/`。
- 如需追溯本轮参数试验的独立配置、版本号和搜索预算，请到 `已归档/configs/experiments/` 查看。

约束说明：

- 本目录后续只用于新的独立试验，不再回放已经正式收口的 `v0.45` 参数试验切换。
- 正式运行、默认测试和文档示例均以根目录 `experiment_config.yaml` 为准。
