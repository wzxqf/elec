# 实验配置说明

本目录用于放置不覆盖正式根配置 `experiment_config.yaml` 的参数试验配置。

当前可用测试配置：

- `v0.45_param_opt_balanced.yaml`
  - 假设：在不改奖励结构和市场规则的前提下，适度提升双层 HPSO 的真实维度、粒子数和迭代数，可以改善搜索质量而不显著放大算力成本。
  - 真实维度：上层 `176 -> 186`，下层 `32 -> 48`。
  - 运行命令：`python run_all.py --config configs/experiments/v0.45_param_opt_balanced.yaml`

- `v0.45_param_opt_explore.yaml`
  - 假设：如果当前训练轨迹在尾段仍持续改进，进一步扩大初始探索尺度、群体规模和迭代预算，可能挖到更优粒子对。
  - 真实维度：上层 `176 -> 192`，下层 `32 -> 64`。
  - 运行命令：`python run_all.py --config configs/experiments/v0.45_param_opt_explore.yaml`

约束说明：

- 两份配置都只动参数化容量和 HPSO 训练参数，不改制度时序、市场边界、结算口径与强基准定义。
- 正式主线仍默认使用根目录 `experiment_config.yaml`。
- 实验输出会按各自 `project.version` 单独写入 `outputs/<version>/`，不会覆盖 `outputs/v0.45/`。

