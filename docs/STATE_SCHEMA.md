# STATE_SCHEMA

## v0.4 Scope

v0.4 的状态输入按四层组织：

- Weekly：周度聚合特征，含上一周价格、价差、负荷偏差、新能源扰动、营业时段价差统计、极端事件标记与周度基础元数据。
- Policy：当前制度状态与四组前瞻制度状态，直接来自政策解析链路的结构化状态表。
- Hourly：小时级净负荷、价差、偏差绝对值/方向、营业时段标记、峰谷时段价差以及结算有效标记。
- Bounds：由 `policy_feasible_domain` 编译出的周度/小时级动作可行域与结算语义。

## Runtime Artifacts

每次运行 `prepare_project_context()` 后，会在版本目录下输出：

- `reports/state_schema_snapshot.md`
- `reports/tensor_bundle_audit.md`
- `metrics/feature_manifest.csv`
- `metrics/feasible_domain_manifest.csv`

其中 `state_schema_snapshot.md` 是本次实验真实使用字段的快照，优先级高于本说明文件。
