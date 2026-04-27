# STATE_SCHEMA

## v0.46 Scope

`v0.46` 的状态输入按四层组织：

- `Weekly`：周度价格、负荷、偏差、波动、营业时段统计和周度元数据
- `Policy`：当前制度状态与四组前瞻制度状态
- `Hourly`：小时级净负荷、价差、偏差方向、营业时段标记和结算有效标记
- `Bounds`：由可行域编译器生成的周度与小时级边界、带宽和结算语义

## Runtime Artifacts

每次运行正式流水线后，会在 `outputs/v0.46/` 下生成：

- `reports/state_schema_snapshot.md`
- `reports/tensor_bundle_audit.md`
- `metrics/feature_manifest.csv`
- `metrics/feasible_domain_manifest.csv`

其中 `state_schema_snapshot.md` 是本次实验真实使用字段的快照，优先级高于本说明文件。
