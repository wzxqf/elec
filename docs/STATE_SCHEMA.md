# STATE_SCHEMA

## v0.50 Scope

`v0.50` 的状态输入按四层组织：

- `Weekly`：周度价格、负荷、偏差、波动、营业时段统计和周度元数据
- `Policy`：当前制度状态与四组前瞻制度状态
- `Hourly`：小时级净负荷、价差、偏差方向、营业时段标记和结算有效标记
- `Bounds`：由可行域编译器生成的周度与小时级边界、带宽和结算语义

## Runtime Artifacts

每次运行正式流水线后，会在 `outputs/v0.50/` 下生成：

- `reports/v0.50_human_report.md`
- `reports/v0.50_ai_structured_report.json`
- `raw/metrics/feature_manifest.csv`
- `raw/metrics/feasible_domain_manifest.csv`
- `raw/metadata/compiled_parameter_layout.json`

其中两份正式报告汇总本次实验真实使用字段、参数布局和 raw 证据索引；字段级明细以 `raw/metrics/feature_manifest.csv` 和 `raw/metadata/compiled_parameter_layout.json` 为准。
