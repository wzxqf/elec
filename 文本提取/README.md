# PDF 摘要与正文提取模块

本目录提供独立的论文 PDF 文本抽取工具，只做摘要与正文提取、清洗和记录，不对论文内容进行改写、概括或语义补全。

## 当前结论

本批 `参考文献/` PDF 已验证过 GROBID、GROBID+回退、PyMuPDF 三种路线。GROBID 对中文论文的正文边界识别明显不稳定，多个中文文献只抽出数百字或更少，整体效果低于 `文本提取/outputs/pdf_extract/` 中的 PyMuPDF 结果。

当前正式文本提取口径固定为：

- 默认使用 `--mode pymupdf`
- 不启动、不安装、不依赖 GROBID
- 原始 PDF 作为引用核验来源
- `文本提取/outputs/pdf_extract/` 作为 GPT 深度研究的主输入

## 实现计划与边界

1. `pdf_abstract_body/fallback_pymupdf.py` 逐页提取 PDF 文本，并按关键词做摘要、正文与参考文献边界切分。
2. `pdf_abstract_body/text_cleaner.py` 执行保守清洗，优先保留正文完整性。
3. `extract_pdf_abstract_body.py` 作为批量命令行入口，递归扫描 PDF，输出 txt、Markdown、JSONL 和 CSV 报告。
4. `pdf_abstract_body/tei_parser.py`、`grobid_client.py` 仅作为历史实验代码保留，不进入当前推荐流程。

该工具不接入训练、建模、政策解析和正式 `outputs/v0.50/` 主链路。文献抽取结果写入 `文本提取/outputs/pdf_extract/`，与实验产物分开管理。

## 批量提取

从项目根目录运行：

```powershell
python .\文本提取\extract_pdf_abstract_body.py `
  --input .\参考文献 `
  --output .\文本提取\outputs\pdf_extract `
  --mode pymupdf `
  --workers 4 `
  --overwrite
```

WSL 中可用等价路径运行：

```bash
python 文本提取/extract_pdf_abstract_body.py \
  --input 参考文献 \
  --output 文本提取/outputs/pdf_extract \
  --mode pymupdf \
  --workers 4 \
  --overwrite
```

## 输出

```text
文本提取/outputs/pdf_extract/txt/{safe_stem}.abstract.txt
文本提取/outputs/pdf_extract/txt/{safe_stem}.body.txt
文本提取/outputs/pdf_extract/md/{safe_stem}.md
文本提取/outputs/pdf_extract/jsonl/records.jsonl
文本提取/outputs/pdf_extract/extraction_report.csv
```

当前有效结果：

```text
records: 53
success: 37
partial: 16
failed: 0
method: pymupdf
```

`partial` 表示该文献正文已提取，但摘要边界未可靠识别。做深度研究时可继续使用正文；涉及摘要原文、页码、图表、公式或关键引用时回查原始 PDF。

## 常用参数

- `--mode pymupdf`：当前推荐模式。
- `--workers 4`：并发数量。
- `--overwrite`：覆盖已有单篇 txt/Markdown 输出。
- `--jsonl-text-mode full|paths`：默认在 JSONL 中保存全文；`paths` 只保留路径和统计字段。

## 测试

```powershell
D:\miniforge\envs\torch311\python.exe -m src.scripts.run_pytest tests/test_v050_pdf_abstract_body_extraction.py -q
```
