# PDF 摘要与正文提取模块

本目录提供独立的论文 PDF 文本抽取工具，只做摘要与正文提取、清洗和记录，不对论文内容进行改写、概括或语义补全。

## 实现计划与边界

1. `pdf_abstract_body/tei_parser.py` 解析 GROBID 返回的 TEI XML，提取标题、DOI、摘要和正文。
2. `pdf_abstract_body/grobid_client.py` 调用本地 GROBID REST API。
3. `pdf_abstract_body/text_cleaner.py` 执行保守清洗，保留正文完整性。
4. `pdf_abstract_body/fallback_pymupdf.py` 在 GROBID 不可用或单篇解析失败时执行 PyMuPDF 后备提取。
5. `extract_pdf_abstract_body.py` 作为批量命令行入口，递归扫描 PDF，输出 txt、Markdown、JSONL 和 CSV 报告。

该工具不接入训练、建模、政策解析和正式 `outputs/v0.50/` 主链路。建议把文献抽取结果写入 `文本提取/outputs/pdf_extract/`，便于和实验产物分开管理。

## 启动 GROBID

PowerShell 下可直接运行 Docker：

```powershell
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

如需在 WSL 中使用 Docker：

```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

镜像版本可按本机可拉取情况调整，核心脚本不硬编码版本号。

## 批量提取

从项目根目录运行：

```powershell
python .\文本提取\extract_pdf_abstract_body.py `
  --input .\参考文献 `
  --output .\文本提取\outputs\pdf_extract `
  --grobid-url http://localhost:8070 `
  --mode auto `
  --workers 2
```

WSL 中可用等价路径运行：

```bash
python 文本提取/extract_pdf_abstract_body.py \
  --input 参考文献 \
  --output 文本提取/outputs/pdf_extract \
  --grobid-url http://localhost:8070 \
  --mode auto \
  --workers 2
```

`auto` 模式会优先使用 GROBID；服务不可用或单篇解析失败时，会尝试 PyMuPDF 后备方案，并在报告中标记 `method=pymupdf`。

## 输出

```text
文本提取/outputs/pdf_extract/txt/{safe_stem}.abstract.txt
文本提取/outputs/pdf_extract/txt/{safe_stem}.body.txt
文本提取/outputs/pdf_extract/md/{safe_stem}.md
文本提取/outputs/pdf_extract/jsonl/records.jsonl
文本提取/outputs/pdf_extract/tei/{safe_stem}.tei.xml
文本提取/outputs/pdf_extract/extraction_report.csv
```

若不同目录存在同名 PDF，工具会在安全文件名后追加短哈希，避免覆盖。

## 常用参数

- `--mode grobid|pymupdf|auto`：默认 `auto`。
- `--workers 2`：并发数量，建议本地 GROBID 使用 2 到 4。
- `--overwrite`：覆盖已有单篇 txt/Markdown 输出。
- `--no-save-tei`：不保存 GROBID TEI XML。
- `--jsonl-text-mode full|paths`：默认在 JSONL 中保存全文；`paths` 只保留路径和统计字段。

## 测试

```powershell
python -m pytest tests/test_v050_pdf_abstract_body_extraction.py -q
```
