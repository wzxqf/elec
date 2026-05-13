# Codex 任务说明：批量提取论文 PDF 的摘要与正文文本

## 2026-05-13 修订结论

本项目已经实测过 GROBID、GROBID+回退、PyMuPDF 三条路线。GROBID 对当前 `参考文献/` 中的中文论文版式识别效果差，多个中文文献正文只抽出数百字甚至更少，整体质量低于 `文本提取/outputs/pdf_extract/` 中的 PyMuPDF 结果。

因此，当前文本提取规划改为：

1. 正式抽取路线使用 PyMuPDF；
2. 不再安装、启动或依赖 GROBID；
3. 不再把 GROBID 失败作为报告中的错误原因；
4. 原始 PDF 继续作为引用、页码、图表、公式和关键表述的核验来源；
5. GPT 深度研究优先使用 `文本提取/outputs/pdf_extract/md/` 与 `文本提取/outputs/pdf_extract/jsonl/records.jsonl`。

## 任务目标

在当前仓库中维护一个可复用的批量 PDF 文本提取工具，用于从学术论文 PDF 中提取两类内容：

1. 摘要（abstract）；
2. 正文（body text）。

本任务只做文本抽取与清洗，不进行摘要改写、内容概括、语义补全或基于模型的再生成。输出文本应尽量忠实于 PDF 原文。若某篇 PDF 的摘要边界无法可靠识别，应记录为 `partial`，不得补写或猜测摘要内容。

## 实现范围

当前实现位于 `文本提取/`，不接入训练、建模、政策解析和正式 `outputs/v0.50/` 主链路。

```text
文本提取/
  extract_pdf_abstract_body.py
  pdf_abstract_body/
    __init__.py
    fallback_pymupdf.py
    text_cleaner.py
    runner.py
    tei_parser.py          # 历史实验代码，当前不推荐
    grobid_client.py       # 历史实验代码，当前不推荐
  outputs/
    pdf_extract/
      txt/
      md/
      jsonl/
      extraction_report.csv
```

## 命令行接口

当前推荐调用方式：

```powershell
python .\文本提取\extract_pdf_abstract_body.py `
  --input .\参考文献 `
  --output .\文本提取\outputs\pdf_extract `
  --mode pymupdf `
  --workers 4 `
  --overwrite
```

参数要求：

```text
--input        PDF 文件夹路径，递归读取其中所有 .pdf 文件
--output       输出目录
--mode         当前使用 pymupdf；grobid/auto 仅为历史实验选项
--workers      并发处理数量，当前可使用 4
--overwrite    可选参数，存在时覆盖已有输出
--verbose      可选参数，打印更详细日志
```

## 输入与输出要求

### 输入

输入为一个 PDF 文件夹，例如：

```text
参考文献/
  paper_001.pdf
  paper_002.pdf
```

要求递归扫描所有 `.pdf` 文件。

### 每篇论文输出

```text
文本提取/outputs/pdf_extract/txt/{safe_stem}.abstract.txt
文本提取/outputs/pdf_extract/txt/{safe_stem}.body.txt
文本提取/outputs/pdf_extract/md/{safe_stem}.md
文本提取/outputs/pdf_extract/jsonl/records.jsonl
```

其中 `{safe_stem}` 由 PDF 文件名转换得到。若不同目录下存在同名 PDF，应追加短哈希，避免覆盖。

### Markdown 输出格式

```markdown
# {title_or_filename}

## Metadata

- source_pdf: {relative_pdf_path}
- extraction_method: pymupdf
- extraction_status: {success|partial|failed}
- title: {title_if_available}
- doi: {doi_if_available}

## Abstract

{abstract_text}

## Body

{body_text}
```

如果某项元数据不存在，保留字段但写为空字符串，不通过文件名推断 DOI、标题或作者。

### JSONL 输出格式

`records.jsonl` 每行对应一篇 PDF，字段至少包括：

```json
{
  "source_pdf": "参考文献/example.pdf",
  "safe_stem": "example",
  "status": "success",
  "method": "pymupdf",
  "title": "",
  "doi": "",
  "abstract": "",
  "body": "",
  "abstract_chars": 0,
  "body_chars": 0,
  "error": ""
}
```

### 提取报告

生成 `extraction_report.csv`，字段至少包括：

```text
source_pdf,safe_stem,status,method,title,doi,abstract_chars,body_chars,tei_saved,error
```

当前有效结果中，`error` 不应记录 GROBID 不可用、GROBID 回退等历史实验信息。只有 PDF 损坏、PyMuPDF 无法打开、正文为空等真实失败才写入错误原因。

## PyMuPDF 抽取要求

1. 逐页提取文本；
2. 合并为全文；
3. 根据关键词识别摘要起止位置；
4. 根据关键词识别正文起始位置；
5. 在参考文献、References、Bibliography 等标记之前截断正文。

摘要识别关键词包括：

```text
摘要
摘 要
Abstract
ABSTRACT
```

正文起始关键词包括：

```text
引言
绪论
Introduction
INTRODUCTION
1 引言
1. Introduction
```

参考文献截断关键词包括：

```text
参考文献
References
REFERENCES
Bibliography
```

PyMuPDF 对摘要边界可能误判，因此允许 `status=partial`。只要正文可用，不能把摘要缺失视为整篇失败。

## 文本清洗规则

1. 统一换行符为 `\n`；
2. 去除连续多余空白；
3. 保留段落之间的空行；
4. 去除明显页码行，例如单独的 `1`、`- 1 -`；
5. 去除过短且无意义的孤立行；
6. 处理英文断行连字符，例如 `mar-\nket` 可合并为 `market`；
7. 不删除中文句号、英文句号、括号中的引用编号；
8. 不对正文进行改写、翻译或概括。

清洗应尽量保守，宁可保留少量噪声，也不要误删正文。

## 异常处理与日志

1. 单篇 PDF 失败不影响其他 PDF；
2. 每篇 PDF 都要在 `extraction_report.csv` 中留下记录；
3. 捕获并记录 PyMuPDF 打开失败、空正文、文件损坏等异常；
4. 若摘要为空但正文成功，状态记为 `partial`；
5. 若正文为空，状态记为 `failed` 或 `partial`，并写明原因；
6. 控制台输出进度，例如 `[12/80] success paper.pdf`。

## 依赖管理

`requirements.txt` 至少包含：

```text
pymupdf
pandas
```

不要引入 OCR。扫描版 PDF 的 OCR 可作为后续扩展，不在本次任务中实现。

## 测试要求

至少保留以下测试：

1. PyMuPDF 端到端测试：构造最小 PDF，验证能生成摘要、正文、Markdown、JSONL、CSV；
2. 文本清洗测试：验证页码、重复空格、英文断行连字符等基础清洗逻辑；
3. 安全文件名测试：同名 PDF 不覆盖；
4. 历史 TEI 解析测试可保留，用于防止旧代码损坏，但当前研究输入不采用 GROBID 输出。

## 验收标准

任务完成后，以下命令可以运行：

```powershell
python .\文本提取\extract_pdf_abstract_body.py `
  --input .\参考文献 `
  --output .\文本提取\outputs\pdf_extract `
  --mode pymupdf `
  --workers 4 `
  --overwrite
```

验收结果应满足：

1. 能递归处理输入目录下的所有 PDF；
2. 每篇 PDF 生成摘要文本、正文文本、Markdown 文件；
3. 生成统一的 `records.jsonl`；
4. 生成统一的 `extraction_report.csv`；
5. 不把参考文献当成正文；
6. 单篇失败不影响批量任务；
7. 失败原因可追踪；
8. 测试通过；
9. 报告中不再出现 GROBID 失败或回退标记；
10. 当前 `参考文献/` 的有效结果为 `53` 条记录、`failed=0`。

## 实现注意事项

1. 不使用大语言模型对正文进行重写或摘要生成；
2. 不在提取失败时用标题、文件名或参考文献信息补正文；
3. 不默认把参考文献、附录、作者简介纳入正文；
4. 不因少量页眉页脚噪声而过度清洗，正文完整性优先；
5. 不在脚本中写死用户本机绝对路径；
6. 不改变仓库既有训练、建模和数据处理主流程；
7. 新增功能应可单独运行，便于后续把论文 PDF 提取结果接入文献综述或资料库。
