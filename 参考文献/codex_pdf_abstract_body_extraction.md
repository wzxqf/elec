# Codex 任务说明：批量提取论文 PDF 的摘要与正文文本

## 任务目标

请在当前仓库中实现一个可复用的批量 PDF 文本提取工具，用于从学术论文 PDF 中提取两类内容：

1. 摘要（abstract）；
2. 正文（body text）。

本任务只做“文本抽取与清洗”，不得进行摘要改写、内容概括、语义补全或基于模型的再生成。输出文本必须尽量忠实于 PDF 原文。若某篇 PDF 无法可靠提取，应记录失败原因，而不是编造内容。

建议优先采用 GROBID 解析论文 PDF，并从其 TEI XML 中抽取摘要与正文。普通 PDF 文本提取工具可作为后备方案，但不作为首选方案。

---

## 一、实现范围

请新增一个独立工具模块和命令行入口，用于批量处理文件夹中的 PDF。

推荐目录结构如下：

```text
scripts/
  extract_pdf_abstract_body.py

src/
  pdf_extract/
    __init__.py
    grobid_client.py
    tei_parser.py
    text_cleaner.py
    fallback_pymupdf.py

outputs/
  pdf_extract/
    tei/
    txt/
    md/
    jsonl/
    extraction_report.csv
```

如果当前项目已有更合适的工具目录，可以按现有结构调整，但需要保持功能边界清晰，不要把该功能混入训练、建模或政策解析主链路中。

---

## 二、命令行接口要求

新增脚本 `scripts/extract_pdf_abstract_body.py`，支持如下调用方式：

```bash
python scripts/extract_pdf_abstract_body.py \
  --input papers \
  --output outputs/pdf_extract \
  --grobid-url http://localhost:8070 \
  --mode auto \
  --workers 4
```

参数要求：

```text
--input        PDF 文件夹路径，递归读取其中所有 .pdf 文件
--output       输出目录
--grobid-url   GROBID 服务地址，默认 http://localhost:8070
--mode         grobid | pymupdf | auto，默认 auto
--workers      并发处理数量，默认 2 或 4，避免过高并发压垮本地 GROBID 服务
--overwrite    可选参数，存在时覆盖已有输出
--save-tei     可选参数，保存 GROBID 返回的 TEI XML，默认保存
--verbose      可选参数，打印更详细日志
```

`auto` 模式下，优先调用 GROBID。若 GROBID 服务不可用或单篇 PDF 解析失败，再尝试 PyMuPDF 后备提取，并在报告中标记该篇文献使用了 fallback。

---

## 三、输入与输出要求

### 1. 输入

输入为一个 PDF 文件夹，例如：

```text
papers/
  paper_001.pdf
  paper_002.pdf
  subdir/
    paper_003.pdf
```

要求递归扫描所有 `.pdf` 文件。

### 2. 每篇论文的输出

对于每个 PDF，输出以下文件：

```text
outputs/pdf_extract/txt/{safe_stem}.abstract.txt
outputs/pdf_extract/txt/{safe_stem}.body.txt
outputs/pdf_extract/md/{safe_stem}.md
outputs/pdf_extract/jsonl/records.jsonl
outputs/pdf_extract/tei/{safe_stem}.tei.xml
```

其中 `{safe_stem}` 应由 PDF 文件名转换得到，去除或替换不适合文件名的特殊字符。若不同目录下存在同名 PDF，应避免覆盖，可在文件名后追加短哈希。

### 3. Markdown 输出格式

每篇论文的 `.md` 文件应采用如下格式：

```markdown
# {title_or_filename}

## Metadata

- source_pdf: {relative_pdf_path}
- extraction_method: {grobid|pymupdf}
- extraction_status: {success|partial|failed}
- title: {title_if_available}
- doi: {doi_if_available}

## Abstract

{abstract_text}

## Body

{body_text}
```

如果某项元数据不存在，保留字段但写为空字符串，不要猜测。

### 4. JSONL 输出格式

`records.jsonl` 每行对应一篇 PDF，字段至少包括：

```json
{
  "source_pdf": "papers/example.pdf",
  "safe_stem": "example",
  "status": "success",
  "method": "grobid",
  "title": "",
  "doi": "",
  "abstract": "",
  "body": "",
  "abstract_chars": 0,
  "body_chars": 0,
  "error": ""
}
```

若正文过长，也可以额外提供 `--jsonl-text-mode full|paths` 参数。默认可使用 `full`，即 JSONL 中保存全文；若实现较复杂，可先只保存全文，不强制实现该扩展参数。

### 5. 提取报告

生成 `extraction_report.csv`，字段至少包括：

```text
source_pdf,safe_stem,status,method,title,doi,abstract_chars,body_chars,tei_saved,error
```

报告用于快速检查哪些 PDF 成功、哪些失败、哪些只提取到了部分内容。

---

## 四、GROBID 解析要求

### 1. 调用方式

通过 REST API 调用：

```text
POST {grobid-url}/api/processFulltextDocument
```

请求中传入 PDF 文件，建议参数：

```text
consolidateHeader=0
consolidateCitations=0
includeRawCitations=0
includeRawAffiliations=0
```

需要设置超时时间，例如 120 秒。单篇失败不能中断整个批处理流程。

### 2. TEI XML 中的摘要抽取

从 GROBID 返回的 TEI XML 中优先抽取：

```text
//tei:profileDesc/tei:abstract
```

其中段落文本来自：

```text
.//tei:p
```

如果不存在 `<p>`，则使用 abstract 节点下的全部文本。多个段落之间保留空行。

### 3. TEI XML 中的正文抽取

正文从以下节点抽取：

```text
//tei:text/tei:body
```

正文抽取规则：

1. 保留正文中的章节标题 `<head>`；
2. 保留正文段落 `<p>`；
3. 保留段落顺序；
4. 不抽取 `<back>`、`<listBibl>`、参考文献部分；
5. 不抽取作者、单位、期刊信息等头部元数据；
6. 图题、表题、公式、脚注可以暂不纳入正文，除非它们已经自然出现在正文段落中。

建议将章节标题写成 Markdown 二级或三级标题，例如：

```markdown
### Introduction

正文段落……
```

### 4. 元数据抽取

在能力范围内抽取以下元数据：

```text
title: //tei:titleStmt/tei:title
abstract: //tei:profileDesc/tei:abstract
doi: //tei:idno[@type="DOI"]
```

只抽取明确存在的内容，不要通过文件名推断 DOI、标题或作者。

---

## 五、PyMuPDF 后备方案要求

当 GROBID 不可用或单篇解析失败时，可以使用 PyMuPDF 后备提取。后备方案只需提供尽量可用的摘要与正文，不要求达到 GROBID 的结构化效果。

后备方案建议流程：

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

注意：后备方案容易误判，因此必须在 `extraction_report.csv` 中标记 `method=pymupdf`，并允许 `status=partial`。

---

## 六、文本清洗规则

请实现 `text_cleaner.py`，至少包含以下清洗逻辑：

1. 统一换行符为 `\n`；
2. 去除连续多余空白；
3. 保留段落之间的空行；
4. 去除明显页码行，例如单独的 `1`、`- 1 -`；
5. 去除过短且无意义的孤立行；
6. 处理英文断行连字符，例如 `mar-\nket` 可合并为 `market`；
7. 不要删除中文句号、英文句号、括号中的引用编号；
8. 不要对正文进行改写、翻译或概括。

清洗应尽量保守，宁可保留少量噪声，也不要误删正文。

---

## 七、异常处理与日志

批量处理时必须满足：

1. 单篇 PDF 失败不影响其他 PDF；
2. 每篇 PDF 都要在 `extraction_report.csv` 中留下记录；
3. 捕获并记录 HTTP 错误、超时、XML 解析错误、空正文、文件损坏等异常；
4. 若摘要为空但正文成功，状态记为 `partial`；
5. 若正文为空，状态记为 `failed` 或 `partial`，并写明原因；
6. 控制台输出进度，例如 `[12/80] success paper.pdf`。

---

## 八、依赖管理

请根据项目现有依赖管理方式添加依赖。若项目使用 `requirements.txt`，添加：

```text
requests
lxml
pymupdf
pandas
```

如果项目使用 `pyproject.toml`，则添加到对应依赖区。

不要引入过重依赖。不要默认引入 OCR。扫描版 PDF 的 OCR 可作为后续扩展，不在本次任务中实现。

---

## 九、测试要求

至少新增以下测试，若当前项目没有测试框架，可以新增简单的 `pytest` 测试。

### 1. TEI 摘要抽取测试

构造一个最小 TEI XML，包含：

```xml
<profileDesc>
  <abstract>
    <p>abstract paragraph one</p>
    <p>abstract paragraph two</p>
  </abstract>
</profileDesc>
```

验证输出包含两个摘要段落，且段落顺序正确。

### 2. TEI 正文抽取测试

构造一个最小 TEI XML，包含：

```xml
<body>
  <div>
    <head>Introduction</head>
    <p>body paragraph one</p>
  </div>
  <div>
    <head>Method</head>
    <p>body paragraph two</p>
  </div>
</body>
```

验证正文包含章节标题和两个正文段落。

### 3. 参考文献排除测试

构造 TEI XML，确保 `<back>` 或 `<listBibl>` 中的参考文献不会进入正文输出。

### 4. 清洗函数测试

验证页码、重复空格、英文断行连字符等基础清洗逻辑。

---

## 十、验收标准

任务完成后，请确保以下命令可以运行：

```bash
python scripts/extract_pdf_abstract_body.py \
  --input papers \
  --output outputs/pdf_extract \
  --mode auto
```

验收结果应满足：

1. 能递归处理输入目录下的所有 PDF；
2. 每篇 PDF 生成摘要文本、正文文本、Markdown 文件；
3. 生成统一的 `records.jsonl`；
4. 生成统一的 `extraction_report.csv`；
5. GROBID 成功时优先使用 TEI XML 的 `<abstract>` 与 `<body>`；
6. 不把参考文献当成正文；
7. 单篇失败不影响批量任务；
8. 失败原因可追踪；
9. 测试通过；
10. README 或脚本注释中说明启动 GROBID 的基本方式。

---

## 十一、README 补充内容

请在 README 或单独文档中补充简短说明：

```markdown
### 批量提取论文 PDF 摘要与正文

1. 启动 GROBID：

```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

2. 运行批量提取：

```bash
python scripts/extract_pdf_abstract_body.py \
  --input papers \
  --output outputs/pdf_extract \
  --mode auto
```

3. 查看结果：

```text
outputs/pdf_extract/txt/
outputs/pdf_extract/md/
outputs/pdf_extract/jsonl/records.jsonl
outputs/pdf_extract/extraction_report.csv
```
```

若 Docker 镜像版本需要调整，请以本地可拉取版本为准，不要把版本号硬编码到核心逻辑中。

---

## 十二、实现注意事项

1. 不要使用大语言模型对正文进行重写或摘要生成。
2. 不要在提取失败时用标题、文件名或参考文献信息补正文。
3. 不要默认把参考文献、附录、作者简介纳入正文。
4. 不要因为少量页眉页脚噪声而过度清洗，正文完整性优先。
5. 不要在脚本中写死用户本机绝对路径。
6. 不要改变仓库既有训练、建模和数据处理主流程。
7. 新增功能应可单独运行，便于后续把论文 PDF 提取结果接入文献综述或资料库。

---

## 十三、建议实现顺序

1. 先实现 `tei_parser.py`，完成 TEI XML 中 title、doi、abstract、body 的解析；
2. 再实现 `grobid_client.py`，负责调用 GROBID 并保存 TEI；
3. 实现 `text_cleaner.py`，提供保守清洗函数；
4. 实现 `fallback_pymupdf.py`，作为后备方案；
5. 实现命令行脚本，串联批量扫描、解析、输出和报告；
6. 添加最小单元测试；
7. 用少量 PDF 手动验证摘要、正文和参考文献边界。

最终提交时，请给出简要变更说明，并列出新增文件、运行命令和测试命令。
