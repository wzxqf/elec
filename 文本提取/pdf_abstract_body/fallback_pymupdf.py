"""PyMuPDF fallback extraction for PDFs that GROBID cannot parse."""

from __future__ import annotations

from pathlib import Path
import re

from .tei_parser import ParsedPaper
from .text_cleaner import clean_text

_ABSTRACT_RE = re.compile(r"(?im)^\s*(摘要|摘\s*要|abstract)\s*[:：]?\s*$")
_ABSTRACT_INLINE_RE = re.compile(r"(?im)^\s*(摘要|摘\s*要|abstract)\s*[:：]\s*(.+)$")
_BODY_START_RE = re.compile(
    r"(?im)^\s*((?:1(?:\.|\s|、))?\s*(引言|绪论)|(?:1\.\s*)?introduction)\s*$"
)
_REF_RE = re.compile(r"(?im)^\s*(参考文献|references|bibliography)\s*$")
_ABSTRACT_END_RE = re.compile(
    r"(?im)^\s*(关键词|关键字|key\s*words?|"
    r"(?:1(?:\.|\s|、))?\s*(引言|绪论)|(?:1\.\s*)?introduction)\b"
)


def extract_with_pymupdf(pdf_path: Path) -> ParsedPaper:
    """Extract approximate abstract and body text with PyMuPDF."""

    try:
        import fitz
    except Exception as exc:
        raise RuntimeError(f"pymupdf unavailable: {exc}") from exc

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"cannot open PDF with pymupdf: {exc}") from exc

    try:
        pages = [page.get_text("text") for page in document]
    finally:
        document.close()

    full_text = clean_text("\n\n".join(pages))
    if not full_text:
        return ParsedPaper()

    abstract = _slice_abstract(full_text)
    body = _slice_body(full_text)
    return ParsedPaper(abstract=abstract, body=body)


def _slice_abstract(text: str) -> str:
    inline_match = _ABSTRACT_INLINE_RE.search(text)
    if inline_match:
        start = inline_match.start(2)
    else:
        marker = _ABSTRACT_RE.search(text)
        if not marker:
            return ""
        start = marker.end()

    end_match = _ABSTRACT_END_RE.search(text, start)
    end = end_match.start() if end_match else min(len(text), start + 3000)
    return clean_text(text[start:end])


def _slice_body(text: str) -> str:
    start_match = _BODY_START_RE.search(text)
    start = start_match.start() if start_match else 0
    end_match = _REF_RE.search(text, start)
    end = end_match.start() if end_match else len(text)
    return clean_text(text[start:end])
