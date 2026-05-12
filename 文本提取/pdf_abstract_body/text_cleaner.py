"""Conservative text cleaning for extracted academic PDF text."""

from __future__ import annotations

import re

_PAGE_NUMBER_RE = re.compile(
    r"^(?:[-–—]?\s*\d{1,4}\s*[-–—]?|第\s*\d{1,4}\s*页|page\s+\d{1,4})$",
    re.IGNORECASE,
)
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def clean_text(text: str, *, drop_short_noise: bool = True) -> str:
    """Clean extracted text without rewriting its meaning.

    The function keeps paragraph breaks, removes obvious page-number lines, and
    merges English words split by line-break hyphenation.
    """

    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", normalized)
    normalized = normalized.replace("\u00a0", " ")

    cleaned_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if _PAGE_NUMBER_RE.match(line):
            continue
        if drop_short_noise and _is_short_noise(line):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_joined_blocks(blocks: list[str], *, separator: str = "\n\n") -> str:
    """Clean individual blocks and join non-empty results."""

    cleaned_blocks = [clean_text(block) for block in blocks]
    return separator.join(block for block in cleaned_blocks if block)


def _is_short_noise(line: str) -> bool:
    if len(line) == 1 and _PUNCT_ONLY_RE.match(line):
        return True
    if len(line) <= 2 and _PUNCT_ONLY_RE.match(line):
        return True
    return False
