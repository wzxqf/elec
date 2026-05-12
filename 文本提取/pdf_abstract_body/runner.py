"""Batch runner for PDF abstract/body extraction."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

from .fallback_pymupdf import extract_with_pymupdf
from .grobid_client import GrobidClient
from .tei_parser import ParsedPaper, parse_tei_xml


@dataclass(slots=True)
class ExtractionRecord:
    source_pdf: str
    safe_stem: str
    status: str
    method: str
    title: str
    doi: str
    abstract: str
    body: str
    abstract_chars: int
    body_chars: int
    error: str
    tei_saved: bool = False
    abstract_path: str = ""
    body_path: str = ""
    markdown_path: str = ""
    tei_path: str = ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"input directory does not exist: {input_root}", file=sys.stderr)
        return 2

    pdfs = sorted(input_root.rglob("*.pdf"))
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("tei", "txt", "md", "jsonl"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    stem_map = build_safe_stem_map(input_root, pdfs)
    client = GrobidClient(args.grobid_url, timeout=args.timeout)
    grobid_available = args.mode in {"grobid", "auto"} and client.is_available()
    if args.verbose and args.mode in {"grobid", "auto"}:
        state = "available" if grobid_available else "unavailable"
        print(f"GROBID {state}: {args.grobid_url}")

    records: list[ExtractionRecord] = []
    if not pdfs:
        write_outputs(output_root, records, args.jsonl_text_mode)
        print(f"no PDF files found under {input_root}")
        return 0

    worker_count = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                process_pdf,
                pdf_path,
                input_root,
                output_root,
                stem_map[pdf_path],
                args,
                client,
                grobid_available,
            ): pdf_path
            for pdf_path in pdfs
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            record = future.result()
            records.append(record)
            print(f"[{completed}/{len(pdfs)}] {record.status} {Path(record.source_pdf).name}")

    records.sort(key=lambda item: item.source_pdf)
    write_outputs(output_root, records, args.jsonl_text_mode)
    return 0 if all(record.status != "failed" for record in records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch extract abstracts and body text from academic PDFs."
    )
    parser.add_argument("--input", required=True, help="PDF folder; scanned recursively.")
    parser.add_argument("--output", required=True, help="Output folder for txt/md/jsonl/report.")
    parser.add_argument("--grobid-url", default="http://localhost:8070")
    parser.add_argument("--mode", choices=("grobid", "pymupdf", "auto"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-tei", dest="save_tei", action="store_true", default=True)
    parser.add_argument("--no-save-tei", dest="save_tei", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--jsonl-text-mode", choices=("full", "paths"), default="full")
    return parser


def build_safe_stem_map(input_root: Path, pdfs: list[Path]) -> dict[Path, str]:
    raw: dict[Path, str] = {pdf: safe_filename(pdf.stem) for pdf in pdfs}
    counts: dict[str, int] = {}
    for stem in raw.values():
        counts[stem] = counts.get(stem, 0) + 1
    mapped: dict[Path, str] = {}
    for pdf, stem in raw.items():
        if counts[stem] > 1:
            rel = pdf.relative_to(input_root).as_posix()
            suffix = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
            stem = f"{stem}_{suffix}"
        mapped[pdf] = stem
    return mapped


def safe_filename(value: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value)
    safe = re.sub(r"\s+", "_", safe, flags=re.UNICODE)
    safe = safe.strip("._ ")
    return safe or "paper"


def process_pdf(
    pdf_path: Path,
    input_root: Path,
    output_root: Path,
    safe_stem: str,
    args: argparse.Namespace,
    client: GrobidClient,
    grobid_available: bool,
) -> ExtractionRecord:
    paths = output_paths(output_root, safe_stem)
    source_pdf = source_path(pdf_path)

    if not args.overwrite and paths["abstract"].exists() and paths["body"].exists() and paths["md"].exists():
        abstract = paths["abstract"].read_text(encoding="utf-8", errors="replace")
        body = paths["body"].read_text(encoding="utf-8", errors="replace")
        status = classify_status(ParsedPaper(abstract=abstract, body=body))
        return make_record(
            source_pdf=source_pdf,
            safe_stem=safe_stem,
            status=status,
            method="cached",
            parsed=ParsedPaper(abstract=abstract, body=body),
            paths=paths,
            error="existing extraction reused; pass --overwrite to refresh",
            tei_saved=paths["tei"].exists(),
        )

    errors: list[str] = []
    parsed = ParsedPaper()
    method = ""
    tei_saved = False

    if args.mode in {"grobid", "auto"}:
        if grobid_available:
            grobid_result = client.process_fulltext_document(pdf_path)
            if grobid_result.ok:
                try:
                    parsed = parse_tei_xml(grobid_result.tei_xml)
                    method = "grobid"
                    if args.save_tei:
                        paths["tei"].write_text(grobid_result.tei_xml, encoding="utf-8")
                        tei_saved = True
                    if not parsed.body:
                        errors.append("grobid parsed empty body")
                except Exception as exc:
                    errors.append(f"grobid TEI parse failed: {exc}")
                    parsed = ParsedPaper()
            else:
                errors.append(grobid_result.error)
        else:
            errors.append(f"grobid service unavailable: {args.grobid_url}")

    should_fallback = args.mode == "pymupdf" or (args.mode == "auto" and not parsed.body)
    if should_fallback:
        try:
            parsed = extract_with_pymupdf(pdf_path)
            method = "pymupdf"
        except Exception as exc:
            errors.append(str(exc))
            if not method:
                method = "pymupdf"

    status = classify_status(parsed)
    if status == "failed" and not errors:
        errors.append("empty body")

    write_paper_files(paths, source_pdf, method, status, parsed)
    return make_record(
        source_pdf=source_pdf,
        safe_stem=safe_stem,
        status=status,
        method=method or args.mode,
        parsed=parsed,
        paths=paths,
        error="; ".join(error for error in errors if error),
        tei_saved=tei_saved,
    )


def classify_status(parsed: ParsedPaper) -> str:
    if parsed.abstract and parsed.body:
        return "success"
    if parsed.abstract or parsed.body:
        return "partial"
    return "failed"


def output_paths(output_root: Path, safe_stem: str) -> dict[str, Path]:
    return {
        "abstract": output_root / "txt" / f"{safe_stem}.abstract.txt",
        "body": output_root / "txt" / f"{safe_stem}.body.txt",
        "md": output_root / "md" / f"{safe_stem}.md",
        "tei": output_root / "tei" / f"{safe_stem}.tei.xml",
    }


def write_paper_files(
    paths: dict[str, Path],
    source_pdf: str,
    method: str,
    status: str,
    parsed: ParsedPaper,
) -> None:
    paths["abstract"].write_text(parsed.abstract, encoding="utf-8")
    paths["body"].write_text(parsed.body, encoding="utf-8")
    markdown = build_markdown(source_pdf, method, status, parsed)
    paths["md"].write_text(markdown, encoding="utf-8")


def build_markdown(source_pdf: str, method: str, status: str, parsed: ParsedPaper) -> str:
    title = parsed.title or Path(source_pdf).stem
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Metadata",
            "",
            f"- source_pdf: {source_pdf}",
            f"- extraction_method: {method}",
            f"- extraction_status: {status}",
            f"- title: {parsed.title}",
            f"- doi: {parsed.doi}",
            "",
            "## Abstract",
            "",
            parsed.abstract,
            "",
            "## Body",
            "",
            parsed.body,
            "",
        ]
    )


def make_record(
    *,
    source_pdf: str,
    safe_stem: str,
    status: str,
    method: str,
    parsed: ParsedPaper,
    paths: dict[str, Path],
    error: str,
    tei_saved: bool,
) -> ExtractionRecord:
    return ExtractionRecord(
        source_pdf=source_pdf,
        safe_stem=safe_stem,
        status=status,
        method=method,
        title=parsed.title,
        doi=parsed.doi,
        abstract=parsed.abstract,
        body=parsed.body,
        abstract_chars=len(parsed.abstract),
        body_chars=len(parsed.body),
        error=error,
        tei_saved=tei_saved,
        abstract_path=source_path(paths["abstract"]),
        body_path=source_path(paths["body"]),
        markdown_path=source_path(paths["md"]),
        tei_path=source_path(paths["tei"]) if paths["tei"].exists() else "",
    )


def write_outputs(output_root: Path, records: list[ExtractionRecord], text_mode: str) -> None:
    jsonl_path = output_root / "jsonl" / "records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            if text_mode == "paths":
                payload["abstract"] = ""
                payload["body"] = ""
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    report_path = output_root / "extraction_report.csv"
    report_fields = [
        "source_pdf",
        "safe_stem",
        "status",
        "method",
        "title",
        "doi",
        "abstract_chars",
        "body_chars",
        "tei_saved",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in report_fields})


def source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
