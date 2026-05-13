from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parents[1] / "文本提取"
sys.path.insert(0, str(MODULE_ROOT))

from pdf_abstract_body.runner import build_safe_stem_map, safe_filename
from pdf_abstract_body.runner import main as extraction_main
from pdf_abstract_body.tei_parser import parse_tei_xml
from pdf_abstract_body.text_cleaner import clean_text


def test_v051_tei_extracts_abstract_paragraphs_in_order():
    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt><title>Sample Paper</title></titleStmt>
          <publicationStmt><p/></publicationStmt>
          <sourceDesc><p/></sourceDesc>
        </fileDesc>
        <profileDesc>
          <abstract>
            <p>abstract paragraph one</p>
            <p>abstract paragraph two</p>
          </abstract>
        </profileDesc>
      </teiHeader>
    </TEI>
    """

    parsed = parse_tei_xml(xml)

    assert parsed.title == "Sample Paper"
    assert parsed.abstract == "abstract paragraph one\n\nabstract paragraph two"


def test_v051_tei_extracts_body_titles_and_paragraphs():
    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
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
      </text>
    </TEI>
    """

    parsed = parse_tei_xml(xml)

    assert "### Introduction" in parsed.body
    assert "body paragraph one" in parsed.body
    assert "### Method" in parsed.body
    assert "body paragraph two" in parsed.body


def test_v051_tei_excludes_back_matter_and_bibliography():
    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <div>
            <head>Results</head>
            <p>main body paragraph</p>
            <listBibl><bibl>body embedded bibliography noise</bibl></listBibl>
          </div>
        </body>
        <back>
          <div><p>reference paragraph that must stay out</p></div>
        </back>
      </text>
    </TEI>
    """

    parsed = parse_tei_xml(xml)

    assert "main body paragraph" in parsed.body
    assert "body embedded bibliography noise" not in parsed.body
    assert "reference paragraph that must stay out" not in parsed.body


def test_v051_clean_text_keeps_content_and_removes_basic_noise():
    raw = "  mar-\nket   risk  \n\n- 1 -\n\n正文  内容\n.\n"

    cleaned = clean_text(raw)

    assert "market risk" in cleaned
    assert "- 1 -" not in cleaned
    assert "正文 内容" in cleaned
    assert "\n.\n" not in f"\n{cleaned}\n"


def test_v051_safe_stems_disambiguate_duplicate_pdf_names(tmp_path):
    input_root = tmp_path / "papers"
    first = input_root / "a" / "same name.pdf"
    second = input_root / "b" / "same name.pdf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"%PDF-1.4")
    second.write_bytes(b"%PDF-1.4")

    mapped = build_safe_stem_map(input_root, [first, second])

    assert safe_filename("same name") == "same_name"
    assert mapped[first] != mapped[second]
    assert mapped[first].startswith("same_name_")
    assert mapped[second].startswith("same_name_")


def test_v051_cli_pymupdf_mode_writes_records_and_report(tmp_path):
    import fitz

    input_root = tmp_path / "papers"
    output_root = tmp_path / "out"
    input_root.mkdir()
    pdf_path = input_root / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Abstract:\nThis paper studies market risk.\n1. Introduction\nBody paragraph one.\nReferences\n[1] ref",
        fontsize=11,
    )
    document.save(pdf_path)
    document.close()

    code = extraction_main(
        [
            "--input",
            str(input_root),
            "--output",
            str(output_root),
            "--mode",
            "pymupdf",
        ]
    )

    assert code == 0
    assert (output_root / "txt" / "paper.abstract.txt").exists()
    assert (output_root / "txt" / "paper.body.txt").exists()
    assert (output_root / "md" / "paper.md").exists()
    records = (output_root / "jsonl" / "records.jsonl").read_text(encoding="utf-8")
    report = (output_root / "extraction_report.csv").read_text(encoding="utf-8-sig")
    assert '"method": "pymupdf"' in records
    assert "paper,success,pymupdf" in report
