"""Parse GROBID TEI XML into title, DOI, abstract, and body text."""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

from .text_cleaner import clean_joined_blocks, clean_text

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_SKIP_BODY_NODES = {"back", "listBibl", "figure", "table", "formula", "note"}


@dataclass(slots=True)
class ParsedPaper:
    title: str = ""
    doi: str = ""
    abstract: str = ""
    body: str = ""


def parse_tei_xml(xml_text: str) -> ParsedPaper:
    """Parse a TEI XML document returned by GROBID."""

    root = ET.fromstring(xml_text)
    title = _node_text(_find_first(root, ".//tei:titleStmt/tei:title", ".//titleStmt/title"))
    doi = _find_doi(root)
    abstract = _extract_abstract(root)
    body = _extract_body(root)
    return ParsedPaper(title=title, doi=doi, abstract=abstract, body=body)


def _find_doi(root: ET.Element) -> str:
    for node in _find_all(root, ".//tei:idno", ".//idno"):
        if node.attrib.get("type", "").lower() == "doi":
            return clean_text(" ".join(node.itertext()))
    return ""


def _extract_abstract(root: ET.Element) -> str:
    abstract_node = _find_first(root, ".//tei:profileDesc/tei:abstract", ".//profileDesc/abstract")
    if abstract_node is None:
        return ""

    paragraphs = [_node_text(node) for node in _find_all(abstract_node, ".//tei:p", ".//p")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if paragraphs:
        return clean_joined_blocks(paragraphs)
    return clean_text(" ".join(abstract_node.itertext()))


def _extract_body(root: ET.Element) -> str:
    body_node = _find_first(root, ".//tei:text/tei:body", ".//text/body")
    if body_node is None:
        return ""

    blocks: list[str] = []
    for child in list(body_node):
        _walk_body(child, blocks)
    return clean_joined_blocks(blocks)


def _walk_body(node: ET.Element, blocks: list[str]) -> None:
    name = _local_name(node.tag)
    if name in _SKIP_BODY_NODES:
        return
    if name == "head":
        text = _node_text(node)
        if text:
            blocks.append(f"### {text}")
        return
    if name == "p":
        text = _node_text(node)
        if text:
            blocks.append(text)
        return
    for child in list(node):
        _walk_body(child, blocks)


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return clean_text(" ".join(node.itertext()))


def _find_first(root: ET.Element, namespaced_path: str, plain_path: str) -> ET.Element | None:
    node = root.find(namespaced_path, TEI_NS)
    if node is not None:
        return node
    return root.find(plain_path)


def _find_all(root: ET.Element, namespaced_path: str, plain_path: str) -> list[ET.Element]:
    nodes = root.findall(namespaced_path, TEI_NS)
    if nodes:
        return nodes
    return root.findall(plain_path)


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag
