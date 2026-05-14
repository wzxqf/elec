from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

ET.register_namespace(
    "wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
)
ET.register_namespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture")
ET.register_namespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace("w14", "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace("w15", "http://schemas.microsoft.com/office/word/2012/wordml")


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


@dataclass
class Text:
    value: str


@dataclass
class Fraction:
    numerator: list
    denominator: list


@dataclass
class Script:
    base: object
    sub: list | None = None
    sup: list | None = None


GREEK_AND_SYMBOLS = {
    "Delta": "Δ",
    "Phi": "Φ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "lambda": "λ",
    "mu": "μ",
    "omega": "ω",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "theta": "θ",
    "eta": "η",
    "epsilon": "ε",
    "cdot": "·",
    "ge": "≥",
    "le": "≤",
    "in": "∈",
    "ldots": "…",
    "mid": "|",
    "sum": "∑",
}

WORD_OPERATORS = {
    "max",
    "min",
    "tanh",
    "softmax",
    "ReLU",
    "Proj",
    "Proj_hour",
    "clip",
    "E",
}

MATHCAL = {
    "B": "𝓑",
    "F": "𝓕",
}


class Parser:
    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def parse(self, stop: str | None = None) -> list:
        nodes: list = []
        while self.i < len(self.text):
            ch = self.text[self.i]
            if stop and ch == stop:
                self.i += 1
                break
            if ch == "\\":
                node = self.parse_command()
                if node is None:
                    continue
                nodes.append(node)
                continue
            if ch in "_^":
                self.apply_script(nodes, ch)
                continue
            if ch == "{":
                self.i += 1
                nodes.extend(self.parse("}"))
                continue
            if ch == "}":
                if stop == "}":
                    self.i += 1
                break
            if ch.isalpha():
                nodes.append(Text(self.consume_identifier()))
                continue
            nodes.append(Text(ch))
            self.i += 1
        return self.merge_text(nodes)

    def consume_identifier(self) -> str:
        start = self.i
        while self.i < len(self.text) and (
            self.text[self.i].isalpha() or self.text[self.i].isdigit()
        ):
            self.i += 1
        return self.text[start : self.i]

    def consume_command_name(self) -> str:
        self.i += 1
        if self.i < len(self.text) and not self.text[self.i].isalpha():
            ch = self.text[self.i]
            self.i += 1
            return ch
        start = self.i
        while self.i < len(self.text) and self.text[self.i].isalpha():
            self.i += 1
        return self.text[start : self.i]

    def parse_command(self):
        name = self.consume_command_name()
        if name in {"left", "right"}:
            return None
        if name == "quad":
            return Text("  ")
        if name == "{":
            return Text("{")
        if name == "}":
            return Text("}")
        if name == "frac":
            return Fraction(self.read_group(), self.read_group())
        if name == "operatorname":
            value = "".join(flatten_text(self.read_group()))
            return Text(value)
        if name == "mathcal":
            value = "".join(flatten_text(self.read_group()))
            return Text(MATHCAL.get(value, value))
        if name == "widehat":
            group = self.read_group()
            value = "".join(flatten_text(group))
            return Text(value + "\u0302")
        if name in GREEK_AND_SYMBOLS:
            return Text(GREEK_AND_SYMBOLS[name])
        if name in WORD_OPERATORS:
            return Text(name)
        return Text(name)

    def read_group(self) -> list:
        self.skip_spaces()
        if self.i < len(self.text) and self.text[self.i] == "{":
            self.i += 1
            return self.parse("}")
        if self.i < len(self.text):
            if self.text[self.i] == "\\":
                node = self.parse_command()
                return [] if node is None else [node]
            ch = self.text[self.i]
            if ch.isalpha():
                return [Text(self.consume_identifier())]
            self.i += 1
            return [Text(ch)]
        return []

    def skip_spaces(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def apply_script(self, nodes: list, marker: str) -> None:
        self.i += 1
        script_nodes = self.read_group()
        if not nodes:
            nodes.append(Text(marker))
            nodes.extend(script_nodes)
            return
        base = nodes.pop()
        if isinstance(base, Script):
            if marker == "_":
                base.sub = script_nodes
            else:
                base.sup = script_nodes
            nodes.append(base)
            return
        if marker == "_":
            nodes.append(Script(base=base, sub=script_nodes))
        else:
            nodes.append(Script(base=base, sup=script_nodes))

    @staticmethod
    def merge_text(nodes: list) -> list:
        merged: list = []
        for node in nodes:
            if isinstance(node, Text) and merged and isinstance(merged[-1], Text):
                merged[-1].value += node.value
            else:
                merged.append(node)
        return merged


def flatten_text(nodes: list) -> list[str]:
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, Text):
            parts.append(node.value)
        elif isinstance(node, Fraction):
            parts.extend(flatten_text(node.numerator))
            parts.append("/")
            parts.extend(flatten_text(node.denominator))
        elif isinstance(node, Script):
            parts.extend(flatten_text([node.base]))
            if node.sub:
                parts.append("_")
                parts.extend(flatten_text(node.sub))
            if node.sup:
                parts.append("^")
                parts.extend(flatten_text(node.sup))
    return parts


def m_text(value: str) -> ET.Element:
    run = ET.Element(qn("m", "r"))
    text = ET.SubElement(run, qn("m", "t"))
    if value.startswith(" ") or value.endswith(" "):
        text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text.text = value
    return run


def append_nodes(parent: ET.Element, nodes: list) -> None:
    for node in nodes:
        parent.append(to_omml(node))


def to_omml(node) -> ET.Element:
    if isinstance(node, Text):
        return m_text(node.value)
    if isinstance(node, Fraction):
        frac = ET.Element(qn("m", "f"))
        num = ET.SubElement(frac, qn("m", "num"))
        append_nodes(num, node.numerator)
        den = ET.SubElement(frac, qn("m", "den"))
        append_nodes(den, node.denominator)
        return frac
    if isinstance(node, Script):
        if node.sub is not None and node.sup is not None:
            wrapper = ET.Element(qn("m", "sSubSup"))
            base_tag = qn("m", "e")
            sub_tag = qn("m", "sub")
            sup_tag = qn("m", "sup")
        elif node.sub is not None:
            wrapper = ET.Element(qn("m", "sSub"))
            base_tag = qn("m", "e")
            sub_tag = qn("m", "sub")
            sup_tag = None
        else:
            wrapper = ET.Element(qn("m", "sSup"))
            base_tag = qn("m", "e")
            sub_tag = None
            sup_tag = qn("m", "sup")
        base = ET.SubElement(wrapper, base_tag)
        base.append(to_omml(node.base))
        if node.sub is not None and sub_tag is not None:
            sub = ET.SubElement(wrapper, sub_tag)
            append_nodes(sub, node.sub)
        if node.sup is not None and sup_tag is not None:
            sup = ET.SubElement(wrapper, sup_tag)
            append_nodes(sup, node.sup)
        return wrapper
    return m_text(str(node))


def formula_text(omath: ET.Element) -> str:
    return "".join(t.text or "" for t in omath.findall(".//m:t", NS)).strip()


def rewrite_omath(omath: ET.Element) -> tuple[str, str]:
    raw = formula_text(omath)
    nodes = Parser(raw).parse()
    for child in list(omath):
        omath.remove(child)
    append_nodes(omath, nodes)
    return raw, "".join(flatten_text(nodes))


def rewrite_docx(input_path: Path, output_path: Path, report_path: Path | None) -> dict:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if input_path.resolve() != output_path.resolve():
        shutil.copyfile(input_path, temp_path)
    else:
        shutil.copyfile(input_path, temp_path)

    changed: list[dict] = []
    with zipfile.ZipFile(input_path, "r") as zin:
        document_xml = zin.read("word/document.xml")
        root = ET.fromstring(document_xml)
        for index, omath in enumerate(root.findall(".//m:oMath", NS), start=1):
            raw, rendered = rewrite_omath(omath)
            changed.append({"index": index, "raw": raw, "rendered_text": rendered})
        new_document = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = new_document
                zout.writestr(item, data)

    # Copy is more reliable than os.replace on this Windows host when a DOCX
    # path has recently been touched by Word automation.
    shutil.copyfile(temp_path, output_path)
    try:
        temp_path.unlink()
    except OSError:
        pass
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "converted_omath": len(changed),
        "formulas": changed,
    }
    if report_path:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = rewrite_docx(args.input, args.output, args.report)
    print(json.dumps({k: v for k, v in report.items() if k != "formulas"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
