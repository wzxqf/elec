from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass
class PolicyParseResult:
    inventory: pd.DataFrame
    rule_table: pd.DataFrame
    failures: pd.DataFrame


def _list_policy_files(policy_dir: str | Path) -> list[Path]:
    root = Path(policy_dir)
    files: list[Path] = []
    for suffix in (".docx", ".xlsx", ".xls", ".pdf", ".doc"):
        files.extend(sorted(root.rglob(f"*{suffix}")))
    return sorted(set(files))


def _extract_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    texts = [node.text.strip() for node in root.findall(".//w:t", WORD_NS) if node.text and node.text.strip()]
    return "".join(texts)


def _extract_xlsx_summary(path: Path) -> dict[str, Any]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=True)
    worksheet = workbook.worksheets[0]
    rows = list(worksheet.iter_rows(min_row=1, max_row=200, values_only=True))
    frame = pd.DataFrame(rows[2:], columns=rows[1]).dropna(how="all")
    summary = {
        "sheet_names": "|".join(workbook.sheetnames),
        "mechanism_price_floor": None,
        "mechanism_price_ceiling": None,
        "mechanism_volume_ratio_max": None,
        "mechanism_exec_term_years": None,
        "table_preview": frame.head(10).to_dict(orient="records"),
    }
    if "机制电价（元/千瓦时)" in frame.columns:
        prices = pd.to_numeric(frame["机制电价（元/千瓦时)"], errors="coerce").dropna()
        if not prices.empty:
            summary["mechanism_price_floor"] = float(prices.min())
            summary["mechanism_price_ceiling"] = float(prices.max())
    if "机制电量比例(%)" in frame.columns:
        ratios = pd.to_numeric(frame["机制电量比例(%)"], errors="coerce").dropna()
        if not ratios.empty:
            summary["mechanism_volume_ratio_max"] = float(ratios.max()) / 100.0
    if "机制电价执行期限(年)" in frame.columns:
        years = pd.to_numeric(frame["机制电价执行期限(年)"], errors="coerce").dropna()
        if not years.empty:
            summary["mechanism_exec_term_years"] = float(years.max())
    return summary


def _parse_filename_dates(name: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    matches = re.findall(r"(20\d{2})[-.年](\d{1,2})[-.月](\d{1,2})", name)
    if not matches:
        return None, None
    timestamps = [pd.Timestamp(year=int(y), month=int(m), day=int(d)) for y, m, d in matches]
    if len(timestamps) == 1:
        return timestamps[0], timestamps[0]
    return timestamps[0], timestamps[1]


def _extract_notice_date(text: str) -> pd.Timestamp | None:
    matches = re.findall(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text)
    if not matches:
        return None
    year, month, day = map(int, matches[-1])
    return pd.Timestamp(year=year, month=month, day=day)


def _append_rule(
    rows: list[dict[str, Any]],
    *,
    rule_id: str,
    policy_name: str,
    publish_time: pd.Timestamp | None,
    effective_start: pd.Timestamp | None,
    effective_end: pd.Timestamp | None,
    rule_type: str,
    scope: str,
    state_name: str,
    state_value: Any,
    source_file: str,
    note: str,
) -> None:
    rows.append(
        {
            "rule_id": rule_id,
            "policy_name": policy_name,
            "publish_time": publish_time,
            "effective_start": effective_start,
            "effective_end": effective_end,
            "rule_type": rule_type,
            "scope": scope,
            "state_name": state_name,
            "state_value": state_value,
            "source_file": source_file,
            "note": note,
        }
    )


def parse_policy_environment(policy_dir: str | Path) -> PolicyParseResult:
    files = _list_policy_files(policy_dir)
    inventory_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    mechanism_xlsx_summary: dict[str, Any] = {}

    for path in files:
        publish_time, effective_start_guess = _parse_filename_dates(path.name)
        effective_start = effective_start_guess
        extracted_text = ""
        parse_status = "ok"
        parse_note = ""
        try:
            if path.suffix.lower() == ".docx":
                extracted_text = _extract_docx_text(path)
                publish_time = publish_time or _extract_notice_date(extracted_text)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                mechanism_xlsx_summary = _extract_xlsx_summary(path)
            else:
                parse_status = "skipped"
                parse_note = f"暂不解析 {path.suffix}"
        except Exception as exc:  # pragma: no cover - 容错记录
            parse_status = "failed"
            parse_note = str(exc)
            failure_rows.append({"source_file": str(path), "error": str(exc)})

        inventory_rows.append(
            {
                "source_file": str(path),
                "file_name": path.name,
                "suffix": path.suffix.lower(),
                "publish_time": publish_time,
                "effective_start_guess": effective_start,
                "parse_status": parse_status,
                "parse_note": parse_note,
            }
        )

        if parse_status == "failed":
            continue

        lower_name = path.name.lower()
        lower_text = extracted_text.lower()
        policy_name = path.stem
        scope = "湖南省售电周度持仓实验"

        if "现货市场交易实施细则" in path.name:
            base_effective = effective_start or publish_time
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="market_coupling",
                scope=scope,
                state_name="lt_settlement_base",
                state_value=1.0,
                source_file=str(path),
                note="中长期动作用作结算底仓，不直接映射物理执行曲线。",
            )
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="market_coupling",
                scope=scope,
                state_name="spot_marginal_exposure",
                state_value=1.0,
                source_file=str(path),
                note="现货市场承担边际敞口。",
            )
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="market_coupling",
                scope=scope,
                state_name="lt_spot_coupling_state",
                state_value=1.0,
                source_file=str(path),
                note="中长期与现货存在显式市场衔接机制。",
            )
            if "二级价格限值" in extracted_text:
                _append_rule(
                    rule_rows,
                    rule_id=f"rule_{len(rule_rows)+1:03d}",
                    policy_name=policy_name,
                    publish_time=publish_time,
                    effective_start=base_effective,
                    effective_end=None,
                    rule_type="ancillary_coupling",
                    scope=scope,
                    state_name="ancillary_price_cap_tight",
                    state_value=1.0,
                    source_file=str(path),
                    note="现货细则中存在二级价格限值与风险防控安排。",
                )

        if "辅助服务" in path.name or "调频辅助服务" in path.name or "辅助服务市场" in lower_text:
            base_effective = effective_start or publish_time
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="ancillary_coupling",
                scope=scope,
                state_name="ancillary_peak_shaving_pause",
                state_value=float("暂停" in extracted_text or "调峰" in extracted_text),
                source_file=str(path),
                note="辅助服务规则影响现货可调边际空间。",
            )
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="ancillary_coupling",
                scope=scope,
                state_name="ancillary_freq_reserve_tight",
                state_value=1.0,
                source_file=str(path),
                note="调频容量预留会压缩现货边际可调空间。",
            )

        if "新能源机制电价" in path.name or "上网电价市场化改革" in path.name:
            base_effective = effective_start or publish_time
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="renewable_mechanism",
                scope=scope,
                state_name="renewable_mechanism_active",
                state_value=1.0,
                source_file=str(path),
                note="新能源机制电价制度进入执行准备或实施阶段。",
            )
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="renewable_mechanism",
                scope=scope,
                state_name="mechanism_stage_label",
                state_value="竞价实施",
                source_file=str(path),
                note="新能源机制电价阶段标签。",
            )

        if "完善2026年度电力中长期交易价格机制" in path.name:
            base_effective = pd.Timestamp("2026-02-01 00:00:00")
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="lt_price_linkage",
                scope=scope,
                state_name="lt_price_linked_active",
                state_value=1.0,
                source_file=str(path),
                note="2026年2月起中长期价格与现货价格挂钩联动。",
            )
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="lt_price_linkage",
                scope=scope,
                state_name="fixed_price_ratio_max",
                state_value=0.4,
                source_file=str(path),
                note="固定价占比上限 40%。",
            )
            _append_rule(
                rule_rows,
                rule_id=f"rule_{len(rule_rows)+1:03d}",
                policy_name=policy_name,
                publish_time=publish_time,
                effective_start=base_effective,
                effective_end=None,
                rule_type="lt_price_linkage",
                scope=scope,
                state_name="linked_price_ratio_min",
                state_value=0.6,
                source_file=str(path),
                note="联动价占比下限 60%。",
            )

    inventory = pd.DataFrame(inventory_rows).sort_values(["publish_time", "file_name"], na_position="last").reset_index(drop=True)
    if mechanism_xlsx_summary:
        source_file = str(
            next(path for path in files if path.suffix.lower() in {".xlsx", ".xls"} and "机制电价竞价结果公布表" in path.name)
        )
        for state_name, value, note in [
            ("mechanism_price_floor", mechanism_xlsx_summary.get("mechanism_price_floor"), "机制电价下限代理取竞价结果最小值。"),
            ("mechanism_price_ceiling", mechanism_xlsx_summary.get("mechanism_price_ceiling"), "机制电价上限代理取竞价结果最大值。"),
            ("mechanism_volume_ratio_max", mechanism_xlsx_summary.get("mechanism_volume_ratio_max"), "机制电量比例上限取公布结果最大值。"),
            ("mechanism_exec_term_years", mechanism_xlsx_summary.get("mechanism_exec_term_years"), "机制电价执行期限取公布结果最大值。"),
        ]:
            if value is not None:
                _append_rule(
                    rule_rows,
                    rule_id=f"rule_{len(rule_rows)+1:03d}",
                    policy_name="湖南省2025年度新能源增量项目机制电价竞价结果公布表",
                    publish_time=pd.Timestamp("2025-12-25"),
                    effective_start=pd.Timestamp("2025-12-25"),
                    effective_end=None,
                    rule_type="renewable_mechanism",
                    scope="湖南省新能源增量项目",
                    state_name=state_name,
                    state_value=value,
                    source_file=source_file,
                    note=note,
                )

    failures = pd.DataFrame(failure_rows)
    rule_table = pd.DataFrame(rule_rows).sort_values(["effective_start", "rule_id"], na_position="last").reset_index(drop=True)
    return PolicyParseResult(inventory=inventory, rule_table=rule_table, failures=failures)
