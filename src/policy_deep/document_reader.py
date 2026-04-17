from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.policy.policy_parser import parse_policy_environment


@dataclass(frozen=True)
class PolicyDocumentReadResult:
    inventory: pd.DataFrame
    document_units: pd.DataFrame
    parsed_rules: pd.DataFrame
    failures: pd.DataFrame


def read_policy_documents(policy_directory: str | Path, project_root: str | Path | None = None) -> PolicyDocumentReadResult:
    parsed = parse_policy_environment(policy_directory, project_root=project_root)
    unit_rows: list[dict[str, object]] = []
    for index, row in parsed.rule_table.reset_index(drop=True).iterrows():
        unit_rows.append(
            {
                "document_id": f"doc_{index + 1:04d}",
                "unit_id": f"unit_{index + 1:04d}",
                "source_file": row["source_file"],
                "file_name": Path(str(row["source_file"])).name,
                "unit_type": "rule_note",
                "heading_path": str(row.get("state_group", "")),
                "paragraph_index": index,
                "table_index": None,
                "text": str(row.get("note", "")),
                "raw_value": row.get("state_value"),
            }
        )
    document_units = pd.DataFrame(unit_rows)
    if document_units.empty:
        document_units = pd.DataFrame(
            columns=[
                "document_id",
                "unit_id",
                "source_file",
                "file_name",
                "unit_type",
                "heading_path",
                "paragraph_index",
                "table_index",
                "text",
                "raw_value",
            ]
        )
    return PolicyDocumentReadResult(
        inventory=parsed.inventory,
        document_units=document_units,
        parsed_rules=parsed.rule_table,
        failures=parsed.failures,
    )
