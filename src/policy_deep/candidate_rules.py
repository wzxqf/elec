from __future__ import annotations

import pandas as pd


def build_candidate_rules(parsed_rules: pd.DataFrame, document_units: pd.DataFrame) -> pd.DataFrame:
    if parsed_rules.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "policy_name",
                "publish_time",
                "effective_start_candidate",
                "effective_end_candidate",
                "rule_type_candidate",
                "scope_candidate",
                "state_group_candidate",
                "state_name_candidate",
                "state_value_candidate",
                "evidence_unit_id",
                "evidence_text",
                "extractor",
                "confidence",
                "review_status",
            ]
        )

    unit_lookup = document_units.set_index("source_file")["unit_id"].to_dict() if not document_units.empty else {}
    rows: list[dict[str, object]] = []
    for index, row in parsed_rules.reset_index(drop=True).iterrows():
        rows.append(
            {
                "candidate_id": f"candidate_{index + 1:04d}",
                "policy_name": row["policy_name"],
                "publish_time": row["publish_time"],
                "effective_start_candidate": row["effective_start"],
                "effective_end_candidate": row["effective_end"],
                "rule_type_candidate": row["rule_type"],
                "scope_candidate": row["scope"],
                "state_group_candidate": row["state_group"],
                "state_name_candidate": row["state_name"],
                "state_value_candidate": row["state_value"],
                "evidence_unit_id": unit_lookup.get(row["source_file"], ""),
                "evidence_text": row.get("note", ""),
                "extractor": "parsed_rule_bootstrap",
                "confidence": 0.85,
                "review_status": "pending",
            }
        )
    return pd.DataFrame(rows)
