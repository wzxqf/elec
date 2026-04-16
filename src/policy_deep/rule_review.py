from __future__ import annotations

import pandas as pd


def review_candidate_rules(candidate_rules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviewed = candidate_rules.copy()
    if reviewed.empty:
        return reviewed, pd.DataFrame(
            columns=[
                "rule_id",
                "policy_name",
                "publish_time",
                "effective_start",
                "effective_end",
                "rule_type",
                "scope",
                "state_group",
                "state_name",
                "state_value",
                "source_file",
                "note",
            ]
        )

    reviewed["review_status"] = "accepted"
    accepted = reviewed.loc[reviewed["review_status"] == "accepted"].reset_index(drop=True)
    rule_table = pd.DataFrame(
        {
            "rule_id": [f"reviewed_{index + 1:04d}" for index in range(len(accepted))],
            "policy_name": accepted["policy_name"],
            "publish_time": accepted["publish_time"],
            "effective_start": accepted["effective_start_candidate"],
            "effective_end": accepted["effective_end_candidate"],
            "rule_type": accepted["rule_type_candidate"],
            "scope": accepted["scope_candidate"],
            "state_group": accepted["state_group_candidate"],
            "state_name": accepted["state_name_candidate"],
            "state_value": accepted["state_value_candidate"],
            "source_file": accepted.get("source_file", ""),
            "note": accepted["evidence_text"],
        }
    )
    return reviewed, rule_table
