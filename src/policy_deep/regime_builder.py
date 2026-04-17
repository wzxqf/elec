from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.policy.policy_regime import build_policy_state_trace
from src.policy_deep.candidate_rules import build_candidate_rules
from src.policy_deep.document_reader import read_policy_documents
from src.policy_deep.llm_bridge import resolve_llm_candidates
from src.policy_deep.rule_review import review_candidate_rules


@dataclass(frozen=True)
class PolicyDeepContext:
    inventory: pd.DataFrame
    document_units: pd.DataFrame
    candidate_rules: pd.DataFrame
    reviewed_rules: pd.DataFrame
    rule_table: pd.DataFrame
    state_trace: pd.DataFrame
    failures: pd.DataFrame


def build_policy_deep_context(
    *,
    policy_directory: str | Path,
    weekly_metadata: pd.DataFrame,
    config: dict[str, Any],
) -> PolicyDeepContext:
    document_result = read_policy_documents(policy_directory, project_root=config["project_root"])
    candidate_rules = build_candidate_rules(document_result.parsed_rules, document_result.document_units)
    llm_cfg = config.get("policy_deep", {}).get("llm_candidate_parser", {})
    llm_candidates = resolve_llm_candidates(
        candidate_rules,
        enabled=bool(llm_cfg.get("enabled", False)),
        cache_directory=Path(config["outputs"]["root"]) / str(config["project"]["version"]) / "policy_cache",
    )
    reviewed_rules, rule_table = review_candidate_rules(llm_candidates)
    if not rule_table.empty and "source_file" in document_result.parsed_rules.columns:
        source_lookup = (
            document_result.parsed_rules[["policy_name", "state_name", "source_file"]]
            .drop_duplicates()
            .set_index(["policy_name", "state_name"])["source_file"]
            .to_dict()
        )
        rule_table["source_file"] = rule_table.apply(
            lambda row: source_lookup.get((row["policy_name"], row["state_name"]), ""),
            axis=1,
        )
    state_trace = build_policy_state_trace(
        weekly_metadata=weekly_metadata,
        rule_table=rule_table,
        inventory=document_result.inventory,
        config=config,
    )
    return PolicyDeepContext(
        inventory=document_result.inventory,
        document_units=document_result.document_units,
        candidate_rules=llm_candidates,
        reviewed_rules=reviewed_rules,
        rule_table=rule_table,
        state_trace=state_trace,
        failures=document_result.failures,
    )
