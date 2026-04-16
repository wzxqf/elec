from __future__ import annotations

import unittest

import pandas as pd

from src.config.load_config import load_runtime_config
from src.policy.market_constraints import (
    build_market_rule_constraints,
    build_market_rule_constraints_markdown,
    validate_market_rule_alignment,
)
from src.policy.policy_parser import parse_policy_environment
from src.policy.policy_regime import build_policy_state_trace


class MarketRuleConstraintsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_runtime_config(".")
        cls.policy_result = parse_policy_environment(cls.config["policy_directory"])
        cls.constraints = build_market_rule_constraints(cls.config, cls.policy_result.rule_table)

    def test_constraints_include_core_market_rule_mappings(self) -> None:
        ids = set(self.constraints["constraint_id"])

        self.assertIn("lt_contract_settlement_only", ids)
        self.assertIn("spot_15min_96_settlement", ids)
        self.assertIn("renewable_mechanism_from_2026_01_01", ids)
        self.assertIn("lt_price_linkage_from_2026_02_01", ids)

        linkage = self.constraints.set_index("constraint_id").loc["lt_price_linkage_from_2026_02_01"]
        self.assertEqual(pd.Timestamp(linkage["effective_start"]), pd.Timestamp("2026-02-01"))
        self.assertIn("lt_price_linked_active", linkage["model_mapping"])
        self.assertIn("40%", linkage["market_rule"])
        self.assertIn("60%", linkage["market_rule"])

    def test_policy_alignment_keeps_effective_dates_from_leaking_forward(self) -> None:
        weekly_metadata = pd.DataFrame(
            {
                "week_start": pd.to_datetime(["2025-12-22", "2026-01-05", "2026-01-26", "2026-02-02"]),
                "week_end": pd.to_datetime(["2025-12-28", "2026-01-11", "2026-02-01", "2026-02-08"]),
            }
        )
        trace = build_policy_state_trace(
            weekly_metadata=weekly_metadata,
            rule_table=self.policy_result.rule_table,
            inventory=self.policy_result.inventory,
            config=self.config,
        )

        violations = validate_market_rule_alignment(self.config, self.policy_result.rule_table, trace)

        self.assertEqual(violations, [])
        by_week = trace.set_index("week_start")
        self.assertEqual(float(by_week.loc[pd.Timestamp("2025-12-22"), "renewable_mechanism_active"]), 0.0)
        self.assertEqual(float(by_week.loc[pd.Timestamp("2026-01-05"), "renewable_mechanism_active"]), 1.0)
        self.assertEqual(float(by_week.loc[pd.Timestamp("2026-01-26"), "lt_price_linked_active"]), 0.0)
        self.assertEqual(float(by_week.loc[pd.Timestamp("2026-02-02"), "lt_price_linked_active"]), 1.0)

    def test_markdown_report_is_human_readable_and_source_backed(self) -> None:
        markdown = build_market_rule_constraints_markdown(
            config=self.config,
            constraints=self.constraints,
            rule_table=self.policy_result.rule_table,
            violations=[],
        )

        self.assertIn("# 市场规则约束与模型映射", markdown)
        self.assertIn("中长期合约仅作为结算依据", markdown)
        self.assertIn("2026-01-01", markdown)
        self.assertIn("2026-02-01", markdown)
        self.assertIn("政策来源文件", markdown)
        self.assertIn("模型映射", markdown)
        self.assertNotIn("; d:\\elec\\政策环境\\2025.10.15湖南省电力现货市场交易实施细则.docx", markdown)


if __name__ == "__main__":
    unittest.main()
