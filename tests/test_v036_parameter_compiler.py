from __future__ import annotations

import pandas as pd
import pytest

from src.model_layout.compiler import compile_parameter_layout


def _bundle() -> dict:
    weeks = pd.to_datetime(["2026-01-05", "2026-01-12"])
    return {
        "weekly_features": pd.DataFrame(
            {
                "week_start": weeks,
                "feature_a": [1.0, 2.0],
                "feature_b": [0.5, 1.5],
                "feature_c": [3.0, 4.0],
            }
        ),
        "policy_state_trace": pd.DataFrame(
            {
                "week_start": weeks,
                "renewable_mechanism_active": [0.0, 1.0],
                "lt_price_linked_active": [0.0, 1.0],
                "forward_price_linkage_days": [10.0, 3.0],
            }
        ),
        "agent_feature_columns": ["feature_a", "feature_b", "feature_c"],
    }


def test_compile_parameter_layout_builds_stable_upper_and_lower_dimensions() -> None:
    config = {
        "parameter_compiler": {
            "upper": {
                "weekly_feature_source": "agent_feature_columns",
                "policy_feature_source": "policy_state_numeric_columns",
                "blocks": {
                    "weekly_feature_weights": {"source": "weekly_feature_source"},
                    "policy_feature_weights": {
                        "source": "policy_feature_source",
                        "include": ["renewable_mechanism_active", "lt_price_linked_active"],
                    },
                    "contract_curve_latent": {"size": 12},
                    "action_head": {"size": 6},
                },
            },
            "lower": {
                "hourly_feature_groups": {
                    "spread_response": {"columns": ["price_spread"], "response_size": 8},
                    "load_deviation_response": {"columns": ["load_dev"], "response_size": 8},
                    "renewable_response": {"columns": ["renewable_dev"], "response_size": 8},
                    "policy_shrink_response": {"columns": ["ancillary_freq_reserve_tight"], "response_size": 8},
                }
            },
        }
    }

    layout = compile_parameter_layout(config=config, bundle=_bundle())

    assert layout.upper.total_dimension == 3 + 2 + 12 + 6
    assert layout.lower.total_dimension == 32
    assert layout.upper.blocks[0].name == "weekly_feature_weights"
    assert layout.upper.blocks[0].slice_start == 0
    assert layout.upper.blocks[0].slice_end == 3
    assert layout.upper.blocks[1].slice_start == 3
    assert layout.upper.blocks[1].slice_end == 5


def test_compile_parameter_layout_errors_for_missing_declared_columns() -> None:
    config = {
        "parameter_compiler": {
            "upper": {
                "weekly_feature_source": "agent_feature_columns",
                "policy_feature_source": "policy_state_numeric_columns",
                "blocks": {
                    "policy_feature_weights": {
                        "source": "policy_feature_source",
                        "include": ["missing_policy_column"],
                    }
                },
            },
            "lower": {
                "hourly_feature_groups": {
                    "spread_response": {"columns": ["price_spread"], "response_size": 8},
                }
            },
        }
    }

    with pytest.raises(KeyError, match="missing_policy_column"):
        compile_parameter_layout(config=config, bundle=_bundle())
