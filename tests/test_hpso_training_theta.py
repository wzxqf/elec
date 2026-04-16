from __future__ import annotations

import numpy as np
from pathlib import Path

from src.agents.hpso_param_policy import HPSOParamPolicyModel, load_hpso_param_policy, save_hpso_param_policy


def test_hpso_param_policy_roundtrip() -> None:
    model = HPSOParamPolicyModel(
        theta=np.arange(64, dtype=float),
        objective_value=123.0,
        metadata={"version": "v0.32"},
    )
    path = Path(".cache") / "tests" / "hpso_param_policy.json"

    save_hpso_param_policy(model, path)
    loaded = load_hpso_param_policy(path)

    assert loaded.objective_value == 123.0
    assert loaded.metadata["version"] == "v0.32"
    assert loaded.theta.tolist() == list(np.arange(64, dtype=float))
