from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def resolve_llm_candidates(
    candidate_rules: pd.DataFrame,
    *,
    enabled: bool,
    cache_directory: str | Path,
) -> pd.DataFrame:
    cache_path = Path(cache_directory)
    cache_path.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.md5(candidate_rules.to_json(orient="records", force_ascii=False).encode("utf-8")).hexdigest()
    target = cache_path / f"llm_candidates_{content_hash}.json"
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)

    resolved = candidate_rules.copy()
    resolved["extractor"] = "llm_candidate_stub" if enabled else resolved["extractor"]
    resolved["confidence"] = resolved["confidence"].astype(float)
    target.write_text(json.dumps(resolved.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return resolved
