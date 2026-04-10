from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WeekSplit:
    warmup: list[pd.Timestamp]
    train: list[pd.Timestamp]
    val: list[pd.Timestamp]
    test: list[pd.Timestamp]


def _weeks_between(weeks: list[pd.Timestamp], start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return [week for week in weeks if start_ts <= week <= end_ts]


def build_week_split(config: dict, weekly_features: pd.DataFrame, weekly_metadata: pd.DataFrame) -> WeekSplit:
    feature_weeks = sorted(pd.to_datetime(weekly_features["week_start"]).tolist())
    meta_weeks = sorted(pd.to_datetime(weekly_metadata["week_start"]).tolist())
    split_cfg = config["split"]

    train = _weeks_between(feature_weeks, split_cfg["train_start_week"], split_cfg["train_end_week"])
    val = _weeks_between(feature_weeks, split_cfg["val_start_week"], split_cfg["val_end_week"])
    test = _weeks_between(feature_weeks, split_cfg["test_start_week"], split_cfg["test_end_week"])

    warmup_cutoff = pd.Timestamp(split_cfg["train_start_week"])
    warmup = [week for week in meta_weeks if week < warmup_cutoff]

    if not train:
        raise ValueError("训练周为空，无法构建周度训练场景。")
    if not val:
        raise ValueError("验证周为空，无法构建验证场景。")
    if not test:
        raise ValueError("回测周为空，无法构建回测场景。")

    return WeekSplit(warmup=warmup, train=train, val=val, test=test)


def build_bootstrap_sequence(
    train_weeks: list[pd.Timestamp],
    sequence_length: int,
    block_size: int,
    seed: int,
) -> list[pd.Timestamp]:
    if not train_weeks:
        raise ValueError("训练周为空，无法进行 block bootstrap。")
    if len(train_weeks) == 1:
        return train_weeks * sequence_length

    rng = np.random.default_rng(seed)
    sequence: list[pd.Timestamp] = []
    while len(sequence) < sequence_length:
        start = int(rng.integers(0, len(train_weeks)))
        for offset in range(block_size):
            sequence.append(train_weeks[(start + offset) % len(train_weeks)])
            if len(sequence) >= sequence_length:
                break
    return sequence
