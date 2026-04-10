from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonthSplit:
    warmup: list[pd.Timestamp]
    train: list[pd.Timestamp]
    val: list[pd.Timestamp]
    test: list[pd.Timestamp]


def _parse_months(values: list[str]) -> list[pd.Timestamp]:
    return [pd.Period(value, freq="M").to_timestamp() for value in values]


def build_month_split(config: dict, monthly_features: pd.DataFrame) -> MonthSplit:
    available = set(pd.to_datetime(monthly_features["month"]).tolist())
    warmup = _parse_months([config["warmup_month"]])
    train = [month for month in _parse_months(config["train_months"]) if month in available]
    val = [month for month in _parse_months(config["val_months"]) if month in available]
    test = [month for month in _parse_months(config["test_months"]) if month in available]
    return MonthSplit(warmup=warmup, train=train, val=val, test=test)


def build_bootstrap_sequence(
    train_months: list[pd.Timestamp],
    sequence_length: int,
    block_size: int,
    seed: int,
) -> list[pd.Timestamp]:
    if not train_months:
        raise ValueError("训练月份为空，无法生成 bootstrap episode。")
    rng = np.random.default_rng(seed)
    if len(train_months) == 1:
        return train_months * sequence_length

    sequence: list[pd.Timestamp] = []
    while len(sequence) < sequence_length:
        start = int(rng.integers(0, len(train_months)))
        for offset in range(block_size):
            sequence.append(train_months[(start + offset) % len(train_months)])
            if len(sequence) >= sequence_length:
                break
    return sequence
