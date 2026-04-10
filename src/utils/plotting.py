from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_line_plot(
    x: Iterable,
    y: Iterable,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str | None = None,
) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.plot(list(x), list(y), label=label)
    if label:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_multi_line_plot(
    frame: pd.DataFrame,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(10, 4.8))
    for column in frame.columns:
        plt.plot(frame.index.astype(str), frame[column], label=str(column))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bar_plot(
    frame: pd.DataFrame,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(10, 4.8))
    frame.plot(kind="bar", ax=plt.gca())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
