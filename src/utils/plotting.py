from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _csv_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(".csv")


def save_line_plot(
    x: Iterable,
    y: Iterable,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str | None = None,
) -> None:
    frame = pd.DataFrame(
        {
            xlabel: list(x),
            label or ylabel: list(y),
        }
    )
    frame.to_csv(_csv_path(path), index=False)


def save_multi_line_plot(
    frame: pd.DataFrame,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    export_frame = frame.copy()
    export_frame.index.name = xlabel
    export_frame.reset_index().to_csv(_csv_path(path), index=False)


def save_bar_plot(
    frame: pd.DataFrame,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    export_frame = frame.copy()
    export_frame.index.name = xlabel
    export_frame.reset_index().to_csv(_csv_path(path), index=False)
