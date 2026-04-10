from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_COLUMNS = [
    "datetime",
    "省调负荷_日前",
    "新能源负荷-总加_日前",
    "新能源负荷-光伏_日前",
    "新能源负荷-风电_日前",
    "联络线总加_日前",
    "非市场化机组出力_日前",
    "水电出力_日前",
    "全网总出力_日前",
    "全网统一出清价格_日前",
    "省调负荷_日内",
    "新能源负荷-总加_日内",
    "新能源负荷-光伏_日内",
    "新能源负荷-风电_日内",
    "联络线总加_日内",
    "非市场化机组出力_日内",
    "水电出力_日内",
    "全网总出力_日内",
    "全网统一出清价格_日内",
]


def locate_total_csv(project_root: str | Path, candidates: Iterable[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    root = Path(project_root)
    for path in root.rglob("total.csv"):
        if path.name == "total.csv":
            return path

    raise FileNotFoundError("未找到真实数据文件 total.csv，按要求停止执行。")


def load_raw_total_csv(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"total.csv 缺少必要列: {missing}")
    return frame
