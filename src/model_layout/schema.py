from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterBlockSpec:
    name: str
    columns: list[str]
    slice_start: int
    slice_end: int

    @property
    def size(self) -> int:
        return self.slice_end - self.slice_start


@dataclass(frozen=True)
class LayerLayout:
    total_dimension: int
    blocks: list[ParameterBlockSpec]


@dataclass(frozen=True)
class CompiledParameterLayout:
    upper: LayerLayout
    lower: LayerLayout
    feature_sources: dict[str, list[str]]
