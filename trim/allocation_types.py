from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoupledUnitGroup:
    name: str
    unit_names: list[str]


@dataclass
class AllocationCandidate:
    ratio: float
    member_candidates: dict[str, object]
    delta_params: int
    delta_gflops: float
    cost: float
    materialize_supported: bool = True


@dataclass
class AllocationObject:
    name: str
    unit_names: list[str]
    candidates: list[AllocationCandidate]
    kind: str
