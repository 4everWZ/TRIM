from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from .allocation_types import AllocationCandidate, AllocationObject, CoupledUnitGroup

if TYPE_CHECKING:
    from .pruner import TRIMPruner


@dataclass
class SharedAxisCouplingPolicy:
    eligible_kinds: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "backbone_conv",
                "neck_conv",
                "wavelet_spatial",
                "wavelet_mask",
            }
        )
    )

    def is_eligible(self, pruner: "TRIMPruner", unit: Any) -> bool:
        return (
            unit.kind in self.eligible_kinds
            and isinstance(unit.module, nn.Conv2d)
            and pruner.unit_kind_handlers.supports_shared_axis_coupling(unit)
        )


class SharedAxisCouplingEngine:
    def __init__(self, policy: SharedAxisCouplingPolicy | None = None) -> None:
        self.policy = policy or SharedAxisCouplingPolicy()

    def eligible_units(self, pruner: "TRIMPruner") -> list[Any]:
        return [unit for unit in pruner.units if self.policy.is_eligible(pruner, unit)]

    def probe_root_pruning_indices(self, unit: Any) -> list[tuple[int, ...]]:
        if unit.channels <= 0:
            return []
        group_size = max(1, unit.group_size)
        anchors = {
            0,
            max(0, ((unit.channels - group_size) // (2 * group_size)) * group_size),
            max(0, ((unit.channels - group_size) // group_size) * group_size),
        }
        probes: list[tuple[int, ...]] = []
        for anchor in sorted(anchors):
            probe = tuple(range(anchor, min(anchor + group_size, unit.channels)))
            if len(probe) == group_size:
                probes.append(probe)
        return probes

    def canonical_group_signature(
        self,
        pruner: "TRIMPruner",
        unit: Any,
        prune_indices: list[int] | tuple[int, ...],
    ) -> set[tuple[str, str, tuple[int, ...]]]:
        group = pruner.dependency_graph.get_pruning_group(
            unit.module, unit.pruning_fn, idxs=list(prune_indices)
        )
        signature: set[tuple[str, str, tuple[int, ...]]] = set()
        for dependency, dep_indices in group.items:
            target = dependency.target.module
            target_name = pruner.module_name_map.get(id(target))
            if target_name is None:
                continue
            signature.add(
                (
                    target_name,
                    dependency.handler.__name__,
                    tuple(sorted(set(int(index) for index in dep_indices))),
                )
            )
        return signature

    def discover_groups(self, pruner: "TRIMPruner") -> list[CoupledUnitGroup]:
        if not pruner.units:
            pruner.collect_units()

        eligible_units = self.eligible_units(pruner)
        overlaps: dict[str, set[str]] = {unit.name: set() for unit in eligible_units}
        signatures_by_unit: dict[str, set[tuple[str, str, tuple[int, ...]]]] = {}

        for unit in eligible_units:
            combined_signature: set[tuple[str, str, tuple[int, ...]]] = set()
            for probe_indices in self.probe_root_pruning_indices(unit):
                combined_signature.update(
                    self.canonical_group_signature(pruner, unit, probe_indices)
                )
            signatures_by_unit[unit.name] = combined_signature

        for left_index, left_unit in enumerate(eligible_units[:-1]):
            left_signature = signatures_by_unit[left_unit.name]
            if not left_signature:
                continue
            for right_unit in eligible_units[left_index + 1 :]:
                if (
                    left_unit.channels != right_unit.channels
                    or left_unit.group_size != right_unit.group_size
                    or left_unit.channel_dim != right_unit.channel_dim
                ):
                    continue
                right_signature = signatures_by_unit[right_unit.name]
                if left_signature & right_signature:
                    overlaps[left_unit.name].add(right_unit.name)
                    overlaps[right_unit.name].add(left_unit.name)

        groups: list[CoupledUnitGroup] = []
        visited: set[str] = set()
        for unit in eligible_units:
            if unit.name in visited:
                continue
            stack = [unit.name]
            component: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(sorted(overlaps[current] - visited))
            groups.append(
                CoupledUnitGroup(
                    name="SharedAxis::" + "|".join(sorted(component)),
                    unit_names=sorted(component),
                )
            )

        pruner.shared_axis_groups = groups
        pruner.runtime_diagnostics["shared_axis_group_count"] = sum(
            1 for group in groups if len(group.unit_names) > 1
        )
        return groups

    def build_coupled_allocation(
        self, pruner: "TRIMPruner", group: CoupledUnitGroup
    ) -> AllocationObject:
        members = [pruner._unit_by_name(unit_name) for unit_name in group.unit_names]
        candidate_ratios: list[AllocationCandidate] = []
        for ratio in pruner.candidate_ratios:
            member_candidates: dict[str, Any] = {}
            duplicate = False
            for unit in members:
                keep_channels = pruner._rounded_keep_channels(unit, ratio)
                candidate = next(
                    (
                        candidate
                        for candidate in unit.candidates
                        if candidate.keep_channels == keep_channels
                    ),
                    None,
                )
                if candidate is None:
                    duplicate = True
                    break
                member_candidates[unit.name] = candidate
            if duplicate:
                continue
            if all(
                not candidate.prune_indices for candidate in member_candidates.values()
            ):
                delta_params = 0
                delta_gflops = 0.0
            else:
                delta_params, delta_gflops = (
                    pruner._estimate_member_candidate_union_delta(member_candidates)
                )
                if not pruner._materialize_member_candidate_smoke(member_candidates):
                    pruner.runtime_diagnostics[
                        "coupled_candidates_discarded_pre_milp"
                    ] += 1
                    continue
            cost = float(
                sum(candidate.cost for candidate in member_candidates.values())
            )
            candidate_ratios.append(
                AllocationCandidate(
                    ratio=ratio,
                    member_candidates=member_candidates,
                    delta_params=delta_params,
                    delta_gflops=delta_gflops,
                    cost=cost,
                )
            )
        if not candidate_ratios:
            candidate_ratios = [
                AllocationCandidate(
                    ratio=1.0,
                    member_candidates={
                        unit.name: pruner._identity_candidate(unit) for unit in members
                    },
                    delta_params=0,
                    delta_gflops=0.0,
                    cost=0.0,
                )
            ]
        return AllocationObject(
            name=group.name,
            unit_names=[unit.name for unit in members],
            candidates=candidate_ratios,
            kind="coupled",
        )

    def build_allocation_objects(self, pruner: "TRIMPruner") -> list[AllocationObject]:
        groups = self.discover_groups(pruner)
        grouped_names = {
            unit_name
            for group in groups
            if len(group.unit_names) > 1
            for unit_name in group.unit_names
        }
        allocation_objects: list[AllocationObject] = []
        for group in groups:
            if len(group.unit_names) > 1:
                allocation_objects.append(self.build_coupled_allocation(pruner, group))
            else:
                unit = pruner._unit_by_name(group.unit_names[0])
                allocation_objects.append(
                    AllocationObject(
                        name=unit.name,
                        unit_names=[unit.name],
                        candidates=[
                            AllocationCandidate(
                                ratio=candidate.ratio,
                                member_candidates={unit.name: candidate},
                                delta_params=candidate.delta_params,
                                delta_gflops=candidate.delta_gflops,
                                cost=candidate.cost,
                            )
                            for candidate in unit.candidates
                        ],
                        kind="single",
                    )
                )
        for unit in pruner.units:
            if unit.name in grouped_names:
                continue
            if any(
                unit.name in allocation.unit_names for allocation in allocation_objects
            ):
                continue
            allocation_objects.append(
                AllocationObject(
                    name=unit.name,
                    unit_names=[unit.name],
                    candidates=[
                        AllocationCandidate(
                            ratio=candidate.ratio,
                            member_candidates={unit.name: candidate},
                            delta_params=candidate.delta_params,
                            delta_gflops=candidate.delta_gflops,
                            cost=candidate.cost,
                        )
                        for candidate in unit.candidates
                    ],
                    kind="single",
                )
            )
        pruner.allocation_objects = allocation_objects
        return allocation_objects
