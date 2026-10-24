from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch.nn as nn

from .allocation_types import AllocationCandidate, AllocationObject
from .complexity import _conv_output_hw, _count_tokens, _flops_to_gflops

CONTEXT_BLOCK_GATE_PATTERN = re.compile(r"^encoder\.encoder\.(\d+)\.(\d+)$")


@dataclass
class BlockGateUnit:
    name: str
    scope: str
    kind: str
    module_name: str
    delta_params: int
    delta_gflops: float
    cost: float
    materialize_supported: bool


@dataclass
class BlockGateCandidate:
    ratio: float
    keep_channels: int
    prune_indices: list[int]
    delta_params: int
    delta_gflops: float
    cost: float
    materialize_supported: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def _search_space_enabled(pruner: Any, *modes: str) -> bool:
    search_space = getattr(pruner, "search_space", "stable")
    return search_space == "all_experimental" or search_space in modes


def _resolve_module_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    if not parts:
        raise ValueError("module_name must not be empty")
    parent = root
    for part in parts[:-1]:
        if part not in parent._modules:
            raise KeyError(
                f"Unable to resolve module path {module_name}: missing {part}"
            )
        parent = parent._modules[part]
    return parent, parts[-1]


def _set_module_by_name(
    root: nn.Module, module_name: str, replacement: nn.Module
) -> None:
    parent, leaf = _resolve_module_parent(root, module_name)
    if leaf not in parent._modules:
        raise KeyError(f"Unable to resolve module path {module_name}: missing {leaf}")
    parent._modules[leaf] = replacement


def _module_param_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _module_gflops_from_cached_shapes(pruner: Any, module_name: str) -> float:
    module_names = dict(pruner.model.named_modules())
    flops = 0.0
    prefix = f"{module_name}."
    for child_name, child in module_names.items():
        if child_name != module_name and not child_name.startswith(prefix):
            continue
        output_shape = pruner.output_shape_cache.get(child_name)
        if isinstance(child, nn.Conv2d):
            out_h, out_w = _conv_output_hw(output_shape or ())
            kernel_h, kernel_w = child.kernel_size
            input_per_output = child.in_channels // child.groups
            flops += (
                2.0
                * child.out_channels
                * out_h
                * out_w
                * input_per_output
                * kernel_h
                * kernel_w
            )
        elif isinstance(child, nn.Conv1d):
            if output_shape is None or len(output_shape) < 3:
                out_l = 1
                batch = 1
            else:
                batch = int(output_shape[0])
                out_l = int(output_shape[-1])
            kernel = int(child.kernel_size[0])
            input_per_output = child.in_channels // child.groups
            flops += (
                2.0 * batch * child.out_channels * out_l * input_per_output * kernel
            )
        elif isinstance(child, nn.Linear):
            token_count = _count_tokens(output_shape) if output_shape is not None else 1
            flops += 2.0 * token_count * child.in_features * child.out_features
        elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if output_shape is None:
                continue
            flops += 2.0 * int(np.prod(output_shape))
        elif isinstance(child, nn.LayerNorm):
            if output_shape is None:
                continue
            flops += 2.0 * int(np.prod(output_shape))
    return _flops_to_gflops(flops)


def estimate_block_gate_delta(
    pruner: Any, module_name: str, module: nn.Module
) -> tuple[int, float]:
    return _module_param_count(module), _module_gflops_from_cached_shapes(
        pruner, module_name
    )


def build_block_gate_allocation(unit: BlockGateUnit) -> AllocationObject:
    keep_candidate = BlockGateCandidate(
        ratio=1.0,
        keep_channels=1,
        prune_indices=[],
        delta_params=0,
        delta_gflops=0.0,
        cost=0.0,
        materialize_supported=True,
        metadata={
            "kind": "block_gate",
            "module_name": unit.module_name,
            "scope": unit.scope,
            "gate": "keep",
        },
    )
    drop_candidate = BlockGateCandidate(
        ratio=0.0,
        keep_channels=0,
        prune_indices=[0],
        delta_params=int(unit.delta_params),
        delta_gflops=float(unit.delta_gflops),
        cost=float(unit.cost),
        materialize_supported=bool(unit.materialize_supported),
        metadata={
            "kind": "block_gate",
            "module_name": unit.module_name,
            "scope": unit.scope,
            "gate": "drop",
        },
    )
    return AllocationObject(
        name=unit.name,
        unit_names=[unit.name],
        kind="block_gate",
        candidates=[
            AllocationCandidate(
                ratio=1.0,
                member_candidates={unit.name: keep_candidate},
                delta_params=0,
                delta_gflops=0.0,
                cost=0.0,
                materialize_supported=True,
            ),
            AllocationCandidate(
                ratio=0.0,
                member_candidates={unit.name: drop_candidate},
                delta_params=int(unit.delta_params),
                delta_gflops=float(unit.delta_gflops),
                cost=float(unit.cost),
                materialize_supported=bool(unit.materialize_supported),
            ),
        ],
    )


def collect_hybrid_encoder_context_block_gates(pruner: Any) -> list[BlockGateUnit]:
    units: list[BlockGateUnit] = []
    module_names = dict(pruner.model.named_modules())
    gate_module_names = set(context_block_gate_module_names(pruner, module_names))
    for module_name, module in module_names.items():
        match = CONTEXT_BLOCK_GATE_PATTERN.match(module_name)
        if not match or module_name not in gate_module_names:
            continue
        if isinstance(module, nn.Identity):
            continue
        delta_params, delta_gflops = estimate_block_gate_delta(
            pruner, module_name, module
        )
        if delta_params <= 0 and delta_gflops <= 0.0:
            continue
        units.append(
            BlockGateUnit(
                name=f"Encoder::encoder.{match.group(1)}.{match.group(2)}",
                scope="Encoder",
                kind="block_gate",
                module_name=module_name,
                delta_params=delta_params,
                delta_gflops=delta_gflops,
                cost=max(1.0, float(delta_params) / 1_000_000.0),
                materialize_supported=True,
            )
        )
    return units


def context_block_gate_module_names(
    pruner: Any, module_names: dict[str, nn.Module] | None = None
) -> list[str]:
    if not _search_space_enabled(pruner, "block_gate"):
        return []
    module_names = (
        dict(pruner.model.named_modules()) if module_names is None else module_names
    )
    encoder = module_names.get("encoder")
    if getattr(encoder, "context_block", None) != "fire_mamba":
        return []
    return [
        module_name
        for module_name, module in module_names.items()
        if CONTEXT_BLOCK_GATE_PATTERN.match(module_name)
        and not isinstance(module, nn.Identity)
    ]


def build_block_gate_allocation_objects(pruner: Any) -> list[AllocationObject]:
    allocations = [
        build_block_gate_allocation(unit)
        for unit in collect_hybrid_encoder_context_block_gates(pruner)
    ]
    pruner.runtime_diagnostics["block_gate_count"] = len(allocations)
    return allocations


def _selected_block_gate_candidates(selected: dict[str, Any]) -> list[Any]:
    candidates = []
    for candidate in selected.values():
        metadata = getattr(candidate, "metadata", {})
        if metadata.get("kind") == "block_gate" and getattr(
            candidate, "prune_indices", []
        ):
            candidates.append(candidate)
    return candidates


def ensure_block_gate_selection_materializable(selected: dict[str, Any]) -> None:
    for candidate in _selected_block_gate_candidates(selected):
        if not getattr(candidate, "materialize_supported", False):
            module_name = getattr(candidate, "metadata", {}).get(
                "module_name", "<unknown>"
            )
            raise RuntimeError(
                f"unsupported block gate candidate selected for {module_name}; "
                "run budget probe only or add a materialization handler before exporting an artifact."
            )


def apply_block_gate_selection(model: nn.Module, selected: dict[str, Any]) -> list[str]:
    applied = []
    for candidate in _selected_block_gate_candidates(selected):
        module_name = getattr(candidate, "metadata", {}).get("module_name")
        if not module_name:
            raise RuntimeError("block gate candidate is missing module_name metadata")
        _set_module_by_name(model, module_name, nn.Identity())
        applied.append(module_name)
    return applied
