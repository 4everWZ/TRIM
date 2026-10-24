from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from .block_gates import (
    apply_block_gate_selection,
    ensure_block_gate_selection_materializable,
)
from .complexity import _measure_model_complexity
from .runtime import eval_spatial_size_override, safe_gc_collect

if TYPE_CHECKING:
    from .pruner import TRIMPruner, UnitCandidate


def apply_pruning(
    pruner: "TRIMPruner", selected: dict[str, "UnitCandidate"]
) -> nn.Module:
    pruner._log("applying structural pruning...")
    applied_units = 0
    for unit in pruner.units:
        candidate = selected[unit.name]
        if not candidate.prune_indices:
            continue
        note = pruner.unit_kind_handlers.apply_pruning(pruner, unit, candidate)
        applied_units += 1
        if note:
            pruner._log(
                f"pruned {unit.name} {note}: removed {len(candidate.prune_indices)}/{unit.channels} channels."
            )
        else:
            pruner._log(
                f"pruned {unit.name}: removed {len(candidate.prune_indices)}/{unit.channels} channels."
            )
    pruner._log(f"pruning apply complete ({applied_units} units modified).")
    return pruner.model


def validate_cuda_forward(pruner: "TRIMPruner") -> None:
    if pruner.device.type != "cuda":
        return
    pruner._log("validating CUDA forward on pruned model...")
    pruner.model.eval()
    with (
        torch.no_grad(),
        eval_spatial_size_override(pruner.model, pruner.input_shape[-2:]),
    ):
        _ = pruner.model(pruner.example_inputs[0])
        torch.cuda.synchronize(pruner.device)
    pruner._log("CUDA forward validation passed.")


def materialize_selection(
    pruner: "TRIMPruner",
    selected: dict[str, "UnitCandidate"],
    *,
    dense_model: nn.Module | None = None,
    dense_stats: dict[str, Any] | None = None,
    validate_forward: bool = True,
) -> dict[str, Any]:
    from .pruner import TRIMPruner

    source_model = dense_model if dense_model is not None else pruner.model
    clone_model = None
    clone_pruner = None
    try:
        ensure_block_gate_selection_materializable(selected)
        clone_model = copy.deepcopy(source_model).to(pruner.device)
        dense_raw_params = sum(
            parameter.numel() for parameter in clone_model.parameters()
        )
        dense_stats_payload = (
            dense_stats
            if dense_stats is not None
            else _measure_model_complexity(
                clone_model, pruner.input_shape, pruner.device
            )
        )
        apply_block_gate_selection(clone_model, selected)
        clone_pruner = TRIMPruner(
            model=clone_model,
            criterion=pruner.criterion,
            device=pruner.device,
            input_shape=pruner.input_shape,
            candidate_ratios=pruner.candidate_ratios,
            lambda_h=pruner.lambda_h,
            lambda_i=pruner.lambda_i,
            score_mode=pruner.score_mode,
            search_space=pruner.search_space,
            verbose=False,
            progress_interval=pruner.progress_interval,
        )
        clone_pruner.collect_units()
        dense_selected = {
            unit.name: selected.get(unit.name, pruner._identity_candidate(unit))
            for unit in clone_pruner.units
        }
        apply_pruning(clone_pruner, dense_selected)
        if validate_forward:
            validate_cuda_forward(clone_pruner)
        pruned_raw_params = sum(
            parameter.numel() for parameter in clone_pruner.model.parameters()
        )
        pruned_stats = _measure_model_complexity(
            clone_pruner.model, pruner.input_shape, pruner.device
        )
        return {
            "pruner": clone_pruner,
            "model": clone_pruner.model,
            "units": clone_pruner.units,
            "dense_raw_params": int(dense_raw_params),
            "pruned_raw_params": int(pruned_raw_params),
            "dense_stats": dense_stats_payload,
            "pruned_stats": pruned_stats,
        }
    except Exception:
        del clone_pruner
        del clone_model
        safe_gc_collect()
        raise
