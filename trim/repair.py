from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Sequence

import torch.nn as nn

from .runtime import dispose_materialized

if TYPE_CHECKING:
    from .pruner import PruningUnit, TRIMPruner, UnitCandidate


def budget_excess(
    materialized: dict[str, Any],
    *,
    dense_params: int,
    dense_gflops: float,
    target_param_ratio: float,
    target_gflops_ratio: float,
    param_metric: str,
) -> tuple[float, float]:
    actual_param_value = (
        float(materialized["pruned_stats"]["params"])
        if param_metric == "measured"
        else float(materialized["pruned_raw_params"])
    )
    actual_gflops_value = float(materialized["pruned_stats"]["flops_value"])
    param_budget = float(dense_params) * float(target_param_ratio)
    gflops_budget = float(dense_gflops) * float(target_gflops_ratio)
    param_excess = max(0.0, actual_param_value - param_budget)
    gflops_excess = max(0.0, actual_gflops_value - gflops_budget)
    return param_excess, gflops_excess


def candidate_lookup(
    pruner: "TRIMPruner", unit: "PruningUnit", candidate: "UnitCandidate"
) -> int:
    for index, unit_candidate in enumerate(unit.candidates):
        if unit_candidate.keep_channels == candidate.keep_channels:
            return index
    raise RuntimeError(
        f"Unable to locate candidate keep_channels={candidate.keep_channels} for unit {unit.name}"
    )


def allocation_candidate_lookup(
    selected: dict[str, "UnitCandidate"],
    allocation: Any,
) -> int:
    for index, candidate in enumerate(allocation.candidates):
        matches = True
        for unit_name in allocation.unit_names:
            selected_candidate = selected.get(unit_name)
            allocation_candidate = candidate.member_candidates.get(unit_name)
            if selected_candidate is None or allocation_candidate is None:
                matches = False
                break
            if getattr(selected_candidate, "keep_channels", None) != getattr(
                allocation_candidate, "keep_channels", None
            ):
                matches = False
                break
        if matches:
            return index
    raise RuntimeError(
        f"Unable to locate current candidate for allocation {allocation.name}"
    )


def clone_pruner_model_to_cpu(pruner: "TRIMPruner") -> nn.Module:
    return copy.deepcopy(pruner.model.to("cpu")).cpu()


def is_cuda_device_assert(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{current.__class__.__module__}.{current.__class__.__name__}: {current}"
        if "device-side assert" in text or "cudaErrorAssert" in text:
            return True
        current = (
            current.__cause__ if current.__cause__ is not None else current.__context__
        )
    return False


def _format_candidate(unit: "PruningUnit", candidate: "UnitCandidate") -> str:
    return (
        f"{unit.name}(scope={unit.scope}, kind={unit.kind}, "
        f"keep={candidate.keep_channels}/{unit.channels}, "
        f"pruned={len(candidate.prune_indices)}, ratio={candidate.ratio:.4g})"
    )


def summarize_selection(
    pruner: "TRIMPruner",
    selected: dict[str, "UnitCandidate"],
    *,
    limit: int = 20,
) -> str:
    pruned_entries: list[str] = []
    identity_entries: list[str] = []
    for unit in pruner.units:
        candidate = selected.get(unit.name)
        if candidate is None:
            continue
        entry = _format_candidate(unit, candidate)
        if candidate.prune_indices:
            pruned_entries.append(entry)
        else:
            identity_entries.append(entry)

    entries = pruned_entries + identity_entries
    visible_entries = entries[:limit]
    suffix = f"; ... +{len(entries) - limit} more" if len(entries) > limit else ""
    return (
        f"total_units={len(entries)}, pruned_units={len(pruned_entries)}, "
        f"entries=[{'; '.join(visible_entries)}{suffix}]"
    )


def raise_if_cuda_device_assert(
    pruner: "TRIMPruner",
    selected: dict[str, "UnitCandidate"],
    exc: BaseException,
    *,
    phase: str,
    active_unit: "PruningUnit" | None = None,
    active_candidate: "UnitCandidate" | None = None,
) -> None:
    if not is_cuda_device_assert(exc):
        return
    active = "none"
    if active_unit is not None and active_candidate is not None:
        active = _format_candidate(active_unit, active_candidate)
    raise RuntimeError(
        "CUDA device-side assert during TRIM pruning materialization; "
        "the CUDA context is unsafe to reuse, so repair stopped at the first failing selection. "
        f"phase={phase}; active_unit={active}; "
        f"selection={summarize_selection(pruner, selected)}; original_error={exc}"
    ) from exc


def evaluate_actual_marginal_proposal(
    pruner: "TRIMPruner",
    proposal: dict[str, "UnitCandidate"],
    *,
    dense_model: nn.Module,
    dense_stats: dict[str, Any],
    dense_params: int,
    dense_gflops: float,
    target_param_ratio: float,
    target_gflops_ratio: float,
    param_metric: str,
    current_param_excess: float,
    current_gflops_excess: float,
) -> tuple[dict[str, Any], float, float, float] | None:
    try:
        proposal_materialized = pruner.materialize_selection(
            proposal,
            dense_model=dense_model,
            dense_stats=dense_stats,
            validate_forward=True,
        )
    except RuntimeError as exc:
        raise_if_cuda_device_assert(
            pruner,
            proposal,
            exc,
            phase="actual_marginal_proposal",
        )
        return None

    next_param_excess, next_gflops_excess = budget_excess(
        proposal_materialized,
        dense_params=dense_params,
        dense_gflops=dense_gflops,
        target_param_ratio=target_param_ratio,
        target_gflops_ratio=target_gflops_ratio,
        param_metric=param_metric,
    )

    param_gain = current_param_excess - next_param_excess
    gflops_gain = current_gflops_excess - next_gflops_excess
    score = 0.0
    if current_param_excess > 1e-9:
        score += param_gain / current_param_excess
    if current_gflops_excess > 1e-9:
        score += gflops_gain / current_gflops_excess
    return proposal_materialized, next_param_excess, next_gflops_excess, score


def search_pairwise_actual_marginal_step(
    pruner: "TRIMPruner",
    current_selected: dict[str, "UnitCandidate"],
    *,
    invalid_step_units: Sequence["PruningUnit"],
    dense_model: nn.Module,
    dense_stats: dict[str, Any],
    dense_params: int,
    dense_gflops: float,
    target_param_ratio: float,
    target_gflops_ratio: float,
    param_metric: str,
    current_param_excess: float,
    current_gflops_excess: float,
) -> tuple[
    tuple["PruningUnit", dict[str, "UnitCandidate"], dict[str, Any], float, float]
    | None,
    float,
]:
    candidate_units: list[tuple[float, float, str, "PruningUnit"]] = []
    for unit in invalid_step_units:
        if unit.scope != "Backbone":
            continue
        current_candidate = current_selected[unit.name]
        current_index = candidate_lookup(pruner, unit, current_candidate)
        if current_index + 1 >= len(unit.candidates):
            continue
        next_candidate = unit.candidates[current_index + 1]
        candidate_units.append(
            (
                max(0.0, next_candidate.delta_gflops - current_candidate.delta_gflops),
                float(
                    max(0, next_candidate.delta_params - current_candidate.delta_params)
                ),
                unit.name,
                unit,
            )
        )

    if len(candidate_units) < 2:
        return None, -math.inf

    candidate_units.sort(key=lambda item: (-item[0], -item[1], item[2]))
    pair_search_units = [unit for *_rest, unit in candidate_units]

    best_payload = None
    best_score = -math.inf
    for left_index, left_unit in enumerate(pair_search_units[:-1]):
        left_candidate = left_unit.candidates[
            candidate_lookup(pruner, left_unit, current_selected[left_unit.name]) + 1
        ]
        for right_unit in pair_search_units[left_index + 1 :]:
            right_candidate = right_unit.candidates[
                candidate_lookup(pruner, right_unit, current_selected[right_unit.name])
                + 1
            ]
            proposal = dict(current_selected)
            proposal[left_unit.name] = left_candidate
            proposal[right_unit.name] = right_candidate
            evaluation = evaluate_actual_marginal_proposal(
                pruner,
                proposal,
                dense_model=dense_model,
                dense_stats=dense_stats,
                dense_params=dense_params,
                dense_gflops=dense_gflops,
                target_param_ratio=target_param_ratio,
                target_gflops_ratio=target_gflops_ratio,
                param_metric=param_metric,
                current_param_excess=current_param_excess,
                current_gflops_excess=current_gflops_excess,
            )
            if evaluation is None:
                continue
            proposal_materialized, next_param_excess, next_gflops_excess, score = (
                evaluation
            )
            if score > best_score + 1e-12:
                best_score = score
                best_payload = (
                    left_unit,
                    proposal,
                    proposal_materialized,
                    next_param_excess,
                    next_gflops_excess,
                )

    if best_payload is not None:
        pruner.runtime_diagnostics["pairwise_actual_marginal_repair_invocations"] += 1
        left_unit = best_payload[0]
        changed_units = [
            unit_name
            for unit_name, candidate in best_payload[1].items()
            if candidate.keep_channels != current_selected[unit_name].keep_channels
        ]
        pruner._log(
            f"actual-marginal repair used pairwise coupled step anchored at {left_unit.name}: "
            f"changed={','.join(changed_units)}"
        )
    return best_payload, best_score


def search_allocation_actual_marginal_step(
    pruner: "TRIMPruner",
    current_selected: dict[str, "UnitCandidate"],
    *,
    dense_model: nn.Module,
    dense_stats: dict[str, Any],
    dense_params: int,
    dense_gflops: float,
    target_param_ratio: float,
    target_gflops_ratio: float,
    param_metric: str,
    current_param_excess: float,
    current_gflops_excess: float,
) -> tuple[
    tuple[Any, dict[str, "UnitCandidate"], dict[str, Any], float, float] | None,
    float,
    int,
]:
    best_payload = None
    best_score = -math.inf
    matched_allocations = 0

    for allocation in getattr(pruner, "allocation_objects", []):
        try:
            current_index = allocation_candidate_lookup(current_selected, allocation)
        except RuntimeError:
            continue
        matched_allocations += 1
        if current_index + 1 >= len(allocation.candidates):
            continue

        next_candidate = allocation.candidates[current_index + 1]
        proposal = dict(current_selected)
        proposal.update(next_candidate.member_candidates)
        evaluation = evaluate_actual_marginal_proposal(
            pruner,
            proposal,
            dense_model=dense_model,
            dense_stats=dense_stats,
            dense_params=dense_params,
            dense_gflops=dense_gflops,
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
            param_metric=param_metric,
            current_param_excess=current_param_excess,
            current_gflops_excess=current_gflops_excess,
        )
        if evaluation is None:
            continue
        proposal_materialized, next_param_excess, next_gflops_excess, score = evaluation

        if score > best_score + 1e-12:
            if best_payload is not None:
                dispose_materialized(best_payload[2])
            best_score = score
            best_payload = (
                allocation,
                proposal,
                proposal_materialized,
                next_param_excess,
                next_gflops_excess,
            )
        else:
            dispose_materialized(proposal_materialized)

    return best_payload, best_score, matched_allocations


def repair_selection_with_actual_marginals(
    pruner: "TRIMPruner",
    selected: dict[str, "UnitCandidate"],
    materialized: dict[str, Any],
    *,
    dense_model: nn.Module,
    dense_params: int,
    dense_gflops: float,
    target_param_ratio: float,
    target_gflops_ratio: float,
    param_metric: str,
) -> tuple[dict[str, "UnitCandidate"], dict[str, Any]]:
    current_selected = dict(selected)
    current_materialized = materialized
    max_unit_steps = sum(max(0, len(unit.candidates) - 1) for unit in pruner.units)
    max_allocation_steps = sum(
        max(0, len(allocation.candidates) - 1)
        for allocation in getattr(pruner, "allocation_objects", [])
    )
    max_steps = max(max_unit_steps, max_allocation_steps)

    for step in range(1, max_steps + 1):
        current_param_excess, current_gflops_excess = budget_excess(
            current_materialized,
            dense_params=dense_params,
            dense_gflops=dense_gflops,
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
            param_metric=param_metric,
        )
        if current_param_excess <= 1e-6 and current_gflops_excess <= 1e-6:
            pruner._log(f"actual-marginal repair succeeded in {step - 1} greedy steps.")
            return current_selected, current_materialized

        best_payload, best_score, matched_allocations = (
            search_allocation_actual_marginal_step(
                pruner,
                current_selected,
                dense_model=dense_model,
                dense_stats=current_materialized["dense_stats"],
                dense_params=dense_params,
                dense_gflops=dense_gflops,
                target_param_ratio=target_param_ratio,
                target_gflops_ratio=target_gflops_ratio,
                param_metric=param_metric,
                current_param_excess=current_param_excess,
                current_gflops_excess=current_gflops_excess,
            )
        )

        if matched_allocations == 0:
            best_payload = None
            best_score = -math.inf
            invalid_step_units: list["PruningUnit"] = []
            for unit in pruner.units:
                current_candidate = current_selected[unit.name]
                current_index = candidate_lookup(pruner, unit, current_candidate)
                if current_index + 1 >= len(unit.candidates):
                    continue

                proposal = dict(current_selected)
                proposal[unit.name] = unit.candidates[current_index + 1]
                evaluation = evaluate_actual_marginal_proposal(
                    pruner,
                    proposal,
                    dense_model=dense_model,
                    dense_stats=current_materialized["dense_stats"],
                    dense_params=dense_params,
                    dense_gflops=dense_gflops,
                    target_param_ratio=target_param_ratio,
                    target_gflops_ratio=target_gflops_ratio,
                    param_metric=param_metric,
                    current_param_excess=current_param_excess,
                    current_gflops_excess=current_gflops_excess,
                )
                if evaluation is None:
                    invalid_step_units.append(unit)
                    continue
                proposal_materialized, next_param_excess, next_gflops_excess, score = (
                    evaluation
                )

                if score > best_score + 1e-12:
                    if best_payload is not None:
                        dispose_materialized(best_payload[2])
                    best_score = score
                    best_payload = (
                        unit,
                        proposal,
                        proposal_materialized,
                        next_param_excess,
                        next_gflops_excess,
                    )
                else:
                    dispose_materialized(proposal_materialized)

            if best_payload is None or best_score <= 1e-12:
                pairwise_payload, pairwise_score = search_pairwise_actual_marginal_step(
                    pruner,
                    current_selected,
                    invalid_step_units=invalid_step_units,
                    dense_model=dense_model,
                    dense_stats=current_materialized["dense_stats"],
                    dense_params=dense_params,
                    dense_gflops=dense_gflops,
                    target_param_ratio=target_param_ratio,
                    target_gflops_ratio=target_gflops_ratio,
                    param_metric=param_metric,
                    current_param_excess=current_param_excess,
                    current_gflops_excess=current_gflops_excess,
                )
                if pairwise_payload is not None and pairwise_score > best_score + 1e-12:
                    if best_payload is not None:
                        dispose_materialized(best_payload[2])
                    best_payload = pairwise_payload
                    best_score = pairwise_score
                elif pairwise_payload is not None:
                    dispose_materialized(pairwise_payload[2])

        if best_payload is None or best_score <= 1e-12:
            break

        previous_materialized = current_materialized
        (
            best_unit,
            current_selected,
            current_materialized,
            next_param_excess,
            next_gflops_excess,
        ) = best_payload
        if previous_materialized is not current_materialized:
            dispose_materialized(previous_materialized)
        pruner._log(
            f"actual-marginal repair step {step}: strengthened {best_unit.name}; "
            f"remaining param_excess={next_param_excess:.4f}, "
            f"gflops_excess={next_gflops_excess:.4f}"
        )

    final_param_excess, final_gflops_excess = budget_excess(
        current_materialized,
        dense_params=dense_params,
        dense_gflops=dense_gflops,
        target_param_ratio=target_param_ratio,
        target_gflops_ratio=target_gflops_ratio,
        param_metric=param_metric,
    )
    if final_param_excess <= 1e-6 and final_gflops_excess <= 1e-6:
        return current_selected, current_materialized
    final_param_value = (
        float(current_materialized["pruned_stats"]["params"])
        if param_metric == "measured"
        else float(current_materialized["pruned_raw_params"])
    )
    final_gflops_value = float(current_materialized["pruned_stats"]["flops_value"])
    actual_param_ratio = (
        final_param_value / float(dense_params) if dense_params > 0 else math.inf
    )
    actual_gflops_ratio = (
        final_gflops_value / float(dense_gflops) if dense_gflops > 0.0 else math.inf
    )
    raise RuntimeError(
        "Actual-marginal repair could not satisfy pruning budgets after exhausting "
        "positive actual-marginal steps: "
        f"param_excess={final_param_excess:.4f}, gflops_excess={final_gflops_excess:.4f}, "
        f"actual_param_ratio={actual_param_ratio:.4f}, target_param_ratio={target_param_ratio:.4f}, "
        f"actual_gflops_ratio={actual_gflops_ratio:.4f}, target_gflops_ratio={target_gflops_ratio:.4f}"
    )


def repair_invalid_selection_to_valid(
    pruner: "TRIMPruner",
    selected: dict[str, "UnitCandidate"],
    *,
    dense_model: nn.Module,
    dense_stats: dict[str, Any],
) -> tuple[dict[str, "UnitCandidate"], dict[str, Any]]:
    pruner.runtime_diagnostics["invalid_selection_repair_invocations"] += 1
    current_selected = dict(selected)
    max_steps = sum(
        candidate_lookup(pruner, unit, current_selected[unit.name])
        for unit in pruner.units
    )
    last_error: RuntimeError | None = None

    for step in range(1, max_steps + 1):
        relaxable = []
        for unit in pruner.units:
            current_candidate = current_selected[unit.name]
            current_index = candidate_lookup(pruner, unit, current_candidate)
            if current_index <= 0:
                continue
            relaxed_candidate = unit.candidates[current_index - 1]
            relaxable.append(
                (
                    max(
                        0.0,
                        current_candidate.delta_gflops - relaxed_candidate.delta_gflops,
                    ),
                    max(
                        0,
                        current_candidate.delta_params - relaxed_candidate.delta_params,
                    ),
                    max(0.0, current_candidate.cost - relaxed_candidate.cost),
                    unit.name,
                    unit,
                    relaxed_candidate,
                )
            )

        if not relaxable:
            break

        relaxable.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        _, _, _, _, best_unit, relaxed_candidate = relaxable[0]
        current_selected[best_unit.name] = relaxed_candidate
        pruner._log(
            f"invalid-selection repair step {step}: relaxed {best_unit.name} to "
            f"keep={relaxed_candidate.keep_channels}/{best_unit.channels}."
        )

        try:
            materialized = pruner.materialize_selection(
                current_selected,
                dense_model=dense_model,
                dense_stats=dense_stats,
                validate_forward=True,
            )
            pruner._log(
                f"invalid-selection repair succeeded in {step} relaxation steps."
            )
            return current_selected, materialized
        except RuntimeError as exc:
            raise_if_cuda_device_assert(
                pruner,
                current_selected,
                exc,
                phase="invalid_selection_repair",
                active_unit=best_unit,
                active_candidate=relaxed_candidate,
            )
            last_error = exc

    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(
        f"Could not repair invalid pruning selection into a valid model{detail}"
    )


def repair_selection_to_budget(
    pruner: "TRIMPruner",
    selected: dict[str, "UnitCandidate"],
    *,
    dense_params: int,
    dense_gflops: float,
    target_param_ratio: float,
    target_gflops_ratio: float,
    param_metric: str = "measured",
    max_rounds: int = 3,
) -> tuple[dict[str, "UnitCandidate"], dict[str, Any]]:
    if param_metric not in {"measured", "raw"}:
        raise ValueError(f"Unsupported param_metric: {param_metric}")
    if max_rounds <= 0:
        raise ValueError(f"max_rounds must be positive, got: {max_rounds}")

    current_selected = dict(selected)
    try:
        dense_model = clone_pruner_model_to_cpu(pruner)
    except RuntimeError as exc:
        raise_if_cuda_device_assert(
            pruner,
            current_selected,
            exc,
            phase="dense_model_cpu_clone",
        )
        raise
    adjusted_param_ratio = float(target_param_ratio)
    adjusted_gflops_ratio = float(target_gflops_ratio)
    dense_stats_payload = {
        "params": int(dense_params),
        "params_million": float(dense_params) / 1_000_000.0,
        "flops_value": float(dense_gflops),
    }

    materialized = None
    for attempt in range(1, max_rounds + 1):
        try:
            materialized = pruner.materialize_selection(
                current_selected,
                dense_model=dense_model,
                dense_stats=dense_stats_payload,
                validate_forward=True,
            )
        except RuntimeError as exc:
            raise_if_cuda_device_assert(
                pruner,
                current_selected,
                exc,
                phase=f"budget_repair_round_{attempt}",
            )
            pruner._log(
                f"budget repair round {attempt}: selected combination is structurally invalid "
                f"({exc}); relaxing to nearest valid selection before re-solving."
            )
            current_selected, materialized = repair_invalid_selection_to_valid(
                pruner,
                current_selected,
                dense_model=dense_model,
                dense_stats=dense_stats_payload,
            )
        actual_param_value = (
            float(materialized["pruned_stats"]["params"])
            if param_metric == "measured"
            else float(materialized["pruned_raw_params"])
        )
        actual_gflops_value = float(materialized["pruned_stats"]["flops_value"])
        param_budget = float(dense_params) * float(target_param_ratio)
        gflops_budget = float(dense_gflops) * float(target_gflops_ratio)
        param_excess = max(0.0, actual_param_value - param_budget)
        gflops_excess = max(0.0, actual_gflops_value - gflops_budget)

        if param_excess <= 1e-6 and gflops_excess <= 1e-6:
            if attempt > 1:
                pruner._log(
                    f"budget repair succeeded on round {attempt}: "
                    f"params={actual_param_value:.4f}, fwd_gflops={actual_gflops_value:.4f}"
                )
            return current_selected, materialized

        pruner._log(
            f"budget repair round {attempt}: actual params={actual_param_value:.4f} "
            f"(budget={param_budget:.4f}), fwd_gflops={actual_gflops_value:.4f} "
            f"(budget={gflops_budget:.4f}); re-solving with tighter estimated targets."
        )
        if param_excess > 0.0:
            adjusted_param_ratio = max(
                0.0, adjusted_param_ratio - (param_excess + 1.0) / float(dense_params)
            )
        if gflops_excess > 0.0:
            adjusted_gflops_ratio = max(
                0.0,
                adjusted_gflops_ratio - (gflops_excess + 1e-4) / float(dense_gflops),
            )
        try:
            current_selected = pruner.solve_allocation(
                dense_params=dense_params,
                dense_gflops=dense_gflops,
                target_param_ratio=adjusted_param_ratio,
                target_gflops_ratio=adjusted_gflops_ratio,
            )
        except RuntimeError as exc:
            message = str(exc)
            if (
                "MILP pruning allocation budget is unreachable by the current legal candidate set"
                not in message
            ):
                raise
            pruner._log(
                f"budget repair round {attempt}: estimated re-solve could not tighten further "
                f"({exc}); falling back to actual-marginal greedy repair."
            )
            return pruner._repair_selection_with_actual_marginals(
                current_selected,
                materialized,
                dense_model=dense_model,
                dense_params=dense_params,
                dense_gflops=dense_gflops,
                target_param_ratio=target_param_ratio,
                target_gflops_ratio=target_gflops_ratio,
                param_metric=param_metric,
            )

    pruner._log(
        "MILP repair rounds exhausted; falling back to actual-marginal greedy repair."
    )
    if materialized is None:
        raise RuntimeError("budget repair did not materialize any pruning selection")
    stale_materialized = materialized
    try:
        materialized = pruner.materialize_selection(
            current_selected,
            dense_model=dense_model,
            dense_stats=dense_stats_payload,
            validate_forward=True,
        )
    except RuntimeError as exc:
        raise_if_cuda_device_assert(
            pruner,
            current_selected,
            exc,
            phase="actual_marginal_fallback_alignment",
        )
        pruner._log(
            "actual-marginal fallback alignment: latest MILP selection is structurally invalid "
            f"({exc}); relaxing to nearest valid selection before greedy repair."
        )
        current_selected, materialized = repair_invalid_selection_to_valid(
            pruner,
            current_selected,
            dense_model=dense_model,
            dense_stats=dense_stats_payload,
        )
    if materialized is not stale_materialized:
        dispose_materialized(stale_materialized)
    return pruner._repair_selection_with_actual_marginals(
        current_selected,
        materialized,
        dense_model=dense_model,
        dense_params=dense_params,
        dense_gflops=dense_gflops,
        target_param_ratio=target_param_ratio,
        target_gflops_ratio=target_gflops_ratio,
        param_metric=param_metric,
    )
