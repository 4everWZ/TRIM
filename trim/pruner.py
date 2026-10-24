from __future__ import annotations

import copy
import gc
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from scipy.optimize import Bounds, LinearConstraint, milp

from trim.operators.common import Mamba
from trim.operators.feed_forward import FeedForwardMLP, SwiGLUFeedForward
from trim.operators.metadata import collect_state_space_model_metadata
from trim.operators.normalization import FrozenBatchNorm2d
from trim.operators.wavelet import WaveletDownsample

from .adapters import build_default_unit_collector_registry
from .allocation_types import AllocationCandidate, AllocationObject, CoupledUnitGroup
from .artifact import save_pruned_artifact
from .block_gates import build_block_gate_allocation_objects
from .complexity import (
    _conv_output_hw,
    _count_tokens,
    _flops_to_gflops,
    _measure_model_complexity,
)
from .coupling import SharedAxisCouplingEngine, SharedAxisCouplingPolicy
from .handlers import build_default_unit_kind_handler_registry
from .materialize import apply_pruning as materialize_apply_pruning
from .materialize import materialize_selection as materialize_selection_impl
from .materialize import validate_cuda_forward as validate_cuda_forward_impl
from .repair import budget_excess as repair_budget_excess
from .repair import candidate_lookup as repair_candidate_lookup
from .repair import clone_pruner_model_to_cpu, raise_if_cuda_device_assert
from .repair import (
    evaluate_actual_marginal_proposal as evaluate_actual_marginal_proposal_impl,
)
from .repair import (
    repair_invalid_selection_to_valid as repair_invalid_selection_to_valid_impl,
)
from .repair import repair_selection_to_budget as repair_selection_to_budget_impl
from .repair import (
    repair_selection_with_actual_marginals as repair_selection_with_actual_marginals_impl,
)
from .repair import (
    search_pairwise_actual_marginal_step as search_pairwise_actual_marginal_step_impl,
)
from .runtime import (
    cleanup_cuda_memory,
    dispose_materialized,
    eval_spatial_size_override,
    load_dense_checkpoint,
    mamba_fast_path_disabled,
    set_model_eval_spatial_size,
)
from .scoring import (
    normalized_entropy,
    normalized_mutual_information,
    prediction_summary,
)
from .structural_pruners import (
    FrozenBatchNormPruner,
    MambaInnerWidthPruner,
    MLPWidthPruner,
    SwiGLUWidthPruner,
)

FAMILY_SPECIFIC_SCORE_MODE = "family_specific_entropy_mi"
SUPPORTED_SCORE_MODES = {
    "taylor",
    "taylor_entropy",
    "taylor_entropy_mi",
    FAMILY_SPECIFIC_SCORE_MODE,
}
SUPPORTED_SEARCH_SPACES = {
    "stable",
    "expanded_neck",
    "block_gate",
    "transformer_dims",
    "all_experimental",
}
TRANSFORMER_UNIT_KINDS = {"swiglu", "mlp_hidden"}
FISHER_UNIT_KINDS = TRANSFORMER_UNIT_KINDS | {"mamba_inner"}


@dataclass
class UnitCandidate:
    ratio: float
    keep_channels: int
    prune_indices: list[int]
    delta_params: int
    delta_gflops: float
    cost: float


@dataclass
class PruningUnit:
    name: str
    scope: str
    kind: str
    module_name: str
    activation_module_name: str
    module: nn.Module
    activation_module: nn.Module
    pruner: Any
    pruning_fn: Any
    channels: int
    group_size: int
    min_channels: int
    channel_dim: int
    hook_type: str = "forward"
    score: torch.Tensor | None = None
    entropy: float = 0.0
    mi: float = 0.0
    activation_shape: tuple[int, ...] | None = None
    candidates: list[UnitCandidate] = field(default_factory=list)


def parse_candidate_ratios(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        ratios = [float(item) for item in value]
    if not ratios:
        raise ValueError("candidate_ratios must contain at least one ratio.")
    for ratio in ratios:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"Candidate ratio must be in (0, 1], got: {ratio}")
    return sorted(set(ratios), reverse=True)


class TRIMPruner:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: str | torch.device = "cuda:0",
        input_shape: Sequence[int] = (1, 3, 640, 640),
        candidate_ratios: Sequence[float] = (1.0, 0.875, 0.75, 0.625, 0.5),
        lambda_h: float = 0.25,
        lambda_i: float = 0.25,
        score_mode: str = "taylor_entropy_mi",
        search_space: str = "stable",
        verbose: bool = True,
        progress_interval: int = 1,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = torch.device(device)
        self.input_shape = tuple(int(value) for value in input_shape)
        self.candidate_ratios = parse_candidate_ratios(candidate_ratios)
        self.lambda_h = float(lambda_h)
        self.lambda_i = float(lambda_i)
        if score_mode not in SUPPORTED_SCORE_MODES:
            raise ValueError(f"Unsupported score_mode: {score_mode}")
        self.score_mode = score_mode
        if search_space not in SUPPORTED_SEARCH_SPACES:
            raise ValueError(f"Unsupported TRIM search_space: {search_space}")
        self.search_space = search_space
        self.verbose = bool(verbose)
        if progress_interval <= 0:
            raise ValueError(
                f"progress_interval must be positive, got: {progress_interval}"
            )
        self.progress_interval = int(progress_interval)

        self.example_inputs = (torch.randn(self.input_shape, device=self.device),)
        self.custom_pruners: dict[type[nn.Module], Any] = {
            SwiGLUFeedForward: SwiGLUWidthPruner(),
            FeedForwardMLP: MLPWidthPruner(),
            FrozenBatchNorm2d: FrozenBatchNormPruner(),
        }
        if Mamba is not None:
            self.custom_pruners[Mamba] = MambaInnerWidthPruner()

        self.units: list[PruningUnit] = []
        self.shared_axis_groups: list[CoupledUnitGroup] = []
        self.allocation_objects: list[AllocationObject] = []
        self.dependency_graph = None
        self.module_name_map: dict[int, str] = {}
        self.output_shape_cache: dict[str, tuple[int, ...]] = {}
        self.runtime_diagnostics: dict[str, int] = {
            "shared_axis_group_count": 0,
            "coupled_candidates_discarded_pre_milp": 0,
            "block_gate_count": 0,
            "invalid_selection_repair_invocations": 0,
            "pairwise_actual_marginal_repair_invocations": 0,
        }
        self.unit_collectors = build_default_unit_collector_registry()
        self.unit_kind_handlers = build_default_unit_kind_handler_registry()
        self.shared_axis_coupling_engine = SharedAxisCouplingEngine(
            SharedAxisCouplingPolicy()
        )

    @staticmethod
    def _last_non_singleton_dim(parameter: nn.Parameter) -> int | None:
        non_singleton_dims = [
            index for index, size in enumerate(parameter.shape) if size > 1
        ]
        return non_singleton_dims[-1] if non_singleton_dims else None

    def _new_dependency_graph(self) -> tp.DependencyGraph:
        dependency_graph = tp.DependencyGraph()
        for customized_type, customized_pruner in self.custom_pruners.items():
            dependency_graph.register_customized_layer(
                customized_type, customized_pruner
            )
        return dependency_graph

    def _collect_unwrapped_parameters(
        self, dependency_graph: tp.DependencyGraph
    ) -> tuple[list[Any], list[nn.Parameter]]:
        prunable_module_types = dependency_graph.REGISTERED_PRUNERS.keys()
        customized_pruners = dependency_graph.CUSTOMIZED_PRUNERS.keys()
        wrapped_parameter_ids: set[int] = set()

        for module in self.model.modules():
            op_type = tp.ops.module2type(module)
            is_wrapped = (
                (
                    op_type in prunable_module_types
                    and op_type != tp.ops.OPTYPE.ELEMENTWISE
                )
                or module.__class__ in customized_pruners
                or module in customized_pruners
            )
            if not is_wrapped:
                continue
            for parameter in module.parameters():
                wrapped_parameter_ids.add(id(parameter))

        unwrapped_parameters = []
        ignored_unwrapped_parameters: list[nn.Parameter] = []
        for _name, parameter in self.model.named_parameters():
            if id(parameter) in wrapped_parameter_ids:
                continue
            pruning_dim = self._last_non_singleton_dim(parameter)
            if pruning_dim is None:
                if parameter.numel() == 1:
                    ignored_unwrapped_parameters.append(parameter)
                    pruning_dim = 0
                else:
                    continue
            unwrapped_parameters.append(
                tp._helpers.UnwrappedParameters(
                    parameters=parameter, pruning_dim=pruning_dim
                )
            )
        return unwrapped_parameters, ignored_unwrapped_parameters

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[TRIM] {message}", flush=True)

    def _cleanup_cuda_memory(self) -> None:
        cleanup_cuda_memory(self.device)

    @staticmethod
    def _identity_candidate(unit: PruningUnit) -> UnitCandidate:
        return UnitCandidate(
            ratio=1.0,
            keep_channels=int(unit.channels),
            prune_indices=[],
            delta_params=0,
            delta_gflops=0.0,
            cost=0.0,
        )

    @staticmethod
    def _identity_allocation_candidate(unit: PruningUnit) -> AllocationCandidate:
        return AllocationCandidate(
            ratio=1.0,
            member_candidates={unit.name: TRIMPruner._identity_candidate(unit)},
            delta_params=0,
            delta_gflops=0.0,
            cost=0.0,
        )

    @staticmethod
    def _allocation_feasibility_summary(
        allocation_objects: Sequence[AllocationObject],
        dense_params: int,
        dense_gflops: float,
        target_param_ratio: float,
        target_gflops_ratio: float,
    ) -> dict[str, float | int]:
        max_param_savings = sum(
            max(float(candidate.delta_params) for candidate in allocation.candidates)
            for allocation in allocation_objects
        )
        max_gflops_savings = sum(
            max(float(candidate.delta_gflops) for candidate in allocation.candidates)
            for allocation in allocation_objects
        )
        required_param_savings = max(
            0.0, dense_params - dense_params * float(target_param_ratio)
        )
        required_gflops_savings = max(
            0.0, dense_gflops - dense_gflops * float(target_gflops_ratio)
        )
        minimum_feasible_param_ratio = (
            max(0.0, 1.0 - max_param_savings / float(dense_params))
            if dense_params > 0
            else 1.0
        )
        minimum_feasible_gflops_ratio = (
            max(0.0, 1.0 - max_gflops_savings / float(dense_gflops))
            if dense_gflops > 0.0
            else 1.0
        )
        return {
            "allocation_objects": len(allocation_objects),
            "required_param_savings": required_param_savings,
            "max_param_savings": max_param_savings,
            "minimum_feasible_param_ratio": minimum_feasible_param_ratio,
            "required_gflops_savings": required_gflops_savings,
            "max_gflops_savings": max_gflops_savings,
            "minimum_feasible_gflops_ratio": minimum_feasible_gflops_ratio,
        }

    @staticmethod
    def _format_allocation_feasibility_summary(summary: dict[str, float | int]) -> str:
        return (
            f"allocation_objects={int(summary['allocation_objects'])}, "
            f"required_param_savings={float(summary['required_param_savings']):.0f}, "
            f"max_param_savings={float(summary['max_param_savings']):.0f}, "
            f"minimum_feasible_param_ratio={float(summary['minimum_feasible_param_ratio']):.4f}, "
            f"required_gflops_savings={float(summary['required_gflops_savings']):.4f}, "
            f"max_gflops_savings={float(summary['max_gflops_savings']):.4f}, "
            f"minimum_feasible_gflops_ratio={float(summary['minimum_feasible_gflops_ratio']):.4f}"
        )

    @staticmethod
    def _dependency_forward(model: nn.Module, example_inputs):
        x = (
            example_inputs[0]
            if isinstance(example_inputs, (tuple, list))
            else example_inputs
        )
        feats = model.backbone(x)
        return model.encoder(feats)

    def _register_unit(
        self,
        units: list[PruningUnit],
        *,
        name: str,
        scope: str,
        kind: str,
        module_name: str,
        activation_module_name: str,
        module: nn.Module,
        activation_module: nn.Module,
        pruner: Any,
        channel_dim: int,
        hook_type: str = "forward",
    ) -> None:
        group_size = (
            pruner.get_out_channel_groups(module)
            if hasattr(pruner, "get_out_channel_groups")
            else 1
        )
        channels = pruner.get_out_channels(module)
        if channels is None or channels <= 1:
            return
        units.append(
            PruningUnit(
                name=name,
                scope=scope,
                kind=kind,
                module_name=module_name,
                activation_module_name=activation_module_name,
                module=module,
                activation_module=activation_module,
                pruner=pruner,
                pruning_fn=pruner.prune_out_channels,
                channels=int(channels),
                group_size=max(1, int(group_size)),
                min_channels=max(1, int(group_size)),
                channel_dim=channel_dim,
                hook_type=hook_type,
            )
        )

    def build_dependency_graph(self):
        was_training = self.model.training
        self.model.eval()
        self._log("building dependency graph...")
        dependency_graph = self._new_dependency_graph()
        unwrapped_parameters, ignored_parameters = self._collect_unwrapped_parameters(
            dependency_graph
        )
        with (
            eval_spatial_size_override(self.model, self.input_shape[-2:]),
            mamba_fast_path_disabled(self.model),
        ):
            self.dependency_graph = dependency_graph.build_dependency(
                self.model,
                example_inputs=self.example_inputs,
                forward_fn=self._dependency_forward,
                unwrapped_parameters=unwrapped_parameters,
                ignored_params=ignored_parameters,
                verbose=False,
            )
            self.output_shape_cache = self._collect_output_shapes()
        self.model.train(was_training)
        self.module_name_map = {
            id(module): name for name, module in self.model.named_modules()
        }
        self._log("dependency graph ready.")
        return self.dependency_graph

    def _collect_output_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        hooks = []
        module_names = dict(self.model.named_modules())

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                tensor = output if torch.is_tensor(output) else None
                if tensor is not None:
                    shapes[name] = tuple(int(dim) for dim in tensor.shape)

            return hook

        for name, module in module_names.items():
            hooks.append(module.register_forward_hook(make_hook(name)))

        with (
            torch.no_grad(),
            eval_spatial_size_override(self.model, self.input_shape[-2:]),
        ):
            self.model.eval()
            self._dependency_forward(self.model, self.example_inputs)

        for hook in hooks:
            hook.remove()
        return shapes

    def collect_units(self) -> list[PruningUnit]:
        self.units = self.unit_collectors.collect(self)
        scope_counts: dict[str, int] = {}
        for unit in self.units:
            scope_counts[unit.scope] = scope_counts.get(unit.scope, 0) + 1
        scope_summary = ", ".join(
            f"{scope}={count}" for scope, count in sorted(scope_counts.items())
        )
        self._log(f"collected {len(self.units)} pruning units ({scope_summary}).")
        return self.units

    def _project_activation(self, unit: PruningUnit, payload: Any) -> torch.Tensor:
        if unit.hook_type == "pre":
            payload = payload[0]
        if unit.kind == "swiglu":
            x1, x2 = payload.chunk(2, dim=-1)
            payload = F.silu(x1) * x2
        if not torch.is_tensor(payload):
            raise TypeError(
                f"Activation hook for {unit.name} did not receive a tensor payload."
            )
        return payload.detach()

    def _channel_energy(
        self, activation: torch.Tensor, channel_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        activation = activation.float()
        if channel_dim < 0:
            channel_dim += activation.dim()
        reduce_dims = tuple(
            index for index in range(activation.dim()) if index != channel_dim
        )
        energy = activation.pow(2).mean(dim=reduce_dims)
        sample_summary = activation.abs().reshape(activation.shape[0], -1).mean(dim=1)
        return energy, sample_summary

    def _compute_unit_taylor(self, unit: PruningUnit) -> torch.Tensor:
        return self.unit_kind_handlers.compute_taylor(self, unit)

    def _compute_unit_fisher(self, unit: PruningUnit) -> torch.Tensor:
        if unit.kind == "mamba_inner":
            return self._mamba_fisher(unit.module)
        if unit.kind == "swiglu":
            return self._swiglu_fisher(unit.module)
        if unit.kind == "mlp_hidden":
            return self._mlp_fisher(unit.module)
        parameter = next(unit.module.parameters(), None)
        device = parameter.device if parameter is not None else self.device
        return torch.zeros(unit.channels, device=device)

    def _base_score_from_stats(
        self, unit: PruningUnit, unit_stats: dict[str, Any], batch_count: int
    ) -> torch.Tensor:
        divisor = max(1, int(batch_count))
        if self.score_mode == FAMILY_SPECIFIC_SCORE_MODE:
            if unit.kind in FISHER_UNIT_KINDS:
                return unit_stats["fisher"] / divisor
        return unit_stats["taylor"] / divisor

    def _conv_taylor(self, module: nn.Conv2d) -> torch.Tensor:
        if module.weight.grad is None:
            return torch.zeros(module.out_channels, device=module.weight.device)
        score = (
            (module.weight.detach() * module.weight.grad.detach())
            .abs()
            .sum(dim=(1, 2, 3))
        )
        if module.bias is not None and module.bias.grad is not None:
            score = score + (module.bias.detach() * module.bias.grad.detach()).abs()
        return score

    def _eligible_shared_axis_units(self) -> list[PruningUnit]:
        return self.shared_axis_coupling_engine.eligible_units(self)

    def _probe_root_pruning_indices(self, unit: PruningUnit) -> list[tuple[int, ...]]:
        return self.shared_axis_coupling_engine.probe_root_pruning_indices(unit)

    def _canonical_group_signature(
        self, unit: PruningUnit, prune_indices: Sequence[int]
    ) -> set[tuple[str, str, tuple[int, ...]]]:
        return self.shared_axis_coupling_engine.canonical_group_signature(
            self, unit, list(prune_indices)
        )

    def _discover_shared_axis_groups(self) -> list[CoupledUnitGroup]:
        return self.shared_axis_coupling_engine.discover_groups(self)

    def _mamba_taylor(self, module: Mamba) -> torch.Tensor:
        score = torch.zeros(module.d_inner, device=module.out_proj.weight.device)

        if module.in_proj.weight.grad is not None:
            weight_score = (
                (module.in_proj.weight.detach() * module.in_proj.weight.grad.detach())
                .abs()
                .sum(dim=1)
            )
            score += weight_score[: module.d_inner] + weight_score[module.d_inner :]
        if module.in_proj.bias is not None and module.in_proj.bias.grad is not None:
            bias_score = (
                module.in_proj.bias.detach() * module.in_proj.bias.grad.detach()
            ).abs()
            score += bias_score[: module.d_inner] + bias_score[module.d_inner :]

        if module.conv1d.weight.grad is not None:
            score += (
                (module.conv1d.weight.detach() * module.conv1d.weight.grad.detach())
                .abs()
                .sum(dim=(1, 2))
            )
        if module.conv1d.bias is not None and module.conv1d.bias.grad is not None:
            score += (
                module.conv1d.bias.detach() * module.conv1d.bias.grad.detach()
            ).abs()

        if module.x_proj.weight.grad is not None:
            score += (
                (module.x_proj.weight.detach() * module.x_proj.weight.grad.detach())
                .abs()
                .sum(dim=0)
            )

        if module.dt_proj.weight.grad is not None:
            score += (
                (module.dt_proj.weight.detach() * module.dt_proj.weight.grad.detach())
                .abs()
                .sum(dim=1)
            )
        if module.dt_proj.bias is not None and module.dt_proj.bias.grad is not None:
            score += (
                module.dt_proj.bias.detach() * module.dt_proj.bias.grad.detach()
            ).abs()

        if module.A_log.grad is not None:
            score += (
                (module.A_log.detach() * module.A_log.grad.detach()).abs().sum(dim=1)
            )
        if module.D.grad is not None:
            score += (module.D.detach() * module.D.grad.detach()).abs()

        if module.out_proj.weight.grad is not None:
            score += (
                (module.out_proj.weight.detach() * module.out_proj.weight.grad.detach())
                .abs()
                .sum(dim=0)
            )
        return score

    def _mamba_fisher(self, module: Mamba) -> torch.Tensor:
        score = torch.zeros(module.d_inner, device=module.out_proj.weight.device)

        if module.in_proj.weight.grad is not None:
            weight_score = (
                0.5
                * module.in_proj.weight.detach().pow(2)
                * module.in_proj.weight.grad.detach().pow(2)
            )
            weight_score = weight_score.sum(dim=1)
            score += weight_score[: module.d_inner] + weight_score[module.d_inner :]
        if module.in_proj.bias is not None and module.in_proj.bias.grad is not None:
            bias_score = (
                0.5
                * module.in_proj.bias.detach().pow(2)
                * module.in_proj.bias.grad.detach().pow(2)
            )
            score += bias_score[: module.d_inner] + bias_score[module.d_inner :]

        if module.conv1d.weight.grad is not None:
            score += 0.5 * (
                module.conv1d.weight.detach().pow(2)
                * module.conv1d.weight.grad.detach().pow(2)
            ).sum(dim=(1, 2))
        if module.conv1d.bias is not None and module.conv1d.bias.grad is not None:
            score += (
                0.5
                * module.conv1d.bias.detach().pow(2)
                * module.conv1d.bias.grad.detach().pow(2)
            )

        if module.x_proj.weight.grad is not None:
            score += 0.5 * (
                module.x_proj.weight.detach().pow(2)
                * module.x_proj.weight.grad.detach().pow(2)
            ).sum(dim=0)

        if module.dt_proj.weight.grad is not None:
            score += 0.5 * (
                module.dt_proj.weight.detach().pow(2)
                * module.dt_proj.weight.grad.detach().pow(2)
            ).sum(dim=1)
        if module.dt_proj.bias is not None and module.dt_proj.bias.grad is not None:
            score += (
                0.5
                * module.dt_proj.bias.detach().pow(2)
                * module.dt_proj.bias.grad.detach().pow(2)
            )

        if module.A_log.grad is not None:
            score += 0.5 * (
                module.A_log.detach().pow(2) * module.A_log.grad.detach().pow(2)
            ).sum(dim=1)
        if module.D.grad is not None:
            score += 0.5 * module.D.detach().pow(2) * module.D.grad.detach().pow(2)

        if module.out_proj.weight.grad is not None:
            score += 0.5 * (
                module.out_proj.weight.detach().pow(2)
                * module.out_proj.weight.grad.detach().pow(2)
            ).sum(dim=0)
        return score

    def _swiglu_taylor(self, module: SwiGLUFeedForward) -> torch.Tensor:
        hidden_dim = module.w3.in_features
        score = torch.zeros(hidden_dim, device=module.w3.weight.device)
        if module.w12.weight.grad is not None:
            weight_score = (
                (module.w12.weight.detach() * module.w12.weight.grad.detach())
                .abs()
                .sum(dim=1)
            )
            score += weight_score[:hidden_dim] + weight_score[hidden_dim:]
        if module.w12.bias is not None and module.w12.bias.grad is not None:
            bias_score = (
                module.w12.bias.detach() * module.w12.bias.grad.detach()
            ).abs()
            score += bias_score[:hidden_dim] + bias_score[hidden_dim:]
        if module.w3.weight.grad is not None:
            score += (
                (module.w3.weight.detach() * module.w3.weight.grad.detach())
                .abs()
                .sum(dim=0)
            )
        return score

    def _swiglu_fisher(self, module: SwiGLUFeedForward) -> torch.Tensor:
        hidden_dim = module.w3.in_features
        score = torch.zeros(hidden_dim, device=module.w3.weight.device)
        if module.w12.weight.grad is not None:
            weight_score = module.w12.weight.detach().pow(
                2
            ) * module.w12.weight.grad.detach().pow(2)
            weight_score = 0.5 * weight_score.sum(dim=1)
            score += weight_score[:hidden_dim] + weight_score[hidden_dim:]
        if module.w12.bias is not None and module.w12.bias.grad is not None:
            bias_score = (
                0.5
                * module.w12.bias.detach().pow(2)
                * module.w12.bias.grad.detach().pow(2)
            )
            score += bias_score[:hidden_dim] + bias_score[hidden_dim:]
        if module.w3.weight.grad is not None:
            score += 0.5 * (
                module.w3.weight.detach().pow(2) * module.w3.weight.grad.detach().pow(2)
            ).sum(dim=0)
        return score

    def _mlp_taylor(self, module: FeedForwardMLP) -> torch.Tensor:
        hidden_dim = module.layers[0].out_features
        score = torch.zeros(hidden_dim, device=module.layers[0].weight.device)

        first = module.layers[0]
        if first.weight.grad is not None:
            score += (
                (first.weight.detach() * first.weight.grad.detach()).abs().sum(dim=1)
            )
        if first.bias is not None and first.bias.grad is not None:
            score += (first.bias.detach() * first.bias.grad.detach()).abs()

        for hidden_layer in module.layers[1:-1]:
            if hidden_layer.weight.grad is not None:
                weight_score = (
                    hidden_layer.weight.detach() * hidden_layer.weight.grad.detach()
                ).abs()
                score += weight_score.sum(dim=1) + weight_score.sum(dim=0)
            if hidden_layer.bias is not None and hidden_layer.bias.grad is not None:
                score += (
                    hidden_layer.bias.detach() * hidden_layer.bias.grad.detach()
                ).abs()

        last = module.layers[-1]
        if last.weight.grad is not None:
            score += (last.weight.detach() * last.weight.grad.detach()).abs().sum(dim=0)
        return score

    def _mlp_fisher(self, module: FeedForwardMLP) -> torch.Tensor:
        hidden_dim = module.layers[0].out_features
        score = torch.zeros(hidden_dim, device=module.layers[0].weight.device)

        first = module.layers[0]
        if first.weight.grad is not None:
            score += 0.5 * (
                first.weight.detach().pow(2) * first.weight.grad.detach().pow(2)
            ).sum(dim=1)
        if first.bias is not None and first.bias.grad is not None:
            score += 0.5 * first.bias.detach().pow(2) * first.bias.grad.detach().pow(2)

        for hidden_layer in module.layers[1:-1]:
            if hidden_layer.weight.grad is not None:
                weight_score = (
                    0.5
                    * hidden_layer.weight.detach().pow(2)
                    * hidden_layer.weight.grad.detach().pow(2)
                )
                score += weight_score.sum(dim=1) + weight_score.sum(dim=0)
            if hidden_layer.bias is not None and hidden_layer.bias.grad is not None:
                score += (
                    0.5
                    * hidden_layer.bias.detach().pow(2)
                    * hidden_layer.bias.grad.detach().pow(2)
                )

        last = module.layers[-1]
        if last.weight.grad is not None:
            score += 0.5 * (
                last.weight.detach().pow(2) * last.weight.grad.detach().pow(2)
            ).sum(dim=0)
        return score

    def score_units(
        self, calibration_loader: Iterable, calibration_batches: int = 64
    ) -> list[PruningUnit]:
        if not self.units:
            self.collect_units()

        hooks = []
        cache: dict[str, torch.Tensor] = {}
        stats = {
            unit.name: {
                "energy": torch.zeros(unit.channels),
                "taylor": torch.zeros(unit.channels),
                "fisher": torch.zeros(unit.channels),
                "samples": [],
                "targets": [],
            }
            for unit in self.units
        }

        def make_forward_hook(unit: PruningUnit):
            def hook(_module, _inputs, output):
                cache[unit.name] = self._project_activation(unit, output).detach()

            return hook

        def make_pre_hook(unit: PruningUnit):
            def hook(_module, inputs):
                cache[unit.name] = self._project_activation(unit, inputs).detach()

            return hook

        for unit in self.units:
            if unit.hook_type == "pre":
                hooks.append(
                    unit.activation_module.register_forward_pre_hook(
                        make_pre_hook(unit)
                    )
                )
            else:
                hooks.append(
                    unit.activation_module.register_forward_hook(
                        make_forward_hook(unit)
                    )
                )

        self.model.train()
        self.criterion.train()
        self._log(
            f"scoring {len(self.units)} units on up to {calibration_batches} calibration batches "
            f"(score_mode={self.score_mode}, lambda_h={self.lambda_h}, lambda_i={self.lambda_i})..."
        )
        score_start_time = time.perf_counter()
        processed_batches = 0
        with mamba_fast_path_disabled(self.model):
            for batch_index, (samples, targets) in enumerate(calibration_loader):
                if batch_index >= calibration_batches:
                    break

                samples = samples.to(self.device)
                targets = [
                    {key: value.to(self.device) for key, value in target.items()}
                    for target in targets
                ]

                self.model.zero_grad(set_to_none=True)
                outputs = self.model(samples, targets=targets)
                target_summary = prediction_summary(outputs).cpu()

                loss_dict = self.criterion(outputs, targets, epoch=0)
                loss = sum(loss_dict.values())
                loss.backward()
                processed_batches = batch_index + 1

                for unit in self.units:
                    activation = cache.get(unit.name)
                    if activation is None:
                        raise RuntimeError(
                            f"Missing activation cache for unit {unit.name}"
                        )
                    unit.activation_shape = tuple(int(dim) for dim in activation.shape)
                    energy, sample_summary = self._channel_energy(
                        activation, unit.channel_dim
                    )
                    stats[unit.name]["energy"] += energy.cpu()
                    stats[unit.name]["samples"].append(sample_summary.cpu())
                    stats[unit.name]["targets"].append(target_summary)
                    if self.score_mode == FAMILY_SPECIFIC_SCORE_MODE:
                        if unit.kind in FISHER_UNIT_KINDS:
                            stats[unit.name]["fisher"] += (
                                self._compute_unit_fisher(unit).detach().cpu()
                            )
                        else:
                            stats[unit.name]["taylor"] += (
                                self._compute_unit_taylor(unit).detach().cpu()
                            )
                    else:
                        stats[unit.name]["taylor"] += (
                            self._compute_unit_taylor(unit).detach().cpu()
                        )

                cache.clear()
                if (
                    processed_batches == 1
                    or processed_batches % self.progress_interval == 0
                    or processed_batches == calibration_batches
                ):
                    elapsed = time.perf_counter() - score_start_time
                    self._log(
                        f"calibration batch {processed_batches}/{calibration_batches} "
                        f"completed (loss={loss.item():.4f}, elapsed={elapsed:.1f}s)."
                    )

        for hook in hooks:
            hook.remove()
        self.model.zero_grad(set_to_none=True)
        gc.collect()

        for unit in self.units:
            unit_stats = stats[unit.name]
            entropy = normalized_entropy(unit_stats["energy"])
            sample_summary = (
                torch.cat(unit_stats["samples"])
                if unit_stats["samples"]
                else torch.zeros(0)
            )
            target_summary = (
                torch.cat(unit_stats["targets"])
                if unit_stats["targets"]
                else torch.zeros(0)
            )
            mi = (
                normalized_mutual_information(sample_summary, target_summary)
                if sample_summary.numel()
                else 0.0
            )
            batch_count = max(1, min(calibration_batches, len(unit_stats["samples"])))
            base_score = self._base_score_from_stats(
                unit, unit_stats, batch_count=batch_count
            )
            if self.score_mode == "taylor":
                score = base_score
            elif self.score_mode == "taylor_entropy":
                score = base_score * (1.0 + self.lambda_h * entropy)
            else:
                score = base_score * (
                    1.0 + self.lambda_h * entropy + self.lambda_i * mi
                )
            unit.score = score.float()
            unit.entropy = float(entropy)
            unit.mi = float(mi)
        elapsed = time.perf_counter() - score_start_time
        self._log(
            f"scoring complete in {elapsed:.1f}s over {processed_batches} batches."
        )
        return self.units

    def _rounded_keep_channels(self, unit: PruningUnit, ratio: float) -> int:
        keep = int(round(unit.channels * ratio / unit.group_size)) * unit.group_size
        keep = min(unit.channels, keep)
        keep = max(unit.min_channels, keep)
        return keep

    def _estimate_module_action(
        self,
        module: nn.Module,
        handler_name: str,
        prune_count: int,
        output_shape: Sequence[int] | None,
    ) -> tuple[int, float]:
        if prune_count <= 0:
            return 0, 0.0

        if isinstance(module, nn.Conv2d):
            kernel_h, kernel_w = module.kernel_size
            out_h, out_w = _conv_output_hw(output_shape or ())
            if handler_name == "prune_out_channels":
                if module.groups == module.in_channels == module.out_channels:
                    param_delta = prune_count * kernel_h * kernel_w
                    flops_delta = (
                        2.0 * prune_count * out_h * out_w * kernel_h * kernel_w
                    )
                else:
                    input_per_output = module.in_channels // module.groups
                    param_delta = prune_count * input_per_output * kernel_h * kernel_w
                    flops_delta = (
                        2.0
                        * prune_count
                        * out_h
                        * out_w
                        * input_per_output
                        * kernel_h
                        * kernel_w
                    )
                if module.bias is not None:
                    param_delta += prune_count
                return int(param_delta), _flops_to_gflops(flops_delta)

            if handler_name == "prune_in_channels":
                if module.groups == module.in_channels == module.out_channels:
                    param_delta = prune_count * kernel_h * kernel_w
                    flops_delta = (
                        2.0 * prune_count * out_h * out_w * kernel_h * kernel_w
                    )
                else:
                    removed_per_group = (
                        prune_count // module.groups
                        if module.groups > 1
                        else prune_count
                    )
                    param_delta = (
                        module.out_channels * removed_per_group * kernel_h * kernel_w
                    )
                    flops_delta = (
                        2.0
                        * module.out_channels
                        * out_h
                        * out_w
                        * removed_per_group
                        * kernel_h
                        * kernel_w
                    )
                return int(param_delta), _flops_to_gflops(flops_delta)

        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            param_delta = 2 * prune_count if module.affine else 0
            if output_shape is None:
                return int(param_delta), 0.0
            elements = int(np.prod(output_shape[2:])) if len(output_shape) > 2 else 1
            flops_delta = 2.0 * output_shape[0] * elements * prune_count
            return int(param_delta), _flops_to_gflops(flops_delta)

        if isinstance(module, nn.Linear):
            if output_shape is None:
                token_count = 1
            else:
                token_count = _count_tokens(output_shape)
            if handler_name == "prune_out_channels":
                param_delta = prune_count * module.in_features + (
                    prune_count if module.bias is not None else 0
                )
                flops_delta = 2.0 * token_count * module.in_features * prune_count
                return int(param_delta), _flops_to_gflops(flops_delta)
            if handler_name == "prune_in_channels":
                param_delta = prune_count * module.out_features
                flops_delta = 2.0 * token_count * module.out_features * prune_count
                return int(param_delta), _flops_to_gflops(flops_delta)

        if isinstance(module, nn.LayerNorm):
            param_delta = 2 * prune_count if module.elementwise_affine else 0
            if output_shape is None:
                return int(param_delta), 0.0
            token_count = _count_tokens(output_shape)
            flops_delta = 2.0 * token_count * prune_count
            return int(param_delta), _flops_to_gflops(flops_delta)

        return 0, 0.0

    def _estimate_mamba_delta(
        self, module: Mamba, prune_count: int, activation_shape: Sequence[int]
    ) -> tuple[int, float]:
        token_count = _count_tokens(activation_shape)
        param_delta = 0
        param_delta += 2 * prune_count * module.d_model
        if module.in_proj.bias is not None:
            param_delta += 2 * prune_count
        param_delta += prune_count * module.conv1d.kernel_size[0]
        if module.conv1d.bias is not None:
            param_delta += prune_count
        param_delta += prune_count * module.x_proj.out_features
        param_delta += prune_count * module.dt_proj.in_features
        if module.dt_proj.bias is not None:
            param_delta += prune_count
        param_delta += prune_count * module.A_log.shape[1]
        param_delta += prune_count
        param_delta += prune_count * module.out_proj.out_features

        flops_delta = 0.0
        flops_delta += 2.0 * token_count * module.d_model * (2 * prune_count)
        flops_delta += 2.0 * token_count * prune_count * module.conv1d.kernel_size[0]
        flops_delta += 2.0 * token_count * module.x_proj.out_features * prune_count
        flops_delta += 2.0 * token_count * module.dt_proj.in_features * prune_count
        flops_delta += 2.0 * token_count * module.out_proj.out_features * prune_count
        return int(param_delta), _flops_to_gflops(flops_delta)

    def _estimate_swiglu_delta(
        self,
        module: SwiGLUFeedForward,
        prune_count: int,
        activation_shape: Sequence[int],
    ) -> tuple[int, float]:
        token_count = _count_tokens(activation_shape)
        param_delta = 2 * prune_count * module.w12.in_features
        if module.w12.bias is not None:
            param_delta += 2 * prune_count
        param_delta += prune_count * module.w3.out_features

        flops_delta = 2.0 * token_count * module.w12.in_features * (2 * prune_count)
        flops_delta += 2.0 * token_count * module.w3.out_features * prune_count
        flops_delta += 2.0 * token_count * prune_count
        return int(param_delta), _flops_to_gflops(flops_delta)

    def _estimate_mlp_delta(
        self, module: FeedForwardMLP, prune_count: int, activation_shape: Sequence[int]
    ) -> tuple[int, float]:
        token_count = _count_tokens(activation_shape)
        hidden_dim = module.layers[0].out_features
        param_delta = module.layers[0].in_features * prune_count + (
            prune_count if module.layers[0].bias is not None else 0
        )
        for hidden_layer in module.layers[1:-1]:
            param_delta += 2 * hidden_dim * prune_count - prune_count * prune_count
            if hidden_layer.bias is not None:
                param_delta += prune_count
        param_delta += module.layers[-1].out_features * prune_count
        flops_delta = 2.0 * token_count * param_delta
        return int(param_delta), _flops_to_gflops(flops_delta)

    def _resolve_wavelet_fuse_consumers(
        self, module_name: str, module_names: dict[str, nn.Module] | None = None
    ) -> list[tuple[str, nn.Conv2d]]:
        if ".downsample_convs." not in module_name:
            raise ValueError(
                f"Unsupported WaveletDownsample fuse module path: {module_name}"
            )

        module_names = (
            dict(self.model.named_modules()) if module_names is None else module_names
        )
        prefix, suffix = module_name.split(".downsample_convs.", 1)
        idx_str, _ = suffix.split(".fuse.conv", 1)
        block_prefix = f"{prefix}.pan_blocks.{idx_str}"

        consumer_names: list[str] = []
        if f"{block_prefix}.cv1.conv" in module_names:
            consumer_names.append(f"{block_prefix}.cv1.conv")
        elif (
            f"{block_prefix}.conv1.conv" in module_names
            and f"{block_prefix}.conv2.conv" in module_names
        ):
            consumer_names.extend(
                [f"{block_prefix}.conv1.conv", f"{block_prefix}.conv2.conv"]
            )
        else:
            raise RuntimeError(
                f"Unable to resolve PAN consumer convs for WaveletDownsample fuse module {module_name}. "
                f"Expected RepNCSPELAN/CSPLayer-style input projections under {block_prefix}."
            )

        consumers: list[tuple[str, nn.Conv2d]] = []
        for consumer_name in consumer_names:
            consumer_module = module_names.get(consumer_name)
            if not isinstance(consumer_module, nn.Conv2d):
                raise RuntimeError(
                    f"Expected {consumer_name} to be nn.Conv2d for WaveletDownsample fuse pruning, "
                    f"got {type(consumer_module)}"
                )
            consumers.append((consumer_name, consumer_module))
        return consumers

    def _is_sum_fused_wavelet_output(
        self, module_name: str, module_names: dict[str, nn.Module] | None = None
    ) -> bool:
        if ".downsample_convs." not in module_name:
            return False
        module_names = (
            dict(self.model.named_modules()) if module_names is None else module_names
        )
        encoder_prefix = module_name.split(".downsample_convs.", 1)[0]
        encoder_module = module_names.get(encoder_prefix)
        if encoder_module is None:
            return False
        return getattr(encoder_module, "fuse_op", None) == "sum"

    def _can_use_direct_wavelet_fuse_path(
        self, unit: PruningUnit, module_names: dict[str, nn.Module] | None = None
    ) -> bool:
        if unit.kind != "wavelet_fusion":
            return False
        module_names = (
            dict(self.model.named_modules()) if module_names is None else module_names
        )
        if self._is_sum_fused_wavelet_output(unit.module_name, module_names):
            return False
        try:
            self._resolve_wavelet_fuse_consumers(unit.module_name, module_names)
        except Exception:
            return False
        return True

    def _estimate_wavelet_fuse_delta(
        self, unit: PruningUnit, prune_indices: Sequence[int]
    ) -> tuple[int, float]:
        prune_count = len(prune_indices)
        if prune_count <= 0:
            return 0, 0.0

        module_names = dict(self.model.named_modules())
        if not self._can_use_direct_wavelet_fuse_path(unit, module_names):
            raise RuntimeError(
                f"WaveletDownsample fuse direct estimate is not available for {unit.module_name}"
            )

        delta_params = 0
        delta_gflops = 0.0

        conv_output_shape = self.output_shape_cache.get(unit.module_name)
        module_params, module_gflops = self._estimate_module_action(
            unit.module,
            "prune_out_channels",
            prune_count,
            conv_output_shape,
        )
        delta_params += module_params
        delta_gflops += module_gflops

        bn_name = unit.module_name.rsplit(".", 1)[0] + ".bn"
        bn_module = module_names.get(bn_name)
        if isinstance(bn_module, nn.BatchNorm2d):
            module_params, module_gflops = self._estimate_module_action(
                bn_module,
                "prune_out_channels",
                prune_count,
                self.output_shape_cache.get(bn_name, conv_output_shape),
            )
            delta_params += module_params
            delta_gflops += module_gflops

        for consumer_name, consumer_module in self._resolve_wavelet_fuse_consumers(
            unit.module_name, module_names
        ):
            module_params, module_gflops = self._estimate_module_action(
                consumer_module,
                "prune_in_channels",
                prune_count,
                self.output_shape_cache.get(consumer_name),
            )
            delta_params += module_params
            delta_gflops += module_gflops

        return int(delta_params), float(delta_gflops)

    def _apply_wavelet_fuse_pruning(
        self, unit: PruningUnit, prune_indices: Sequence[int]
    ) -> None:
        module_names = dict(self.model.named_modules())
        if not self._can_use_direct_wavelet_fuse_path(unit, module_names):
            raise RuntimeError(
                f"WaveletDownsample fuse direct apply is not available for {unit.module_name}"
            )

        conv_module = module_names.get(unit.module_name)
        if not isinstance(conv_module, nn.Conv2d):
            raise RuntimeError(
                f"Expected {unit.module_name} to resolve to nn.Conv2d during WaveletDownsample fuse apply, "
                f"got {type(conv_module)}"
            )
        tp.prune_conv_out_channels(conv_module, list(prune_indices))

        bn_name = unit.module_name.rsplit(".", 1)[0] + ".bn"
        bn_module = module_names.get(bn_name)
        if isinstance(bn_module, nn.BatchNorm2d):
            tp.prune_batchnorm_out_channels(bn_module, list(prune_indices))

        for _consumer_name, consumer_module in self._resolve_wavelet_fuse_consumers(
            unit.module_name, module_names
        ):
            tp.prune_conv_in_channels(consumer_module, list(prune_indices))

    def _estimate_candidate_delta(
        self, unit: PruningUnit, prune_indices: Sequence[int]
    ) -> tuple[int, float]:
        prune_count = len(prune_indices)
        if prune_count == 0:
            return 0, 0.0
        return self.unit_kind_handlers.estimate_candidate_delta(
            self, unit, list(prune_indices)
        )

    def _prune_indices_are_structurally_valid(
        self, unit: PruningUnit, prune_indices: Sequence[int]
    ) -> bool:
        if not prune_indices:
            return True
        if unit.kind not in {
            "backbone_conv",
            "neck_conv",
            "wavelet_spatial",
            "wavelet_mask",
        }:
            return True
        if self.dependency_graph is None:
            raise RuntimeError(
                "Dependency graph must be built before validating pruning indices."
            )
        handler = self.unit_kind_handlers.get(unit.kind)
        if not hasattr(handler, "validate_pruning_group_indices"):
            return True
        group = self.dependency_graph.get_pruning_group(
            unit.module, unit.pruning_fn, idxs=list(prune_indices)
        )
        if not self.dependency_graph.check_pruning_group(group):
            return False
        try:
            handler.validate_pruning_group_indices(self, unit, group)
        except RuntimeError:
            return False
        return True

    def _prune_indices_overflow_dependency_axes(
        self, unit: PruningUnit, prune_indices: Sequence[int]
    ) -> bool:
        if not prune_indices:
            return False
        if unit.kind not in {
            "backbone_conv",
            "neck_conv",
            "wavelet_spatial",
            "wavelet_mask",
        }:
            return False
        if self.dependency_graph is None:
            raise RuntimeError(
                "Dependency graph must be built before validating pruning indices."
            )
        handler = self.unit_kind_handlers.get(unit.kind)
        logical_axis_size = getattr(handler, "_logical_axis_size", None)
        select_axis_size = getattr(handler, "_select_axis_size", None)
        torch_pruning_keep_indices = getattr(
            handler, "_torch_pruning_keep_indices", None
        )
        if (
            logical_axis_size is None
            or select_axis_size is None
            or torch_pruning_keep_indices is None
        ):
            return False

        group = self.dependency_graph.get_pruning_group(
            unit.module, unit.pruning_fn, idxs=list(prune_indices)
        )
        if not self.dependency_graph.check_pruning_group(group):
            return True
        for dependency, dep_indices in group.items:
            target = dependency.target.module
            handler_name = dependency.handler.__name__
            logical_size = logical_axis_size(target, handler_name)
            select_size = select_axis_size(target, handler_name)
            if logical_size is None or select_size is None:
                continue
            indices = [int(getattr(index, "idx", index)) for index in dep_indices]
            if not indices:
                continue
            keep_indices = torch_pruning_keep_indices(target, handler_name, indices)
            keep_max = max(keep_indices) if keep_indices else -1
            if (
                min(indices) < 0
                or max(indices) >= logical_size
                or not keep_indices
                or keep_max >= select_size
            ):
                return True
        return False

    def _select_valid_prune_indices(
        self,
        unit: PruningUnit,
        sorted_indices: Sequence[int],
        prune_count: int,
    ) -> list[int]:
        if prune_count <= 0:
            return []

        initial = sorted(int(index) for index in sorted_indices[:prune_count])
        if self._prune_indices_are_structurally_valid(unit, initial):
            return initial

        safe_ranked_indices: list[int] = []
        safe_index_set: set[int] = set()
        for index in sorted_indices:
            index = int(index)
            if index in safe_index_set:
                continue
            if self._prune_indices_overflow_dependency_axes(unit, [index]):
                continue
            safe_ranked_indices.append(index)
            safe_index_set.add(index)
            if len(safe_ranked_indices) == prune_count:
                break

        proposal = sorted(safe_ranked_indices[:prune_count])
        if proposal and self._prune_indices_are_structurally_valid(unit, proposal):
            return proposal

        selected: list[int] = []
        for index in safe_ranked_indices:
            candidate = sorted(selected + [index])
            if self._prune_indices_are_structurally_valid(unit, candidate):
                selected.append(index)
        best = sorted(selected)
        best_count = len(best)
        if best_count == 0:
            return []
        self._log(
            f"candidate for {unit.name} requested {prune_count} channels, but only {best_count} "
            "could be pruned without overflowing downstream dependency axes."
        )
        return best

    def _unit_by_name(self, unit_name: str) -> PruningUnit:
        for unit in self.units:
            if unit.name == unit_name:
                return unit
        raise KeyError(unit_name)

    def _collect_dependency_actions(
        self, unit: PruningUnit, prune_indices: Sequence[int]
    ) -> dict[tuple[int, str], dict[str, Any]]:
        if not prune_indices:
            return {}
        group = self.dependency_graph.get_pruning_group(
            unit.module, unit.pruning_fn, idxs=list(prune_indices)
        )
        actions: dict[tuple[int, str], dict[str, Any]] = {}
        for dependency, dep_indices in group.items:
            target = dependency.target.module
            target_name = self.module_name_map.get(id(target))
            if target_name is None:
                continue
            key = (id(target), dependency.handler.__name__)
            payload = actions.setdefault(
                key,
                {
                    "target": target,
                    "target_name": target_name,
                    "handler_name": dependency.handler.__name__,
                    "indices": set(),
                },
            )
            payload["indices"].update(int(index) for index in dep_indices)
        return actions

    def _estimate_member_candidate_union_delta(
        self, member_candidates: dict[str, UnitCandidate]
    ) -> tuple[int, float]:
        combined_actions: dict[tuple[int, str], dict[str, Any]] = {}
        for unit_name, candidate in member_candidates.items():
            if not candidate.prune_indices:
                continue
            unit = self._unit_by_name(unit_name)
            for key, payload in self._collect_dependency_actions(
                unit, candidate.prune_indices
            ).items():
                target_payload = combined_actions.setdefault(
                    key,
                    {
                        "target": payload["target"],
                        "target_name": payload["target_name"],
                        "handler_name": payload["handler_name"],
                        "indices": set(),
                    },
                )
                target_payload["indices"].update(payload["indices"])

        delta_params = 0
        delta_gflops = 0.0
        for payload in combined_actions.values():
            output_shape = self.output_shape_cache.get(payload["target_name"])
            module_params, module_gflops = self._estimate_module_action(
                payload["target"],
                payload["handler_name"],
                len(payload["indices"]),
                output_shape,
            )
            delta_params += module_params
            delta_gflops += module_gflops
        return int(delta_params), float(delta_gflops)

    def _materialize_member_candidate_smoke(
        self, member_candidates: dict[str, UnitCandidate]
    ) -> bool:
        try:
            selected = {
                unit.name: member_candidates.get(
                    unit.name, self._identity_candidate(unit)
                )
                for unit in self.units
            }
            materialized = self.materialize_selection(
                selected,
                dense_model=clone_pruner_model_to_cpu(self),
                dense_stats=None,
                validate_forward=True,
            )
        except Exception as exc:
            raise_if_cuda_device_assert(
                self,
                selected,
                exc,
                phase="coupled_candidate_smoke",
            )
            return False
        self._dispose_materialized(materialized)
        return True

    def _build_coupled_allocation(self, group: CoupledUnitGroup) -> AllocationObject:
        return self.shared_axis_coupling_engine.build_coupled_allocation(self, group)

    def _build_allocation_objects(self) -> list[AllocationObject]:
        allocation_objects = self.shared_axis_coupling_engine.build_allocation_objects(
            self
        )
        allocation_objects.extend(build_block_gate_allocation_objects(self))
        self.allocation_objects = allocation_objects
        return self.allocation_objects

    def build_candidates(self) -> list[PruningUnit]:
        if any(unit.score is None for unit in self.units):
            raise RuntimeError("Call score_units(...) before build_candidates().")

        self._log("building legal pruning candidates...")
        total_units = len(self.units)
        build_start_time = time.perf_counter()
        for unit_index, unit in enumerate(self.units, start=1):
            unit_start_time = time.perf_counter()
            sorted_indices = torch.argsort(unit.score).tolist()
            candidates: list[UnitCandidate] = []
            seen_keep: set[int] = set()
            total_candidate_ratios = len(self.candidate_ratios)
            for candidate_index, ratio in enumerate(self.candidate_ratios, start=1):
                keep_channels = self._rounded_keep_channels(unit, ratio)
                if keep_channels in seen_keep:
                    continue
                prune_count = unit.channels - keep_channels
                prune_indices = self._select_valid_prune_indices(
                    unit,
                    sorted_indices,
                    prune_count,
                )
                if prune_count > 0 and len(prune_indices) != prune_count:
                    keep_channels = unit.channels - len(prune_indices)
                if keep_channels in seen_keep:
                    continue
                seen_keep.add(keep_channels)
                if (
                    self.progress_interval == 1
                    and prune_count > 0
                    and (
                        candidate_index == 1
                        or candidate_index == total_candidate_ratios
                    )
                ):
                    self._log(
                        f"estimating candidate {candidate_index}/{total_candidate_ratios} for "
                        f"unit {unit_index}/{total_units}: {unit.name} "
                        f"(keep={keep_channels}/{unit.channels}, prune={prune_count})..."
                    )
                delta_params, delta_gflops = self._estimate_candidate_delta(
                    unit, prune_indices
                )
                cost = (
                    float(unit.score[prune_indices].sum().item())
                    if prune_indices
                    else 0.0
                )
                candidates.append(
                    UnitCandidate(
                        ratio=keep_channels / float(unit.channels),
                        keep_channels=keep_channels,
                        prune_indices=prune_indices,
                        delta_params=delta_params,
                        delta_gflops=delta_gflops,
                        cost=cost,
                    )
                )
            unit.candidates = sorted(
                candidates, key=lambda candidate: candidate.keep_channels, reverse=True
            )
            if (
                unit_index == 1
                or unit_index % self.progress_interval == 0
                or unit_index == total_units
            ):
                self._log(
                    f"candidate unit {unit_index}/{total_units} complete: {unit.name} "
                    f"({len(unit.candidates)} candidates, elapsed={time.perf_counter() - unit_start_time:.1f}s, "
                    f"total={time.perf_counter() - build_start_time:.1f}s)."
                )
        total_candidates = sum(len(unit.candidates) for unit in self.units)
        self._build_allocation_objects()
        if self.runtime_diagnostics["shared_axis_group_count"] > 0:
            self._log(
                "shared-axis legalization: "
                f"coupled_groups={self.runtime_diagnostics['shared_axis_group_count']}, "
                f"discarded_coupled_candidates={self.runtime_diagnostics['coupled_candidates_discarded_pre_milp']}"
            )
        self._log(
            f"candidate enumeration complete ({total_candidates} candidates across {len(self.units)} units)."
        )
        return self.units

    def solve_allocation(
        self,
        dense_params: int,
        dense_gflops: float,
        target_param_ratio: float,
        target_gflops_ratio: float,
    ):
        if target_param_ratio <= 0.0 or target_gflops_ratio <= 0.0:
            raise ValueError("Both target ratios must be positive.")

        if not self.allocation_objects:
            self._build_allocation_objects()

        all_candidates: list[tuple[int, AllocationCandidate]] = []
        for allocation_index, allocation in enumerate(self.allocation_objects):
            if not allocation.candidates:
                raise RuntimeError(
                    f"Allocation object {allocation.name} has no candidate ratios."
                )
            for candidate in allocation.candidates:
                all_candidates.append((allocation_index, candidate))

        num_variables = len(all_candidates)
        objective = np.array(
            [candidate.cost for _, candidate in all_candidates], dtype=np.float64
        )
        integrality = np.ones(num_variables, dtype=np.int8)
        bounds = Bounds(np.zeros(num_variables), np.ones(num_variables))

        constraints: list[LinearConstraint] = []
        for allocation_index, allocation in enumerate(self.allocation_objects):
            coeffs = np.zeros(num_variables, dtype=np.float64)
            for candidate_index, (candidate_allocation_index, _) in enumerate(
                all_candidates
            ):
                if candidate_allocation_index == allocation_index:
                    coeffs[candidate_index] = 1.0
            constraints.append(LinearConstraint(coeffs, 1.0, 1.0))

        required_param_savings = max(
            0.0, dense_params - dense_params * float(target_param_ratio)
        )
        required_gflops_savings = max(
            0.0, dense_gflops - dense_gflops * float(target_gflops_ratio)
        )
        if required_param_savings > 0.0:
            constraints.append(
                LinearConstraint(
                    -np.array(
                        [candidate.delta_params for _, candidate in all_candidates],
                        dtype=np.float64,
                    ),
                    -np.inf,
                    -required_param_savings,
                )
            )
        if required_gflops_savings > 0.0:
            constraints.append(
                LinearConstraint(
                    -np.array(
                        [candidate.delta_gflops for _, candidate in all_candidates],
                        dtype=np.float64,
                    ),
                    -np.inf,
                    -required_gflops_savings,
                )
            )
        feasibility_summary = self._allocation_feasibility_summary(
            self.allocation_objects,
            dense_params=dense_params,
            dense_gflops=dense_gflops,
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
        )
        feasibility_text = self._format_allocation_feasibility_summary(
            feasibility_summary
        )
        self._log(f"candidate feasibility upper bound: {feasibility_text}")
        if (
            required_param_savings
            > float(feasibility_summary["max_param_savings"]) + 1e-9
            or required_gflops_savings
            > float(feasibility_summary["max_gflops_savings"]) + 1e-9
        ):
            raise RuntimeError(
                "MILP pruning allocation budget is unreachable by the current legal candidate set: "
                f"{feasibility_text}. "
                f"candidate_ratios={self.candidate_ratios}, "
                f"shared_axis_groups={self.runtime_diagnostics.get('shared_axis_group_count', 0)}, "
                f"discarded_coupled_candidates={self.runtime_diagnostics.get('coupled_candidates_discarded_pre_milp', 0)}"
            )

        self._log(
            "solving MILP allocation "
            f"(variables={num_variables}, units={len(self.allocation_objects)}, "
            f"required_param_savings={required_param_savings:.0f}, "
            f"required_gflops_savings={required_gflops_savings:.4f})..."
        )
        solve_start_time = time.perf_counter()
        result = milp(
            c=objective,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"disp": False},
        )
        if not result.success:
            raise RuntimeError(
                f"MILP pruning allocation failed: {result.message}; {feasibility_text}"
            )
        self._log(f"MILP solved in {time.perf_counter() - solve_start_time:.1f}s.")

        selected: dict[str, UnitCandidate] = {}
        for value, (allocation_index, candidate) in zip(result.x, all_candidates):
            if value > 0.5:
                selected.update(candidate.member_candidates)
        pruned_units = sum(
            1 for candidate in selected.values() if candidate.prune_indices
        )
        self._log(
            f"MILP selected ratios for {len(selected)} units; {pruned_units} units will be structurally pruned."
        )
        return selected

    def _rebuild_dependency_graph(self):
        was_training = self.model.training
        self.model.eval()
        dependency_graph = self._new_dependency_graph()
        unwrapped_parameters, ignored_parameters = self._collect_unwrapped_parameters(
            dependency_graph
        )
        with (
            eval_spatial_size_override(self.model, self.input_shape[-2:]),
            mamba_fast_path_disabled(self.model),
        ):
            self.dependency_graph = dependency_graph.build_dependency(
                self.model,
                example_inputs=self.example_inputs,
                forward_fn=self._dependency_forward,
                unwrapped_parameters=unwrapped_parameters,
                ignored_params=ignored_parameters,
                verbose=False,
            )
        self.model.train(was_training)

    def apply_pruning(self, selected: dict[str, UnitCandidate]) -> nn.Module:
        return materialize_apply_pruning(self, selected)

    def validate_cuda_forward(self) -> None:
        validate_cuda_forward_impl(self)

    def materialize_selection(
        self,
        selected: dict[str, UnitCandidate],
        *,
        dense_model: nn.Module | None = None,
        dense_stats: dict[str, Any] | None = None,
        validate_forward: bool = True,
    ) -> dict[str, Any]:
        return materialize_selection_impl(
            self,
            selected,
            dense_model=dense_model,
            dense_stats=dense_stats,
            validate_forward=validate_forward,
        )

    def _dispose_materialized(self, materialized: dict[str, Any] | None) -> None:
        dispose_materialized(materialized)

    def _budget_excess(
        self,
        materialized: dict[str, Any],
        *,
        dense_params: int,
        dense_gflops: float,
        target_param_ratio: float,
        target_gflops_ratio: float,
        param_metric: str,
    ) -> tuple[float, float]:
        return repair_budget_excess(
            materialized,
            dense_params=dense_params,
            dense_gflops=dense_gflops,
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
            param_metric=param_metric,
        )

    def _candidate_lookup(self, unit: PruningUnit, candidate: UnitCandidate) -> int:
        return repair_candidate_lookup(self, unit, candidate)

    def _repair_selection_with_actual_marginals(
        self,
        selected: dict[str, UnitCandidate],
        materialized: dict[str, Any],
        *,
        dense_model: nn.Module,
        dense_params: int,
        dense_gflops: float,
        target_param_ratio: float,
        target_gflops_ratio: float,
        param_metric: str,
    ) -> tuple[dict[str, UnitCandidate], dict[str, Any]]:
        return repair_selection_with_actual_marginals_impl(
            self,
            selected,
            materialized,
            dense_model=dense_model,
            dense_params=dense_params,
            dense_gflops=dense_gflops,
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
            param_metric=param_metric,
        )

    def _evaluate_actual_marginal_proposal(
        self,
        proposal: dict[str, UnitCandidate],
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
        return evaluate_actual_marginal_proposal_impl(
            self,
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

    def _search_pairwise_actual_marginal_step(
        self,
        current_selected: dict[str, UnitCandidate],
        *,
        invalid_step_units: Sequence[PruningUnit],
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
        tuple[PruningUnit, dict[str, UnitCandidate], dict[str, Any], float, float]
        | None,
        float,
    ]:
        return search_pairwise_actual_marginal_step_impl(
            self,
            current_selected,
            invalid_step_units=invalid_step_units,
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

    def _repair_invalid_selection_to_valid(
        self,
        selected: dict[str, UnitCandidate],
        *,
        dense_model: nn.Module,
        dense_stats: dict[str, Any],
    ) -> tuple[dict[str, UnitCandidate], dict[str, Any]]:
        return repair_invalid_selection_to_valid_impl(
            self,
            selected,
            dense_model=dense_model,
            dense_stats=dense_stats,
        )

    def repair_selection_to_budget(
        self,
        selected: dict[str, UnitCandidate],
        *,
        dense_params: int,
        dense_gflops: float,
        target_param_ratio: float,
        target_gflops_ratio: float,
        param_metric: str = "measured",
        max_rounds: int = 3,
    ) -> tuple[dict[str, UnitCandidate], dict[str, Any]]:
        return repair_selection_to_budget_impl(
            self,
            selected,
            dense_params=dense_params,
            dense_gflops=dense_gflops,
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
            param_metric=param_metric,
            max_rounds=max_rounds,
        )

    def prune(
        self,
        calibration_loader: Iterable,
        calibration_batches: int,
        target_param_ratio: float,
        target_gflops_ratio: float,
        artifact_dir: str | Path,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.runtime_diagnostics = {
            "shared_axis_group_count": 0,
            "coupled_candidates_discarded_pre_milp": 0,
            "block_gate_count": 0,
            "invalid_selection_repair_invocations": 0,
            "pairwise_actual_marginal_repair_invocations": 0,
        }
        self._log(
            f"start prune: device={self.device}, input_shape={self.input_shape}, "
            f"target_param_ratio={target_param_ratio}, target_gflops_ratio={target_gflops_ratio}"
        )

        dense_stats = _measure_model_complexity(
            self.model, self.input_shape, self.device
        )
        self._log(
            f"dense model stats: params={dense_stats['params_million']:.4f}M, "
            f"fwd_gflops={dense_stats['flops_value']:.4f}"
        )
        self.build_dependency_graph()
        if not self.units:
            self.collect_units()
        self.score_units(calibration_loader, calibration_batches=calibration_batches)
        self.build_candidates()

        selected = self.solve_allocation(
            dense_params=dense_stats["params"],
            dense_gflops=dense_stats["flops_value"],
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
        )
        selected, materialized = self.repair_selection_to_budget(
            selected,
            dense_params=dense_stats["params"],
            dense_gflops=dense_stats["flops_value"],
            target_param_ratio=target_param_ratio,
            target_gflops_ratio=target_gflops_ratio,
            param_metric="measured",
            max_rounds=3,
        )
        self.model = materialized["model"]
        pruned_stats = materialized["pruned_stats"]
        self._log(
            f"pruned model stats: params={pruned_stats['params_million']:.4f}M, "
            f"fwd_gflops={pruned_stats['flops_value']:.4f}"
        )

        param_budget = dense_stats["params"] * float(target_param_ratio)
        gflops_budget = dense_stats["flops_value"] * float(target_gflops_ratio)
        if pruned_stats["params"] > param_budget + 1e-6:
            raise RuntimeError(
                f"Pruned model exceeds parameter budget: {pruned_stats['params']} > {param_budget:.4f}"
            )
        if pruned_stats["flops_value"] > gflops_budget + 1e-6:
            raise RuntimeError(
                f"Pruned model exceeds GFLOPs budget: {pruned_stats['flops_value']:.4f} > {gflops_budget:.4f}"
            )

        selection_summary = {}
        for unit in self.units:
            candidate = selected[unit.name]
            selection_summary[unit.name] = {
                "scope": unit.scope,
                "kind": unit.kind,
                "module_name": unit.module_name,
                "channels": unit.channels,
                "selected_ratio": candidate.ratio,
                "keep_channels": candidate.keep_channels,
                "pruned_channels": len(candidate.prune_indices),
                "delta_params_est": candidate.delta_params,
                "delta_gflops_est": candidate.delta_gflops,
                "entropy": unit.entropy,
                "mi": unit.mi,
            }
        unit_names = set(selection_summary)
        for unit_name, candidate in selected.items():
            if unit_name in unit_names:
                continue
            metadata = getattr(candidate, "metadata", {})
            if metadata.get("kind") != "block_gate":
                continue
            selection_summary[unit_name] = {
                "scope": metadata.get("scope", "unknown"),
                "kind": "block_gate",
                "module_name": metadata.get("module_name", unit_name),
                "channels": 1,
                "selected_ratio": candidate.ratio,
                "keep_channels": candidate.keep_channels,
                "pruned_channels": len(candidate.prune_indices),
                "delta_params_est": candidate.delta_params,
                "delta_gflops_est": candidate.delta_gflops,
                "entropy": 0.0,
                "mi": 0.0,
                "materialize_supported": getattr(
                    candidate, "materialize_supported", False
                ),
            }

        summary = {
            "dense": dense_stats,
            "pruned": pruned_stats,
            "target_param_ratio": float(target_param_ratio),
            "target_gflops_ratio": float(target_gflops_ratio),
            "candidate_ratios": self.candidate_ratios,
            "lambda_h": self.lambda_h,
            "lambda_i": self.lambda_i,
            "score_mode": self.score_mode,
            "model_metadata": collect_state_space_model_metadata(self.model),
            "runtime_diagnostics": dict(self.runtime_diagnostics),
            "units": selection_summary,
        }

        manifest_payload = dict(manifest or {})
        artifact_paths = save_pruned_artifact(
            self.model, artifact_dir, manifest_payload, summary
        )
        self._log(f"artifact saved to {artifact_dir}.")
        return {
            "artifact_paths": {
                key: str(value) for key, value in artifact_paths.items()
            },
            "summary": summary,
            "selected": selected,
            "units": self.units,
        }
