from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .pruner import TRIMPruner


class UnitKindHandler:
    def compute_taylor(self, pruner: "TRIMPruner", unit: Any) -> torch.Tensor:
        raise NotImplementedError

    def estimate_candidate_delta(
        self, pruner: "TRIMPruner", unit: Any, prune_indices: list[int]
    ) -> tuple[int, float]:
        raise NotImplementedError

    def apply_pruning(
        self, pruner: "TRIMPruner", unit: Any, candidate: Any
    ) -> str | None:
        raise NotImplementedError

    def supports_shared_axis_coupling(self, unit: Any) -> bool:
        return False


class DepGraphConvHandler(UnitKindHandler):
    @staticmethod
    def _select_axis_size(module: nn.Module, handler_name: str) -> int | None:
        if isinstance(module, nn.Conv2d):
            if handler_name == "prune_in_channels":
                dim = 0 if module.transposed else 1
                return int(module.weight.shape[dim])
            if handler_name == "prune_out_channels":
                dim = 1 if module.transposed else 0
                return int(module.weight.shape[dim])
        if isinstance(module, nn.Linear):
            if handler_name == "prune_in_channels":
                return int(module.weight.shape[1])
            if handler_name == "prune_out_channels":
                return int(module.weight.shape[0])
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if handler_name in {"prune_in_channels", "prune_out_channels"}:
                return int(module.num_features)
        return None

    @staticmethod
    def _logical_axis_size(module: nn.Module, handler_name: str) -> int | None:
        if isinstance(module, nn.Conv2d):
            if handler_name == "prune_in_channels":
                return int(module.in_channels)
            if handler_name == "prune_out_channels":
                return int(module.out_channels)
        if isinstance(module, nn.Linear):
            if handler_name == "prune_in_channels":
                return int(module.in_features)
            if handler_name == "prune_out_channels":
                return int(module.out_features)
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if handler_name in {"prune_in_channels", "prune_out_channels"}:
                return int(module.num_features)
        return None

    @staticmethod
    def _torch_pruning_keep_indices(
        module: nn.Module, handler_name: str, indices: list[int]
    ) -> list[int] | None:
        logical_size = DepGraphConvHandler._logical_axis_size(module, handler_name)
        if logical_size is None:
            return None
        keep_indices = sorted(set(range(logical_size)) - set(indices))
        if (
            isinstance(module, nn.Conv2d)
            and handler_name == "prune_in_channels"
            and module.groups > 1
        ):
            keep_indices = keep_indices[: len(keep_indices) // module.groups]
        return keep_indices

    @staticmethod
    def _grouped_conv_in_channel_indices_are_symmetric(
        module: nn.Conv2d, indices: list[int]
    ) -> bool:
        if module.groups <= 1:
            return True
        channels_per_group = module.in_channels // module.groups
        grouped_indices = [set() for _ in range(module.groups)]
        for index in indices:
            group_index = index // channels_per_group
            if group_index < 0 or group_index >= module.groups:
                return False
            grouped_indices[group_index].add(index % channels_per_group)
        return all(index_set == grouped_indices[0] for index_set in grouped_indices[1:])

    @staticmethod
    def _plain_indices(indices: Any) -> list[int]:
        plain_indices = []
        for index in indices:
            plain_indices.append(int(getattr(index, "idx", index)))
        return plain_indices

    def validate_pruning_group_indices(
        self, pruner: "TRIMPruner", unit: Any, group: Any
    ) -> None:
        for dependency, dep_indices in group.items:
            target = dependency.target.module
            handler_name = dependency.handler.__name__
            select_axis_size = self._select_axis_size(target, handler_name)
            logical_axis_size = self._logical_axis_size(target, handler_name)
            if select_axis_size is None or logical_axis_size is None:
                continue
            indices = self._plain_indices(dep_indices)
            if not indices:
                continue
            min_index = min(indices)
            max_index = max(indices)
            if (
                isinstance(target, nn.Conv2d)
                and handler_name == "prune_in_channels"
                and target.groups > 1
                and not self._grouped_conv_in_channel_indices_are_symmetric(
                    target, indices
                )
            ):
                target_name = pruner.module_name_map.get(id(target), "<unnamed>")
                raise RuntimeError(
                    "Invalid grouped Conv2d in-channel pruning before torch_pruning execution: "
                    f"unit={unit.name}, target={target_name}, groups={target.groups}, "
                    f"indices={indices[:16]}{'...' if len(indices) > 16 else ''}, module={target}"
                )
            keep_indices = self._torch_pruning_keep_indices(
                target, handler_name, indices
            )
            keep_max = max(keep_indices) if keep_indices else -1
            if (
                min_index < 0
                or max_index >= logical_axis_size
                or not keep_indices
                or keep_max >= select_axis_size
            ):
                target_name = pruner.module_name_map.get(id(target), "<unnamed>")
                raise RuntimeError(
                    "Invalid pruning group before torch_pruning execution: "
                    f"unit={unit.name}, target={target_name}, handler={handler_name}, "
                    f"index_range=[{min_index}, {max_index}], logical_axis_size={logical_axis_size}, "
                    f"select_axis_size={select_axis_size}, keep_max={keep_max}, "
                    f"module={target}"
                )

    def _validate_pruning_group_indices(
        self, pruner: "TRIMPruner", unit: Any, group: Any
    ) -> None:
        self.validate_pruning_group_indices(pruner, unit, group)

    def compute_taylor(self, pruner: "TRIMPruner", unit: Any) -> torch.Tensor:
        return pruner._conv_taylor(unit.module)

    def estimate_candidate_delta(
        self, pruner: "TRIMPruner", unit: Any, prune_indices: list[int]
    ) -> tuple[int, float]:
        group = pruner.dependency_graph.get_pruning_group(
            unit.module, unit.pruning_fn, idxs=list(prune_indices)
        )
        delta_params = 0
        delta_gflops = 0.0
        for dependency, dep_indices in group.items:
            target = dependency.target.module
            target_name = pruner.module_name_map.get(id(target))
            output_shape = pruner.output_shape_cache.get(target_name)
            module_params, module_gflops = pruner._estimate_module_action(
                target,
                dependency.handler.__name__,
                len(dep_indices),
                output_shape,
            )
            delta_params += module_params
            delta_gflops += module_gflops
        return int(delta_params), float(delta_gflops)

    def apply_pruning(
        self, pruner: "TRIMPruner", unit: Any, candidate: Any
    ) -> str | None:
        pruner._rebuild_dependency_graph()
        current_modules = dict(pruner.model.named_modules())
        module = current_modules[unit.module_name]
        group = pruner.dependency_graph.get_pruning_group(
            module, unit.pruning_fn, idxs=candidate.prune_indices
        )
        if not pruner.dependency_graph.check_pruning_group(group):
            raise RuntimeError(f"Unsafe pruning group detected for {unit.name}")
        self.validate_pruning_group_indices(pruner, unit, group)
        group.prune()
        return None

    def supports_shared_axis_coupling(self, unit: Any) -> bool:
        return isinstance(unit.module, nn.Conv2d)


class WaveletFusionHandler(UnitKindHandler):
    def compute_taylor(self, pruner: "TRIMPruner", unit: Any) -> torch.Tensor:
        return pruner._conv_taylor(unit.module)

    def estimate_candidate_delta(
        self, pruner: "TRIMPruner", unit: Any, prune_indices: list[int]
    ) -> tuple[int, float]:
        return pruner._estimate_wavelet_fuse_delta(unit, prune_indices)

    def apply_pruning(
        self, pruner: "TRIMPruner", unit: Any, candidate: Any
    ) -> str | None:
        pruner._apply_wavelet_fuse_pruning(unit, candidate.prune_indices)
        return "via direct WaveletDownsample fuse path"


class MambaInnerHandler(UnitKindHandler):
    def compute_taylor(self, pruner: "TRIMPruner", unit: Any) -> torch.Tensor:
        return pruner._mamba_taylor(unit.module)

    def estimate_candidate_delta(
        self, pruner: "TRIMPruner", unit: Any, prune_indices: list[int]
    ) -> tuple[int, float]:
        return pruner._estimate_mamba_delta(
            unit.module,
            len(prune_indices),
            unit.activation_shape or (1, 1, unit.channels),
        )

    def apply_pruning(
        self, pruner: "TRIMPruner", unit: Any, candidate: Any
    ) -> str | None:
        current_module = dict(pruner.model.named_modules())[unit.module_name]
        unit.pruner(current_module, candidate.prune_indices, to_output=True)
        return None


class SwiGLUHandler(UnitKindHandler):
    def compute_taylor(self, pruner: "TRIMPruner", unit: Any) -> torch.Tensor:
        return pruner._swiglu_taylor(unit.module)

    def estimate_candidate_delta(
        self, pruner: "TRIMPruner", unit: Any, prune_indices: list[int]
    ) -> tuple[int, float]:
        return pruner._estimate_swiglu_delta(
            unit.module, len(prune_indices), unit.activation_shape or (1, unit.channels)
        )

    def apply_pruning(
        self, pruner: "TRIMPruner", unit: Any, candidate: Any
    ) -> str | None:
        current_module = dict(pruner.model.named_modules())[unit.module_name]
        unit.pruner(current_module, candidate.prune_indices, to_output=True)
        return None


class MLPHiddenHandler(UnitKindHandler):
    def compute_taylor(self, pruner: "TRIMPruner", unit: Any) -> torch.Tensor:
        return pruner._mlp_taylor(unit.module)

    def estimate_candidate_delta(
        self, pruner: "TRIMPruner", unit: Any, prune_indices: list[int]
    ) -> tuple[int, float]:
        return pruner._estimate_mlp_delta(
            unit.module, len(prune_indices), unit.activation_shape or (1, unit.channels)
        )

    def apply_pruning(
        self, pruner: "TRIMPruner", unit: Any, candidate: Any
    ) -> str | None:
        current_module = dict(pruner.model.named_modules())[unit.module_name]
        unit.pruner(current_module, candidate.prune_indices, to_output=True)
        return None


@dataclass
class UnitKindHandlerRegistry:
    handlers: dict[str, UnitKindHandler]

    def get(self, kind: str) -> UnitKindHandler:
        if kind not in self.handlers:
            raise KeyError(f"No unit-kind handler registered for {kind}")
        return self.handlers[kind]

    def compute_taylor(self, pruner: "TRIMPruner", unit: Any) -> torch.Tensor:
        return self.get(unit.kind).compute_taylor(pruner, unit)

    def estimate_candidate_delta(
        self, pruner: "TRIMPruner", unit: Any, prune_indices: list[int]
    ) -> tuple[int, float]:
        return self.get(unit.kind).estimate_candidate_delta(pruner, unit, prune_indices)

    def apply_pruning(
        self, pruner: "TRIMPruner", unit: Any, candidate: Any
    ) -> str | None:
        return self.get(unit.kind).apply_pruning(pruner, unit, candidate)

    def supports_shared_axis_coupling(self, unit: Any) -> bool:
        return self.get(unit.kind).supports_shared_axis_coupling(unit)


def build_default_unit_kind_handler_registry() -> UnitKindHandlerRegistry:
    conv_handler = DepGraphConvHandler()
    return UnitKindHandlerRegistry(
        handlers={
            "backbone_conv": conv_handler,
            "neck_conv": conv_handler,
            "wavelet_spatial": conv_handler,
            "wavelet_mask": conv_handler,
            "wavelet_fusion": WaveletFusionHandler(),
            "mamba_inner": MambaInnerHandler(),
            "swiglu": SwiGLUHandler(),
            "mlp_hidden": MLPHiddenHandler(),
        }
    )
