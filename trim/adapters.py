from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch.nn as nn
import torch_pruning as tp

from trim.operators.common import Mamba
from trim.operators.feed_forward import FeedForwardMLP, SwiGLUFeedForward
from trim.operators.wavelet import WaveletDownsample

from .block_gates import context_block_gate_module_names

if TYPE_CHECKING:
    from .pruner import TRIMPruner


CollectorFn = Callable[["TRIMPruner", list[Any], dict[str, nn.Module]], None]


def _search_space_enabled(pruner: "TRIMPruner", *modes: str) -> bool:
    search_space = getattr(pruner, "search_space", "stable")
    return search_space == "all_experimental" or search_space in modes


class UnitCollectorRegistry:
    def __init__(self, collectors: list[CollectorFn]) -> None:
        self.collectors = collectors

    def collect(self, pruner: "TRIMPruner") -> list[Any]:
        if pruner.dependency_graph is None:
            pruner.build_dependency_graph()

        units: list[Any] = []
        module_names = dict(pruner.model.named_modules())
        for collector in self.collectors:
            collector(pruner, units, module_names)

        gated_module_names = context_block_gate_module_names(pruner, module_names)
        if gated_module_names:
            gated_module_prefixes = tuple(f"{name}." for name in gated_module_names)
            units = [
                unit
                for unit in units
                if unit.module_name not in gated_module_names
                and not unit.module_name.startswith(gated_module_prefixes)
            ]

        direct_units = {"mamba_inner", "swiglu", "mlp_hidden"}
        module2node = getattr(pruner.dependency_graph, "module2node", {})
        pruner.units = [
            unit
            for unit in units
            if unit.kind in direct_units or unit.module in module2node
        ]
        return pruner.units


def _collect_backbone_convs(
    pruner: "TRIMPruner", units: list[Any], module_names: dict[str, nn.Module]
) -> None:
    conv_pruner = tp.prune_conv_out_channels.__self__
    wavelet_prefixes = tuple(
        f"{name}."
        for name, module in module_names.items()
        if name.startswith("backbone.") and isinstance(module, WaveletDownsample)
    )
    for name, module in module_names.items():
        if name.startswith("backbone.") and isinstance(module, nn.Conv2d):
            if name.startswith(wavelet_prefixes):
                continue
            if ".aggregation.1.conv" in name:
                pruner._log(
                    f"skipping Backbone block output root {name}: downstream concatenated block concat widths "
                    "require a dedicated structural adapter."
                )
                continue
            pruner._register_unit(
                units,
                name=f"Backbone::{name}",
                scope="Backbone",
                kind="backbone_conv",
                module_name=name,
                activation_module_name=name,
                module=module,
                activation_module=module,
                pruner=conv_pruner,
                channel_dim=1,
            )


def _collect_wavelet_units(
    pruner: "TRIMPruner", units: list[Any], module_names: dict[str, nn.Module]
) -> None:
    conv_pruner = tp.prune_conv_out_channels.__self__
    for name, module in module_names.items():
        if not isinstance(module, WaveletDownsample):
            continue

        if getattr(module, "impl", None) == "wfed_hl_pan":
            frequency_roots = (
                ("high_conv.pointwise.conv", "wavelet_spatial"),
                ("low_conv.pointwise.conv", "wavelet_mask"),
            )
            for suffix, kind in frequency_roots:
                root_name = f"{name}.{suffix}"
                root_module = module_names.get(root_name)
                if not isinstance(root_module, nn.Conv2d):
                    continue
                pruner._register_unit(
                    units,
                    name=f"WaveletDownsample::{root_name}",
                    scope="WaveletDownsample",
                    kind=kind,
                    module_name=root_name,
                    activation_module_name=root_name,
                    module=root_module,
                    activation_module=root_module,
                    pruner=conv_pruner,
                    channel_dim=1,
                )

            reduce_name = f"{name}.reduce.conv"
            if pruner._is_sum_fused_wavelet_output(reduce_name, module_names):
                pruner._log(
                    f"skipping WFED-WaveletDownsample reduce root {reduce_name}: current Encoder uses sum fusion, "
                    "so reduce output channels are shared hidden-dim states rather than a local pruning axis."
                )
            continue

        if isinstance(module.spatial_branch, nn.Sequential):
            spatial_name = f"{name}.spatial_branch.1.conv"
            spatial_module = module_names.get(spatial_name)
        else:
            spatial_name = f"{name}.spatial_branch.conv"
            spatial_module = module_names.get(spatial_name)
        if isinstance(spatial_module, nn.Conv2d):
            pruner._register_unit(
                units,
                name=f"WaveletDownsample::{spatial_name}",
                scope="WaveletDownsample",
                kind="wavelet_spatial",
                module_name=spatial_name,
                activation_module_name=spatial_name,
                module=spatial_module,
                activation_module=spatial_module,
                pruner=conv_pruner,
                channel_dim=1,
            )

        mask_name = f"{name}.mask_conv1.conv"
        mask_module = module_names.get(mask_name)
        if isinstance(mask_module, nn.Conv2d):
            pruner._register_unit(
                units,
                name=f"WaveletDownsample::{mask_name}",
                scope="WaveletDownsample",
                kind="wavelet_mask",
                module_name=mask_name,
                activation_module_name=mask_name,
                module=mask_module,
                activation_module=mask_module,
                pruner=conv_pruner,
                channel_dim=1,
            )

        fuse_name = f"{name}.fuse.conv"
        fuse_module = module_names.get(fuse_name)
        if not isinstance(fuse_module, nn.Conv2d):
            continue
        if pruner._is_sum_fused_wavelet_output(fuse_name, module_names):
            pruner._log(
                f"skipping WaveletDownsample fuse root {fuse_name}: current Encoder uses sum fusion, "
                "so fuse output channels are shared hidden-dim states rather than a local pruning axis."
            )
            continue
        try:
            pruner._resolve_wavelet_fuse_consumers(fuse_name, module_names)
        except (RuntimeError, ValueError):
            pruner._log(
                f"skipping WaveletDownsample fuse root {fuse_name}: no direct fuse consumer adapter is available."
            )
            continue
        pruner._register_unit(
            units,
            name=f"WaveletDownsample::{fuse_name}",
            scope="WaveletDownsample",
            kind="wavelet_fusion",
            module_name=fuse_name,
            activation_module_name=fuse_name,
            module=fuse_module,
            activation_module=fuse_module,
            pruner=conv_pruner,
            channel_dim=1,
        )


def _collect_encoder_convolutions(
    pruner: "TRIMPruner",
    units: list[Any],
    module_names: dict[str, nn.Module],
) -> None:
    if not _search_space_enabled(pruner, "expanded_neck"):
        return

    conv_pruner = tp.prune_conv_out_channels.__self__
    wavelet_prefixes = tuple(
        f"{name}."
        for name, module in module_names.items()
        if name.startswith("encoder.") and isinstance(module, WaveletDownsample)
    )
    safe_prefixes = (
        "encoder.input_proj.",
        "encoder.lateral_convs.",
        "encoder.fpn_blocks.",
        "encoder.downsample_convs.",
        "encoder.pan_blocks.",
    )
    for name, module in module_names.items():
        if not name.startswith(safe_prefixes):
            continue
        if name.startswith(wavelet_prefixes):
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        pruner._register_unit(
            units,
            name=f"Encoder::{name}",
            scope="Encoder",
            kind="neck_conv",
            module_name=name,
            activation_module_name=name,
            module=module,
            activation_module=module,
            pruner=conv_pruner,
            channel_dim=1,
        )


def _collect_mamba_units(
    pruner: "TRIMPruner", units: list[Any], module_names: dict[str, nn.Module]
) -> None:
    if Mamba is None:
        return
    for name, module in module_names.items():
        if not isinstance(module, Mamba):
            continue
        activation_name = f"{name}.out_proj"
        activation_module = module_names[activation_name]
        pruner._register_unit(
            units,
            name=f"Mamba::{name}",
            scope="Mamba",
            kind="mamba_inner",
            module_name=name,
            activation_module_name=activation_name,
            module=module,
            activation_module=activation_module,
            pruner=pruner.custom_pruners[Mamba],
            channel_dim=-1,
            hook_type="pre",
        )


def _collect_swiglu_units(
    pruner: "TRIMPruner", units: list[Any], module_names: dict[str, nn.Module]
) -> None:
    for name, module in module_names.items():
        if not isinstance(module, SwiGLUFeedForward):
            continue
        activation_name = f"{name}.w12"
        activation_module = module_names[activation_name]
        pruner._register_unit(
            units,
            name=f"Transformer::{name}",
            scope="Transformer",
            kind="swiglu",
            module_name=name,
            activation_module_name=activation_name,
            module=module,
            activation_module=activation_module,
            pruner=pruner.custom_pruners[SwiGLUFeedForward],
            channel_dim=-1,
        )


def _collect_feed_forward_units(
    pruner: "TRIMPruner", units: list[Any], module_names: dict[str, nn.Module]
) -> None:
    for name, module in module_names.items():
        if not isinstance(module, FeedForwardMLP):
            continue
        is_decoder_mlp = name in {
            "decoder.enc_bbox_head",
            "decoder.query_pos_head",
            "decoder.pre_bbox_head",
        } or name.startswith("decoder.dec_bbox_head.")
        if not is_decoder_mlp:
            continue
        activation_name = f"{name}.layers.0"
        activation_module = module_names.get(activation_name)
        if activation_module is None:
            continue
        pruner._register_unit(
            units,
            name=f"Transformer::{name}",
            scope="Transformer",
            kind="mlp_hidden",
            module_name=name,
            activation_module_name=activation_name,
            module=module,
            activation_module=activation_module,
            pruner=pruner.custom_pruners[FeedForwardMLP],
            channel_dim=-1,
        )


def build_default_unit_collector_registry() -> UnitCollectorRegistry:
    return UnitCollectorRegistry(
        [
            _collect_backbone_convs,
            _collect_wavelet_units,
            _collect_encoder_convolutions,
            _collect_mamba_units,
            _collect_swiglu_units,
            _collect_feed_forward_units,
        ]
    )
