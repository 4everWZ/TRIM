from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch_pruning as tp

from trim.operators.common import Mamba
from trim.operators.feed_forward import FeedForwardMLP, SwiGLUFeedForward
from trim.operators.normalization import FrozenBatchNorm2d


def _keep_indices(total: int, idxs: Sequence[int]) -> list[int]:
    keep_idxs = sorted(set(range(total)) - set(int(idx) for idx in idxs))
    if not keep_idxs:
        raise ValueError("Pruning would remove every channel from the target module.")
    return keep_idxs


def _select_param(
    parameter: torch.nn.Parameter | None, keep_idxs: Sequence[int], dim: int
) -> torch.nn.Parameter | None:
    if parameter is None:
        return None
    index = torch.as_tensor(keep_idxs, dtype=torch.long, device=parameter.device)
    pruned = nn.Parameter(torch.index_select(parameter, dim, index).contiguous())
    if parameter.grad is not None:
        pruned.grad = torch.index_select(parameter.grad, dim, index).contiguous()
    return pruned


def _select_buffer(
    buffer: torch.Tensor | None, keep_idxs: Sequence[int], dim: int
) -> torch.Tensor | None:
    if buffer is None:
        return None
    index = torch.as_tensor(keep_idxs, dtype=torch.long, device=buffer.device)
    return torch.index_select(buffer, dim, index).contiguous()


class FrozenBatchNormPruner(tp.pruner.function.BasePruningFunc):
    TARGET_MODULES = (FrozenBatchNorm2d,)

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        if not isinstance(layer, FrozenBatchNorm2d):
            raise TypeError(
                f"FrozenBatchNormPruner requires FrozenBatchNorm2d, got {type(layer)}"
            )

        keep_idxs = _keep_indices(layer.num_features, idxs)
        layer.weight = _select_buffer(layer.weight, keep_idxs, 0)
        layer.bias = _select_buffer(layer.bias, keep_idxs, 0)
        layer.running_mean = _select_buffer(layer.running_mean, keep_idxs, 0)
        layer.running_var = _select_buffer(layer.running_var, keep_idxs, 0)
        layer.num_features = len(keep_idxs)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.Module):
        return layer.num_features

    get_in_channels = get_out_channels


class MambaInnerWidthPruner(tp.pruner.function.BasePruningFunc):
    TARGET_MODULES = (Mamba,) if Mamba is not None else None

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        if Mamba is None or not isinstance(layer, Mamba):
            raise TypeError(
                f"MambaInnerWidthPruner requires mamba_ssm.Mamba, got {type(layer)}"
            )

        keep_idxs = _keep_indices(layer.d_inner, idxs)
        paired_keep_idxs = keep_idxs + [idx + layer.d_inner for idx in keep_idxs]

        layer.in_proj.out_features = len(paired_keep_idxs)
        layer.in_proj.weight = _select_param(layer.in_proj.weight, paired_keep_idxs, 0)
        layer.in_proj.bias = _select_param(layer.in_proj.bias, paired_keep_idxs, 0)

        layer.conv1d.in_channels = len(keep_idxs)
        layer.conv1d.out_channels = len(keep_idxs)
        layer.conv1d.groups = len(keep_idxs)
        layer.conv1d.weight = _select_param(layer.conv1d.weight, keep_idxs, 0)
        layer.conv1d.bias = _select_param(layer.conv1d.bias, keep_idxs, 0)

        layer.x_proj.in_features = len(keep_idxs)
        layer.x_proj.weight = _select_param(layer.x_proj.weight, keep_idxs, 1)

        layer.dt_proj.out_features = len(keep_idxs)
        layer.dt_proj.weight = _select_param(layer.dt_proj.weight, keep_idxs, 0)
        layer.dt_proj.bias = _select_param(layer.dt_proj.bias, keep_idxs, 0)

        layer.A_log = _select_param(layer.A_log, keep_idxs, 0)
        layer.D = _select_param(layer.D, keep_idxs, 0)

        layer.out_proj.in_features = len(keep_idxs)
        layer.out_proj.weight = _select_param(layer.out_proj.weight, keep_idxs, 1)

        layer.d_inner = len(keep_idxs)
        if getattr(layer, "d_model", None):
            layer.expand = layer.d_inner / float(layer.d_model)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.Module):
        return layer.d_inner

    get_in_channels = get_out_channels


class SwiGLUWidthPruner(tp.pruner.function.BasePruningFunc):
    TARGET_MODULES = (SwiGLUFeedForward,)

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = _keep_indices(layer.w3.in_features, idxs)
        paired_keep_idxs = keep_idxs + [idx + layer.w3.in_features for idx in keep_idxs]

        layer.w12.out_features = len(paired_keep_idxs)
        layer.w12.weight = _select_param(layer.w12.weight, paired_keep_idxs, 0)
        layer.w12.bias = _select_param(layer.w12.bias, paired_keep_idxs, 0)

        layer.w3.in_features = len(keep_idxs)
        layer.w3.weight = _select_param(layer.w3.weight, keep_idxs, 1)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.Module):
        return layer.w3.in_features

    get_in_channels = get_out_channels


class MLPWidthPruner(tp.pruner.function.BasePruningFunc):
    TARGET_MODULES = (FeedForwardMLP,)

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        if layer.num_layers < 2:
            raise ValueError(
                "MLPWidthPruner requires an FeedForwardMLP with at least one hidden layer."
            )

        hidden_dim = layer.layers[0].out_features
        keep_idxs = _keep_indices(hidden_dim, idxs)

        first = layer.layers[0]
        first.out_features = len(keep_idxs)
        first.weight = _select_param(first.weight, keep_idxs, 0)
        first.bias = _select_param(first.bias, keep_idxs, 0)

        for hidden_layer in layer.layers[1:-1]:
            hidden_layer.in_features = len(keep_idxs)
            hidden_layer.weight = _select_param(hidden_layer.weight, keep_idxs, 1)
            hidden_layer.out_features = len(keep_idxs)
            hidden_layer.weight = _select_param(hidden_layer.weight, keep_idxs, 0)
            hidden_layer.bias = _select_param(hidden_layer.bias, keep_idxs, 0)

        last = layer.layers[-1]
        last.in_features = len(keep_idxs)
        last.weight = _select_param(last.weight, keep_idxs, 1)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.Module):
        if layer.num_layers < 2:
            return None
        return layer.layers[0].out_features

    get_in_channels = get_out_channels
