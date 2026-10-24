from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .numerics import amp_safe_residual_product


def _configure_common_wavelet_branches(
    module: nn.Module,
    *,
    channels: int,
    act: str,
    variant: str,
    mask_hidden_channels: int,
    spatial_pw_groups: int,
    conv_block_cls,
):
    if variant == "full":
        module.local_branch = conv_block_cls(
            channels, channels, kernel_size=3, stride=2, act=act
        )
        module.spatial_branch = conv_block_cls(
            channels, channels, kernel_size=7, stride=2, act=act
        )
        mask_input_channels = channels
    else:
        module.local_branch = conv_block_cls(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            groups=channels,
            act=act,
        )
        module.spatial_branch = nn.Sequential(
            conv_block_cls(
                channels,
                channels,
                kernel_size=7,
                stride=2,
                groups=channels,
                act=act,
            ),
            conv_block_cls(
                channels,
                channels,
                kernel_size=1,
                stride=1,
                groups=spatial_pw_groups,
                act=act,
            ),
        )
        mask_input_channels = mask_hidden_channels
    return mask_input_channels


def configure_dual_interaction(
    module: nn.Module,
    *,
    channels: int,
    act: str,
    variant: str,
    mask_hidden_channels: int,
    spatial_pw_groups: int,
    fuse_groups: int,
    conv_block_cls,
    dwt_cls,
):
    mask_input_channels = _configure_common_wavelet_branches(
        module,
        channels=channels,
        act=act,
        variant=variant,
        mask_hidden_channels=mask_hidden_channels,
        spatial_pw_groups=spatial_pw_groups,
        conv_block_cls=conv_block_cls,
    )
    module.wavelet = dwt_cls(channels)
    module.mask_conv1 = conv_block_cls(
        channels * 4, mask_hidden_channels, kernel_size=3, stride=1, act=act
    )
    module.mask_conv2 = nn.Sequential(
        nn.Conv2d(
            mask_input_channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(channels),
    )
    module.local_gate_conv = conv_block_cls(
        channels,
        channels,
        kernel_size=1,
        stride=1,
        groups=fuse_groups,
        act=None,
    )
    module.fuse = conv_block_cls(
        module.fuse_input_channels,
        channels,
        kernel_size=1,
        stride=1,
        groups=fuse_groups,
        act=act,
    )


def configure_three_branch(
    module: nn.Module,
    *,
    channels: int,
    act: str,
    variant: str,
    mask_hidden_channels: int,
    spatial_pw_groups: int,
    fuse_groups: int,
    conv_block_cls,
    dwt_cls,
):
    mask_input_channels = _configure_common_wavelet_branches(
        module,
        channels=channels,
        act=act,
        variant=variant,
        mask_hidden_channels=mask_hidden_channels,
        spatial_pw_groups=spatial_pw_groups,
        conv_block_cls=conv_block_cls,
    )
    module.wavelet = dwt_cls(channels)
    module.mask_conv1 = conv_block_cls(
        channels * 4, mask_hidden_channels, kernel_size=3, stride=1, act=act
    )
    module.mask_conv2 = nn.Sequential(
        nn.Conv2d(
            mask_input_channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(channels),
    )
    module.fuse = conv_block_cls(
        module.fuse_input_channels,
        channels,
        kernel_size=1,
        stride=1,
        groups=fuse_groups,
        act=act,
    )


def compute_wavelet_mask(
    module: nn.Module, x: torch.Tensor, spatial_shape: tuple[int, int]
) -> torch.Tensor:
    c_a, c_h, c_v, c_d = module.wavelet(x)
    wavelet_feat = torch.cat([c_a, c_h, c_v, c_d], dim=1)
    mask = module.mask_conv2(module.mask_conv1(wavelet_feat))
    if mask.shape[-2:] != spatial_shape:
        mask = F.interpolate(mask, size=spatial_shape, mode="nearest")
    return torch.sigmoid(mask)


def dual_interaction_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    local_feat = module.local_branch(x)
    spatial_feat = module.spatial_branch(x)

    if module.use_wavelet_mask:
        spatial_modulated = spatial_feat * compute_wavelet_mask(
            module, x, spatial_feat.shape[-2:]
        )
    else:
        spatial_modulated = spatial_feat

    local_gate = torch.sigmoid(module.local_gate_conv(spatial_modulated))
    local_refined = amp_safe_residual_product(local_feat, local_feat, local_gate)
    combined_feat = torch.cat([local_refined, spatial_modulated], dim=1)
    return module.fuse(combined_feat)


def three_branch_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    local_feat = module.local_branch(x)
    spatial_feat = module.spatial_branch(x)

    if module.use_wavelet_mask:
        spatial_modulated = spatial_feat * compute_wavelet_mask(
            module, x, spatial_feat.shape[-2:]
        )
    else:
        spatial_modulated = spatial_feat

    combined_feat = torch.cat([local_feat, spatial_feat, spatial_modulated], dim=1)
    return module.fuse(combined_feat)
