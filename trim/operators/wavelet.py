"""Wavelet downsampling operators supported by TRIM structural adapters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    ConvBNAct2d,
    GlobalResponseChannelAttention,
    HaarDWT2d,
)
from .numerics import amp_safe_product
from .wavelet_variants import (
    configure_dual_interaction,
    configure_three_branch,
    dual_interaction_forward,
    three_branch_forward,
)

__all__ = ["WaveletDownsample"]


class WaveletDownsample(nn.Module):
    def __init__(
        self,
        channels,
        act="silu",
        impl="wtfm_paper_pan",
        variant="full",
        mask_ratio=1.0,
        mask_hidden_channels=None,
        spatial_pw_groups=1,
        fuse_groups=1,
        use_wavelet_mask=True,
    ):
        super().__init__()
        impl_aliases = {
            "wavelet_dual_interact": "legacy_dual_interact",
        }
        impl = impl_aliases.get(impl, impl)
        if impl not in [
            "wtfm_paper_pan",
            "wfed_hl_pan",
            "legacy_dual_interact",
            "legacy_three_branch",
        ]:
            raise ValueError(f"Unsupported WaveletDownsample impl: {impl}")
        if variant not in ["full", "lite"]:
            raise ValueError(f"Unsupported WaveletDownsample variant: {variant}")
        if mask_ratio is None:
            raise ValueError("WaveletDownsample mask_ratio must not be None.")
        if not (0.0 < mask_ratio <= 1.0):
            raise ValueError(
                f"WaveletDownsample mask_ratio must be in (0, 1], got: {mask_ratio}"
            )
        if mask_hidden_channels is not None:
            if not isinstance(mask_hidden_channels, int):
                raise TypeError(
                    f"WaveletDownsample mask_hidden_channels must be an int when provided, got: {type(mask_hidden_channels)}"
                )
            if mask_hidden_channels <= 0:
                raise ValueError(
                    f"WaveletDownsample mask_hidden_channels must be positive, got: {mask_hidden_channels}"
                )
        if spatial_pw_groups <= 0:
            raise ValueError(
                f"WaveletDownsample spatial_pw_groups must be positive, got: {spatial_pw_groups}"
            )
        if fuse_groups <= 0:
            raise ValueError(
                f"WaveletDownsample fuse_groups must be positive, got: {fuse_groups}"
            )
        if channels % spatial_pw_groups != 0:
            raise ValueError(
                f"WaveletDownsample spatial_pw_groups must divide channels. channels={channels}, "
                f"spatial_pw_groups={spatial_pw_groups}"
            )
        if channels % fuse_groups != 0:
            raise ValueError(
                f"WaveletDownsample fuse_groups must divide channels. channels={channels}, fuse_groups={fuse_groups}"
            )
        if impl == "legacy_dual_interact":
            fuse_input_channels = channels * 2
        elif impl == "wfed_hl_pan":
            fuse_input_channels = channels * 4
        else:
            fuse_input_channels = channels * 3
        if fuse_input_channels % fuse_groups != 0:
            raise ValueError(
                "WaveletDownsample fuse_groups must divide the active fuse width. "
                f"channels={channels}, fuse_input_channels={fuse_input_channels}, fuse_groups={fuse_groups}"
            )

        self.impl = impl
        self.variant = variant
        self.mask_ratio = float(mask_ratio)
        self.mask_hidden_channels = mask_hidden_channels
        self.spatial_pw_groups = int(spatial_pw_groups)
        self.fuse_groups = int(fuse_groups)
        self.use_wavelet_mask = use_wavelet_mask
        self.fuse_input_channels = fuse_input_channels

        if self.impl == "wfed_hl_pan":
            if mask_hidden_channels is None:
                mask_hidden_channels = max(1, int(channels * self.mask_ratio))
            elif mask_hidden_channels > channels:
                raise ValueError(
                    "WFED-WaveletDownsample expects mask_hidden_channels no larger than channels. "
                    f"channels={channels}, mask_hidden_channels={mask_hidden_channels}"
                )
            self.mask_hidden_channels = int(mask_hidden_channels)
            self.wavelet = HaarDWT2d(channels)
            self.laplacian_filter = _FixedDepthwiseFilter2d(
                channels, _laplacian_kernel()
            )
            self.gaussian_filter = _FixedDepthwiseFilter2d(channels, _gaussian_kernel())
            self.high_aggregate = ConvBNAct2d(
                channels * 3,
                self.mask_hidden_channels,
                kernel_size=1,
                stride=1,
                act=act,
            )
            self.high_conv = _SeparableConvBNAct2d(
                self.mask_hidden_channels,
                self.mask_hidden_channels,
                kernel_size=3,
                act=act,
            )
            self.high_grca = GlobalResponseChannelAttention(self.mask_hidden_channels)
            self.high_pixel_attention = nn.Sequential(
                nn.Conv2d(
                    self.mask_hidden_channels,
                    self.mask_hidden_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
            self.high_restore = ConvBNAct2d(
                self.mask_hidden_channels,
                channels * 3,
                kernel_size=1,
                stride=1,
                act=act,
            )
            self.low_conv = _SeparableConvBNAct2d(
                channels,
                self.mask_hidden_channels,
                kernel_size=3,
                act=act,
            )
            self.low_grca = GlobalResponseChannelAttention(self.mask_hidden_channels)
            self.low_restore = ConvBNAct2d(
                self.mask_hidden_channels, channels, kernel_size=1, stride=1, act=act
            )
            self.reduce = ConvBNAct2d(
                self.fuse_input_channels,
                channels,
                kernel_size=1,
                stride=1,
                groups=self.fuse_groups,
                act=act,
            )
            self.shortcut_pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
            return

        if variant == "full":
            if mask_hidden_channels is not None and mask_hidden_channels != channels:
                raise ValueError(
                    "WaveletDownsample full variant fixes mask_hidden_channels to channels. "
                    f"channels={channels}, mask_hidden_channels={mask_hidden_channels}"
                )
            self.local_branch = ConvBNAct2d(
                channels, channels, kernel_size=3, stride=2, act=act
            )
            self.spatial_branch = ConvBNAct2d(
                channels, channels, kernel_size=7, stride=2, act=act
            )
            mask_hidden_channels = channels
            mask_input_channels = channels
        else:
            if mask_hidden_channels is None:
                mask_hidden_channels = max(1, int(channels * self.mask_ratio))
            elif mask_hidden_channels > channels:
                raise ValueError(
                    "WaveletDownsample lite variant expects a bottleneck mask width no larger than channels. "
                    f"channels={channels}, mask_hidden_channels={mask_hidden_channels}"
                )
            self.local_branch = ConvBNAct2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                groups=channels,
                act=act,
            )
            self.spatial_branch = nn.Sequential(
                ConvBNAct2d(
                    channels,
                    channels,
                    kernel_size=7,
                    stride=2,
                    groups=channels,
                    act=act,
                ),
                ConvBNAct2d(
                    channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    groups=self.spatial_pw_groups,
                    act=act,
                ),
            )
            mask_input_channels = mask_hidden_channels

        self.mask_hidden_channels = int(mask_hidden_channels)

        if self.impl == "wtfm_paper_pan":
            self.wavelet = HaarDWT2d(channels)
            self.mask_conv1 = ConvBNAct2d(
                channels * 4,
                self.mask_hidden_channels,
                kernel_size=3,
                stride=1,
                act=act,
            )
            self.mask_conv2 = nn.Sequential(
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
            self.fuse = ConvBNAct2d(
                self.fuse_input_channels,
                channels,
                kernel_size=1,
                stride=1,
                groups=self.fuse_groups,
                act=act,
            )
        elif self.impl == "legacy_dual_interact":
            configure_dual_interaction(
                self,
                channels=channels,
                act=act,
                variant=variant,
                mask_hidden_channels=self.mask_hidden_channels,
                spatial_pw_groups=self.spatial_pw_groups,
                fuse_groups=self.fuse_groups,
                conv_block_cls=ConvBNAct2d,
                dwt_cls=HaarDWT2d,
            )
        else:
            configure_three_branch(
                self,
                channels=channels,
                act=act,
                variant=variant,
                mask_hidden_channels=self.mask_hidden_channels,
                spatial_pw_groups=self.spatial_pw_groups,
                fuse_groups=self.fuse_groups,
                conv_block_cls=ConvBNAct2d,
                dwt_cls=HaarDWT2d,
            )

    def _feature_modulation(self, local_feat, spatial_shape):
        upsampled_bands = []
        for band in self.wavelet(local_feat):
            if band.shape[-2:] != local_feat.shape[-2:]:
                band = F.interpolate(
                    band,
                    size=local_feat.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
            upsampled_bands.append(band)
        wavelet_feat = torch.cat(upsampled_bands, dim=1)
        modulation = self.mask_conv2(self.mask_conv1(wavelet_feat))
        if modulation.shape[-2:] != spatial_shape:
            modulation = F.interpolate(
                modulation, size=spatial_shape, mode="bilinear", align_corners=False
            )
        return modulation

    def _frequency_high_frequency(self, lh, hl, hh):
        high_filtered = torch.cat(
            [
                self.laplacian_filter(lh),
                self.laplacian_filter(hl),
                self.laplacian_filter(hh),
            ],
            dim=1,
        )
        high_merged = self.high_aggregate(high_filtered)
        high_enhanced = self.high_grca(self.high_conv(high_merged))
        high_restored = self.high_restore(
            high_enhanced * self.high_pixel_attention(high_enhanced)
        )
        return torch.chunk(high_restored, chunks=3, dim=1)

    def _frequency_low_frequency(self, band):
        filtered = self.gaussian_filter(band)
        return self.low_restore(self.low_grca(self.low_conv(filtered)))

    def _frequency_enhance_bands(self, bands):
        ll, lh, hl, hh = bands
        lh_enhanced, hl_enhanced, hh_enhanced = self._frequency_high_frequency(
            lh, hl, hh
        )
        return (
            self._frequency_low_frequency(ll),
            lh_enhanced,
            hl_enhanced,
            hh_enhanced,
        )

    def _forward_frequency_hl_pan(self, x):
        bands = self._frequency_enhance_bands(self.wavelet(x))
        reduced = self.reduce(torch.cat(bands, dim=1))
        shortcut = self.shortcut_pool(x)
        if shortcut.shape[-2:] != reduced.shape[-2:]:
            shortcut = F.interpolate(
                shortcut, size=reduced.shape[-2:], mode="bilinear", align_corners=False
            )
        return reduced + shortcut

    def forward(self, x):
        if self.impl == "legacy_dual_interact":
            return dual_interaction_forward(self, x)
        if self.impl == "legacy_three_branch":
            return three_branch_forward(self, x)
        if self.impl == "wfed_hl_pan":
            return self._forward_frequency_hl_pan(x)

        local_feat = self.local_branch(x)
        spatial_feat = self.spatial_branch(x)
        if self.use_wavelet_mask:
            spatial_modulated = amp_safe_product(
                spatial_feat,
                self._feature_modulation(local_feat, spatial_feat.shape[-2:]),
            )
        else:
            spatial_modulated = spatial_feat
        combined_feat = torch.cat([local_feat, spatial_feat, spatial_modulated], dim=1)
        return self.fuse(combined_feat)


def _laplacian_kernel() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )


def _gaussian_kernel() -> torch.Tensor:
    return (
        torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        / 16.0
    )


class _FixedDepthwiseFilter2d(nn.Module):
    def __init__(self, channels: int, kernel: torch.Tensor):
        super().__init__()
        if kernel.dim() != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError(
                f"Fixed depthwise filter expects a square 2D kernel, got: {tuple(kernel.shape)}"
            )
        self.channels = int(channels)
        self.kernel_size = int(kernel.shape[0])
        weight = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(
            self.channels, 1, 1, 1
        )
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight.to(dtype=x.dtype, device=x.device),
            padding=self.kernel_size // 2,
            groups=self.channels,
        )


class _SeparableConvBNAct2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, act="silu"
    ):
        super().__init__()
        self.depthwise = ConvBNAct2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels,
            act=act,
        )
        self.pointwise = ConvBNAct2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))
