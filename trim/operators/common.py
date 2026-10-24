"""Shared wavelet and state-space operator primitives."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .numerics import amp_safe_product, amp_safe_residual_product

try:
    from mamba_ssm import Mamba

    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised in dependency check tests
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc

__all__ = [
    "Mamba",
    "ConvBNAct2d",
    "HaarDWT2d",
    "HaarIDWT2d",
    "LayerNorm2d",
    "MultiSpectralChannelAttention",
    "MultiSpectralDCTLayer",
    "SemanticPromptBankAdapter",
    "GlobalResponseChannelAttention",
    "get_activation",
]


def get_activation(act: str, inplace: bool = True):
    if act is None:
        return nn.Identity()

    if isinstance(act, nn.Module):
        return act

    act = act.lower()
    if act in ["silu", "swish"]:
        module = nn.SiLU()
    elif act == "relu":
        module = nn.ReLU()
    elif act == "leaky_relu":
        module = nn.LeakyReLU()
    elif act == "gelu":
        module = nn.GELU()
    elif act == "hardsigmoid":
        module = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unsupported activation: {act}")

    if hasattr(module, "inplace"):
        module.inplace = inplace
    return module


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvBNAct2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, groups=1, act="silu"
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


_FCA_FREQUENCY_INDICES = {
    "top": (
        [
            0,
            0,
            6,
            0,
            0,
            1,
            1,
            4,
            5,
            1,
            3,
            0,
            0,
            0,
            3,
            2,
            4,
            6,
            3,
            5,
            5,
            2,
            6,
            5,
            5,
            3,
            3,
            4,
            2,
            2,
            6,
            1,
        ],
        [
            0,
            1,
            0,
            5,
            2,
            0,
            2,
            0,
            0,
            6,
            0,
            4,
            6,
            3,
            5,
            2,
            6,
            3,
            3,
            3,
            5,
            1,
            1,
            2,
            4,
            2,
            1,
            1,
            3,
            0,
            5,
            3,
        ],
    ),
    "low": (
        [
            0,
            0,
            1,
            1,
            0,
            2,
            2,
            1,
            2,
            0,
            3,
            4,
            0,
            1,
            3,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            1,
            2,
            3,
            4,
        ],
        [
            0,
            1,
            0,
            1,
            2,
            0,
            1,
            2,
            2,
            3,
            0,
            0,
            4,
            3,
            1,
            5,
            4,
            3,
            2,
            1,
            0,
            6,
            5,
            4,
            3,
            2,
            1,
            0,
            6,
            5,
            4,
            3,
        ],
    ),
    "bot": (
        [
            6,
            1,
            3,
            3,
            2,
            4,
            1,
            2,
            4,
            4,
            5,
            1,
            4,
            6,
            2,
            5,
            6,
            1,
            6,
            2,
            2,
            4,
            3,
            3,
            5,
            5,
            6,
            2,
            5,
            5,
            3,
            6,
        ],
        [
            6,
            4,
            4,
            6,
            6,
            3,
            1,
            4,
            4,
            5,
            6,
            5,
            2,
            2,
            5,
            1,
            4,
            3,
            5,
            0,
            3,
            1,
            1,
            2,
            4,
            2,
            1,
            1,
            5,
            3,
            3,
            3,
        ],
    ),
}


def _resolve_fca_frequency_method(channels: int, preferred: str = "top16") -> str:
    prefix = preferred[:3]
    for count in [16, 8, 4, 2, 1]:
        if channels % count == 0:
            return f"{prefix}{count}"
    return f"{prefix}1"


def _get_fca_frequency_indices(method: str):
    if len(method) < 4:
        raise ValueError(f"Unsupported FcaNet frequency selection: {method}")
    prefix = method[:3]
    if prefix not in _FCA_FREQUENCY_INDICES:
        raise ValueError(f"Unsupported FcaNet frequency selection: {method}")
    count = int(method[3:])
    mapper_x, mapper_y = _FCA_FREQUENCY_INDICES[prefix]
    if count <= 0 or count > len(mapper_x):
        raise ValueError(f"Unsupported FcaNet frequency count in method: {method}")
    return mapper_x[:count], mapper_y[:count]


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, channels, dct_h=7, dct_w=7, freq_sel_method="top16"):
        super().__init__()
        mapper_x, mapper_y = _get_fca_frequency_indices(freq_sel_method)
        if channels % len(mapper_x) != 0:
            raise ValueError(
                "FcaNet attention expects channels divisible by the number of selected frequencies. "
                f"channels={channels}, frequencies={len(mapper_x)}"
            )

        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.register_buffer(
            "weight",
            self._build_dct_filter(dct_h, dct_w, mapper_x, mapper_y, channels),
            persistent=False,
        )

    @staticmethod
    def _build_filter(pos, freq, size):
        result = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
        return result if freq == 0 else result * math.sqrt(2)

    @classmethod
    def _build_dct_filter(cls, tile_h, tile_w, mapper_x, mapper_y, channels):
        dct_filter = torch.zeros(channels, tile_h, tile_w)
        c_part = channels // len(mapper_x)
        for index, (freq_x, freq_y) in enumerate(zip(mapper_x, mapper_y)):
            for pos_x in range(tile_h):
                for pos_y in range(tile_w):
                    dct_filter[index * c_part : (index + 1) * c_part, pos_x, pos_y] = (
                        cls._build_filter(pos_x, freq_x, tile_h)
                        * cls._build_filter(pos_y, freq_y, tile_w)
                    )
        return dct_filter

    def forward(self, x):
        x = x * self.weight.to(dtype=x.dtype)
        return torch.sum(x, dim=[2, 3])


class MultiSpectralChannelAttention(nn.Module):
    def __init__(
        self, channels, reduction=16, dct_h=7, dct_w=7, freq_sel_method="top16"
    ):
        super().__init__()
        freq_sel_method = _resolve_fca_frequency_method(
            channels, preferred=freq_sel_method
        )
        reduced_channels = max(1, channels // reduction)
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.dct_layer = MultiSpectralDCTLayer(
            channels=channels,
            dct_h=dct_h,
            dct_w=dct_w,
            freq_sel_method=freq_sel_method,
        )
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        pooled = x
        if height != self.dct_h or width != self.dct_w:
            pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        weights = self.fc(self.dct_layer(pooled)).view(batch_size, channels, 1, 1)
        return x * weights


class GlobalResponseChannelAttention(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        response = torch.norm(x.float(), p=2, dim=(2, 3), keepdim=True)
        normalized = response / (response.mean(dim=1, keepdim=True) + self.eps)
        modulated = (
            x.float() * normalized
            if x.dtype == torch.float16
            else x * normalized.to(dtype=x.dtype)
        )
        output = amp_safe_residual_product(x, self.gamma.to(dtype=x.dtype), modulated)
        return output + self.beta.to(dtype=output.dtype)


class SemanticPromptBankAdapter(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.norm = nn.LayerNorm(dim)
        self.assignment = nn.Linear(dim, rank, bias=False)
        self.prompt_bank = nn.Parameter(torch.randn(rank, dim) * (dim**-0.5))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        sequence = self.norm(sequence)

        assignment = torch.softmax(self.assignment(sequence), dim=1)
        prompts = torch.einsum("blr,blc->brc", assignment, sequence)
        prompts = prompts + self.prompt_bank.unsqueeze(0)

        prompt_logits = torch.einsum("blc,brc->blr", sequence, prompts) / math.sqrt(
            float(channels)
        )
        prompt_weights = torch.softmax(prompt_logits, dim=-1)
        context = torch.einsum("blr,brc->blc", prompt_weights, prompts)
        gate = torch.sigmoid(self.gate_proj(context))
        output = amp_safe_residual_product(sequence, gate, self.out_proj(context))
        return (
            output.reshape(batch_size, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class HaarDWT2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        base_filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer(
            "weight", base_filters.repeat(channels, 1, 1, 1), persistent=False
        )

    def forward(self, x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        y = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=2, groups=self.channels)
        bsz, _, height, width = y.shape
        y = y.view(bsz, self.channels, 4, height, width)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


class HaarIDWT2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        base_filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer(
            "weight", base_filters.repeat(channels, 1, 1, 1), persistent=False
        )

    def forward(self, bands, output_size=None):
        if isinstance(bands, (tuple, list)):
            if len(bands) != 4:
                raise ValueError(f"HaarIDWT2d expects four subbands, got: {len(bands)}")
            band_tensor = torch.stack(tuple(bands), dim=2).contiguous()
            bsz, channels, _, height, width = band_tensor.shape
            x = band_tensor.reshape(bsz, channels * 4, height, width)
        elif isinstance(bands, torch.Tensor):
            if bands.dim() == 5:
                if bands.shape[1] != self.channels or bands.shape[2] != 4:
                    raise ValueError(
                        "HaarIDWT2d 5D tensor input must be [B, C, 4, H, W], "
                        f"got shape={tuple(bands.shape)}"
                    )
                bsz, channels, _, height, width = bands.shape
                x = bands.reshape(bsz, channels * 4, height, width)
            elif bands.dim() == 4:
                x = bands
            else:
                raise ValueError(
                    f"HaarIDWT2d expects 4D or 5D tensor input, got shape={tuple(bands.shape)}"
                )
        else:
            raise TypeError(
                f"HaarIDWT2d expects a tensor or a four-subband tuple, got: {type(bands)}"
            )

        if x.shape[1] != self.channels * 4:
            raise ValueError(
                f"HaarIDWT2d expects {self.channels * 4} input channels, got: {x.shape[1]}"
            )

        restored = F.conv_transpose2d(
            x,
            self.weight.to(dtype=x.dtype),
            stride=2,
            groups=self.channels,
        )
        if output_size is None:
            return restored
        height, width = output_size
        return restored[:, :, :height, :width]
