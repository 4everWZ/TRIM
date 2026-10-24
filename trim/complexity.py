from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .runtime import eval_spatial_size_override, suppress_stdout_stderr


def _count_tokens(shape: Sequence[int]) -> int:
    if len(shape) <= 1:
        return 1
    return int(np.prod(shape[:-1]))


def _conv_output_hw(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) >= 4:
        return int(shape[-2]), int(shape[-1])
    if len(shape) == 3:
        return 1, int(shape[-1])
    return 1, 1


def _flops_to_gflops(flops: float) -> float:
    return float(flops) / 1_000_000_000.0


def _measure_model_complexity(
    model: nn.Module, input_shape: Sequence[int], device: torch.device
) -> dict[str, Any]:
    try:
        from calflops import calculate_flops
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError(
            "TRIM pruning requires `calflops` in the active environment."
        ) from exc

    model_for_info = copy.deepcopy(model).eval()
    if hasattr(model_for_info, "deploy"):
        model_for_info = model_for_info.deploy()
    model_for_info = model_for_info.to(device)

    with (
        eval_spatial_size_override(model_for_info, input_shape[-2:]),
        suppress_stdout_stderr(),
    ):
        flops, macs, _ = calculate_flops(
            model=model_for_info,
            input_shape=tuple(input_shape),
            output_as_string=True,
            output_precision=4,
            print_detailed=False,
        )
    params = sum(parameter.numel() for parameter in model_for_info.parameters())
    return {
        "params": int(params),
        "params_million": params / 1_000_000.0,
        "flops": flops,
        "macs": macs,
        "flops_value": float(str(flops).split()[0]),
        "macs_value": float(str(macs).split()[0]),
    }
