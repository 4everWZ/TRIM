from __future__ import annotations

import torch

_FP16_MAX = torch.finfo(torch.float16).max


def _has_fp16_tensor(*tensors: torch.Tensor) -> bool:
    return any(
        torch.is_tensor(tensor) and tensor.dtype == torch.float16 for tensor in tensors
    )


def _clamp_finite_to_fp16_range(tensor: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(tensor)
    clamped = tensor.clamp(min=-_FP16_MAX, max=_FP16_MAX)
    return torch.where(finite, clamped, tensor)


def amp_safe_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not _has_fp16_tensor(left, right):
        return left * right
    product = left.float() * right.float()
    return _clamp_finite_to_fp16_range(product).to(dtype=torch.float16)


def amp_safe_residual_product(
    residual: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if not _has_fp16_tensor(residual, left, right):
        return residual + (left * right)
    output = residual.float() + left.float() * right.float()
    return _clamp_finite_to_fp16_range(output).to(dtype=residual.dtype)


def amp_safe_sum_of_products(
    left_a: torch.Tensor,
    right_a: torch.Tensor,
    left_b: torch.Tensor,
    right_b: torch.Tensor,
) -> torch.Tensor:
    if not _has_fp16_tensor(left_a, right_a, left_b, right_b):
        return left_a * right_a + left_b * right_b
    output = left_a.float() * right_a.float() + left_b.float() * right_b.float()
    return _clamp_finite_to_fp16_range(output).to(dtype=torch.float16)
