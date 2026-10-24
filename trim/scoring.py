from __future__ import annotations

import math

import numpy as np
import torch


def prediction_summary(
    outputs: dict[str, torch.Tensor], topk: int = 20
) -> torch.Tensor:
    logits = outputs["pred_logits"].detach().float()
    batch_size = logits.shape[0]
    flattened = logits.sigmoid().reshape(batch_size, -1)
    topk = max(1, min(topk, flattened.shape[-1]))
    return flattened.topk(topk, dim=1).values.mean(dim=1)


def normalized_entropy(energy: torch.Tensor) -> float:
    energy = energy.detach().float().clamp_min_(0)
    if energy.numel() <= 1:
        return 0.0
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    probs = (energy / total).clamp_min_(1e-12)
    entropy = float(-(probs * probs.log()).sum())
    return entropy / math.log(energy.numel())


def _digitize(values: torch.Tensor, bins: int) -> torch.Tensor:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    if torch.allclose(values.min(), values.max()):
        return torch.zeros_like(values, dtype=torch.long)
    edges = torch.linspace(
        values.min(), values.max(), steps=bins + 1, device=values.device
    )
    return torch.bucketize(values, edges[1:-1], right=False)


def normalized_mutual_information(
    x: torch.Tensor, y: torch.Tensor, bins: int = 16, eps: float = 1e-12
) -> float:
    x_bins = _digitize(x, bins)
    y_bins = _digitize(y, bins)
    if x_bins.numel() == 0 or y_bins.numel() == 0:
        return 0.0

    x_bins = x_bins.cpu().numpy()
    y_bins = y_bins.cpu().numpy()
    joint = np.zeros((bins, bins), dtype=np.float64)
    for xb, yb in zip(x_bins, y_bins):
        joint[int(xb), int(yb)] += 1.0

    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    valid = joint > 0
    mutual_info = float(
        (
            joint[valid] * (np.log(joint[valid] + eps) - np.log((px @ py)[valid] + eps))
        ).sum()
    )
    hy = float(-(py[py > 0] * np.log(py[py > 0] + eps)).sum())
    if hy <= eps:
        return 0.0
    return mutual_info / hy
