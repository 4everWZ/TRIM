from __future__ import annotations

import copy
import gc
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Sequence

import torch
import torch.nn as nn

from trim.operators.common import Mamba
from trim.operators.metadata import prepare_state_space_checkpoint_state

__all__ = [
    "cleanup_cuda_memory",
    "eval_spatial_size_override",
    "load_dense_checkpoint",
    "mamba_fast_path_disabled",
    "set_model_eval_spatial_size",
]


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_dense_checkpoint(model: nn.Module, path: str) -> dict[str, Any]:
    if path.startswith("http"):
        state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)

    source_model_metadata = (
        state.get("model_metadata") if isinstance(state, dict) else None
    )
    if (
        isinstance(state, dict)
        and "ema" in state
        and isinstance(state["ema"], dict)
        and "module" in state["ema"]
    ):
        model_state = state["ema"]["module"]
        source = "ema.module"
    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        model_state = state["model"]
        source = "model"
    elif isinstance(state, dict) and all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        model_state = state
        source = "raw_state_dict"
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}")

    prepared_state, remap_info = prepare_state_space_checkpoint_state(
        model,
        remove_module_prefix(model_state),
        source_metadata=source_model_metadata,
    )
    missing, unexpected = model.load_state_dict(prepared_state, strict=False)
    load_info = {
        "source": source,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    load_info.update(remap_info)
    return load_info


def cleanup_cuda_memory(device: str | torch.device | None = None) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    if device is not None and torch.device(device).type != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        return
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def safe_gc_collect() -> None:
    try:
        gc.collect()
    except BaseException:
        pass


@contextmanager
def suppress_stdout_stderr():
    with (
        open(os.devnull, "w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        yield


@contextmanager
def mamba_fast_path_disabled(model: nn.Module):
    states = {}
    for module in model.modules():
        if Mamba is not None and isinstance(module, Mamba):
            states[id(module)] = module.use_fast_path
            module.use_fast_path = False
    try:
        yield
    finally:
        for module in model.modules():
            if Mamba is not None and isinstance(module, Mamba) and id(module) in states:
                module.use_fast_path = states[id(module)]


def _refresh_eval_spatial_size(
    module: nn.Module, spatial_size: Sequence[int] | None
) -> None:
    if not hasattr(module, "eval_spatial_size"):
        return
    module.eval_spatial_size = spatial_size
    if spatial_size is None or not hasattr(module, "_generate_anchors"):
        return
    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = "cpu"
    anchors, valid_mask = module._generate_anchors(device=device)
    if hasattr(module, "anchors"):
        module.anchors = anchors
    if hasattr(module, "valid_mask"):
        module.valid_mask = valid_mask


@contextmanager
def eval_spatial_size_override(model: nn.Module, spatial_size: Sequence[int]):
    previous_sizes: list[tuple[nn.Module, Any]] = []
    for module in model.modules():
        if hasattr(module, "eval_spatial_size"):
            previous_sizes.append((module, getattr(module, "eval_spatial_size")))
            _refresh_eval_spatial_size(module, tuple(spatial_size))
    try:
        yield
    finally:
        for module, previous_size in previous_sizes:
            _refresh_eval_spatial_size(module, previous_size)


def set_model_eval_spatial_size(
    model: nn.Module, spatial_size: Sequence[int] | None
) -> None:
    for module in model.modules():
        if hasattr(module, "eval_spatial_size"):
            _refresh_eval_spatial_size(
                module, None if spatial_size is None else tuple(spatial_size)
            )


def dispose_materialized(materialized: dict[str, Any] | None) -> None:
    if not materialized:
        return
    model = materialized.get("model")
    if isinstance(model, nn.Module):
        model.to("cpu")
    pruner = materialized.get("pruner")
    if pruner is not None and hasattr(pruner, "model") and hasattr(pruner, "criterion"):
        pruner.model = pruner.model.to("cpu")
        pruner.criterion = pruner.criterion.to("cpu")
    materialized.clear()
    safe_gc_collect()
