"""State-space model metadata collection and checkpoint remapping."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .scan import (
    _STATE_SPACE_CODE_TO_SCAN_POLICY,
    _STATE_SPACE_SCAN_POLICY_CODES,
    _default_scan_policy_for_branch_count,
)

__all__ = [
    "collect_state_space_model_metadata",
    "diff_state_space_model_metadata",
    "infer_state_space_model_metadata_from_state_dict",
    "prepare_state_space_checkpoint_state",
]

_STATE_SPACE_NONCAUSAL_ADAPTERS = ("none", "assm_lite")
_STATE_SPACE_NONCAUSAL_ADAPTER_CODES = {
    name: index for index, name in enumerate(_STATE_SPACE_NONCAUSAL_ADAPTERS)
}
_STATE_SPACE_CODE_TO_NONCAUSAL_ADAPTER = {
    code: name for name, code in _STATE_SPACE_NONCAUSAL_ADAPTER_CODES.items()
}
_STATE_SPACE_ABLATIONS = (
    "none",
    "without_sa",
    "without_spatial_mamba",
    "without_grca",
    "without_frequency_attention",
    "without_frequency_mamba",
    "without_frequency_dwconv",
)
_STATE_SPACE_ABLATION_CODES = {
    name: index for index, name in enumerate(_STATE_SPACE_ABLATIONS)
}
_STATE_SPACE_CODE_TO_ABLATION = {
    code: name for name, code in _STATE_SPACE_ABLATION_CODES.items()
}


def _state_space_module_state_key(module_name: str, suffix: str) -> str:
    return f"{module_name}.{suffix}" if module_name else suffix


def _state_space_branch_prefix(module_name: str) -> str:
    return f"{module_name}.branches." if module_name else "branches."


def _resolve_scan_policy_code(value: Any) -> str:
    code = int(value)
    if code not in _STATE_SPACE_CODE_TO_SCAN_POLICY:
        raise ValueError(f"Unsupported Mamba scan policy code: {code}")
    return _STATE_SPACE_CODE_TO_SCAN_POLICY[code]


def _resolve_noncausal_adapter_code(value: Any) -> str:
    code = int(value)
    if code not in _STATE_SPACE_CODE_TO_NONCAUSAL_ADAPTER:
        raise ValueError(f"Unsupported Mamba noncausal adapter code: {code}")
    return _STATE_SPACE_CODE_TO_NONCAUSAL_ADAPTER[code]


def _resolve_ablation_code(value: Any) -> str:
    code = int(value)
    if code not in _STATE_SPACE_CODE_TO_ABLATION:
        raise ValueError(f"Unsupported Mamba ablation code: {code}")
    return _STATE_SPACE_CODE_TO_ABLATION[code]


def infer_state_space_model_metadata_from_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    modules: dict[str, dict[str, Any]] = {}

    for key, value in state_dict.items():
        if key == "scan_policy_code" or key.endswith(".scan_policy_code"):
            module_name = (
                "" if key == "scan_policy_code" else key[: -len(".scan_policy_code")]
            )
            modules.setdefault(module_name, {})["fire_mamba_scan_policy"] = (
                _resolve_scan_policy_code(value)
            )
        elif key == "noncausal_adapter_code" or key.endswith(".noncausal_adapter_code"):
            module_name = (
                ""
                if key == "noncausal_adapter_code"
                else key[: -len(".noncausal_adapter_code")]
            )
            modules.setdefault(module_name, {})["fire_mamba_noncausal_adapter"] = (
                _resolve_noncausal_adapter_code(value)
            )
        elif key == "fire_mamba_ablation_code" or key.endswith(
            ".fire_mamba_ablation_code"
        ):
            module_name = (
                ""
                if key == "fire_mamba_ablation_code"
                else key[: -len(".fire_mamba_ablation_code")]
            )
            modules.setdefault(module_name, {})["fire_mamba_ablation"] = (
                _resolve_ablation_code(value)
            )
        elif key == "branch_count_buffer" or key.endswith(".branch_count_buffer"):
            module_name = (
                ""
                if key == "branch_count_buffer"
                else key[: -len(".branch_count_buffer")]
            )
            modules.setdefault(module_name, {})["fire_mamba_branch_count"] = int(value)
            modules.setdefault(module_name, {})["branch_count_from_metadata"] = True

    for key in state_dict.keys():
        if ".branches." in key:
            module_name, tail = key.split(".branches.", 1)
        elif key.startswith("branches."):
            module_name = ""
            tail = key[len("branches.") :]
        else:
            continue
        branch_str, separator, _ = tail.partition(".")
        if not separator or not branch_str.isdigit():
            continue
        branch_index = int(branch_str)
        module_metadata = modules.setdefault(module_name, {})
        module_metadata["fire_mamba_branch_count"] = max(
            int(module_metadata.get("fire_mamba_branch_count", 0)),
            branch_index + 1,
        )

    for module_metadata in modules.values():
        branch_count = int(module_metadata.get("fire_mamba_branch_count", 0))
        module_metadata.setdefault(
            "fire_mamba_scan_policy",
            _default_scan_policy_for_branch_count(branch_count),
        )
        module_metadata.setdefault("fire_mamba_noncausal_adapter", "none")
        module_metadata.setdefault("fire_mamba_ablation", "none")
        module_metadata.pop("branch_count_from_metadata", None)

    return {"fire_mamba_modules": modules}


def collect_state_space_model_metadata(model: nn.Module) -> dict[str, Any]:
    modules = {}
    for module_name, module in model.named_modules():
        # External host-model identity; retain it for checkpoint metadata compatibility.
        if module.__class__.__name__ == "FireMamba2D" and hasattr(
            module, "export_metadata"
        ):
            modules[module_name] = module.export_metadata()
    return {"fire_mamba_modules": modules}


def diff_state_space_model_metadata(
    source_metadata: dict[str, Any] | None,
    target_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    source_modules = dict((source_metadata or {}).get("fire_mamba_modules", {}))
    target_modules = dict((target_metadata or {}).get("fire_mamba_modules", {}))
    mismatches = {}

    for module_name, target in target_modules.items():
        source = source_modules.get(module_name)
        if source is None:
            mismatches[module_name] = {
                "missing_source_module": True,
                "target": target,
            }
            continue

        module_mismatches = {}
        for key in (
            "fire_mamba_scan_policy",
            "fire_mamba_noncausal_adapter",
            "fire_mamba_ablation",
            "fire_mamba_branch_count",
            "fire_mamba_dwt_band_mode",
        ):
            source_value = source.get(key)
            target_value = target.get(key)
            if key == "fire_mamba_ablation":
                source_value = source_value or "none"
                target_value = target_value or "none"
            if source_value != target_value:
                module_mismatches[key] = {
                    "source": source_value,
                    "target": target_value,
                }
        if module_mismatches:
            mismatches[module_name] = module_mismatches

    extra_modules = sorted(set(source_modules) - set(target_modules))
    if extra_modules:
        mismatches["extra_source_modules"] = extra_modules

    return mismatches


def _count_branch_keys_for_module(state_dict: dict[str, Any], module_name: str) -> int:
    prefix = _state_space_branch_prefix(module_name)
    branch_count = 0
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix) :]
        branch_str, _, _ = tail.partition(".")
        if branch_str.isdigit():
            branch_count = max(branch_count, int(branch_str) + 1)
    return branch_count


def _average_branch_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    reference = tensors[0]
    if not reference.dtype.is_floating_point and not reference.is_complex():
        return reference.clone()
    mean_tensor = torch.stack(
        [tensor.to(dtype=torch.float32) for tensor in tensors], dim=0
    ).mean(dim=0)
    return mean_tensor.to(dtype=reference.dtype)


def _remap_branch_parameters(
    state_dict: dict[str, Any],
    *,
    module_name: str,
    target_branch_count: int,
) -> dict[str, Any]:
    branch_prefix = _state_space_branch_prefix(module_name)
    grouped: dict[str, list[torch.Tensor]] = {}
    updated_state = dict(state_dict)
    delete_keys = []

    for key, value in state_dict.items():
        if not key.startswith(branch_prefix):
            continue
        tail = key[len(branch_prefix) :]
        branch_str, separator, suffix = tail.partition(".")
        if not separator or not branch_str.isdigit():
            continue
        grouped.setdefault(suffix, []).append(value)
        delete_keys.append(key)

    if not grouped:
        return updated_state

    for key in delete_keys:
        updated_state.pop(key, None)

    averaged_suffixes = {
        suffix: _average_branch_tensors(tensors) for suffix, tensors in grouped.items()
    }
    for branch_index in range(target_branch_count):
        for suffix, tensor in averaged_suffixes.items():
            updated_state[f"{branch_prefix}{branch_index}.{suffix}"] = tensor.clone()
    return updated_state


def prepare_state_space_checkpoint_state(
    model: nn.Module,
    state_dict: dict[str, Any],
    *,
    source_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_metadata = collect_state_space_model_metadata(model)
    inferred_source_metadata = (
        source_metadata or infer_state_space_model_metadata_from_state_dict(state_dict)
    )
    source_modules = dict(inferred_source_metadata.get("fire_mamba_modules", {}))
    target_modules = dict(target_metadata.get("fire_mamba_modules", {}))

    prepared_state = dict(state_dict)
    remapped_modules = {}

    for module_name, target in target_modules.items():
        source = dict(source_modules.get(module_name, {}))
        source_branch_count = int(
            source.get("fire_mamba_branch_count")
            or _count_branch_keys_for_module(prepared_state, module_name)
        )
        if source_branch_count <= 0:
            prepared_state[
                _state_space_module_state_key(module_name, "scan_policy_code")
            ] = torch.tensor(
                _STATE_SPACE_SCAN_POLICY_CODES[target["fire_mamba_scan_policy"]],
                dtype=torch.int64,
            )
            prepared_state[
                _state_space_module_state_key(module_name, "noncausal_adapter_code")
            ] = torch.tensor(
                _STATE_SPACE_NONCAUSAL_ADAPTER_CODES[
                    target["fire_mamba_noncausal_adapter"]
                ],
                dtype=torch.int64,
            )
            prepared_state[
                _state_space_module_state_key(module_name, "fire_mamba_ablation_code")
            ] = torch.tensor(
                _STATE_SPACE_ABLATION_CODES[target["fire_mamba_ablation"]],
                dtype=torch.int64,
            )
            prepared_state[
                _state_space_module_state_key(module_name, "branch_count_buffer")
            ] = torch.tensor(
                target["fire_mamba_branch_count"],
                dtype=torch.int64,
            )
            continue
        source_policy = source.get(
            "fire_mamba_scan_policy",
            _default_scan_policy_for_branch_count(source_branch_count),
        )
        source_adapter = source.get("fire_mamba_noncausal_adapter", "none")
        source_ablation = source.get("fire_mamba_ablation", "none")
        target_branch_count = int(target["fire_mamba_branch_count"])
        target_policy = target["fire_mamba_scan_policy"]
        target_adapter = target["fire_mamba_noncausal_adapter"]
        target_ablation = target["fire_mamba_ablation"]

        if (
            source_branch_count == target_branch_count
            and source_policy == target_policy
            and source_adapter == target_adapter
            and source_ablation == target_ablation
        ):
            continue

        prepared_state = _remap_branch_parameters(
            prepared_state,
            module_name=module_name,
            target_branch_count=target_branch_count,
        )
        prepared_state[
            _state_space_module_state_key(module_name, "scan_policy_code")
        ] = torch.tensor(
            _STATE_SPACE_SCAN_POLICY_CODES[target_policy],
            dtype=torch.int64,
        )
        prepared_state[
            _state_space_module_state_key(module_name, "noncausal_adapter_code")
        ] = torch.tensor(
            _STATE_SPACE_NONCAUSAL_ADAPTER_CODES[target_adapter],
            dtype=torch.int64,
        )
        prepared_state[
            _state_space_module_state_key(module_name, "fire_mamba_ablation_code")
        ] = torch.tensor(
            _STATE_SPACE_ABLATION_CODES[target_ablation],
            dtype=torch.int64,
        )
        prepared_state[
            _state_space_module_state_key(module_name, "branch_count_buffer")
        ] = torch.tensor(
            target_branch_count,
            dtype=torch.int64,
        )
        remapped_modules[module_name] = {
            "source_branch_count": source_branch_count,
            "target_branch_count": target_branch_count,
            "source_scan_policy": source_policy,
            "target_scan_policy": target_policy,
            "source_noncausal_adapter": source_adapter,
            "target_noncausal_adapter": target_adapter,
            "source_ablation": source_ablation,
            "target_ablation": target_ablation,
        }

    for module_name, target in target_modules.items():
        prepared_state.setdefault(
            _state_space_module_state_key(module_name, "scan_policy_code"),
            torch.tensor(
                _STATE_SPACE_SCAN_POLICY_CODES[target["fire_mamba_scan_policy"]],
                dtype=torch.int64,
            ),
        )
        prepared_state.setdefault(
            _state_space_module_state_key(module_name, "noncausal_adapter_code"),
            torch.tensor(
                _STATE_SPACE_NONCAUSAL_ADAPTER_CODES[
                    target["fire_mamba_noncausal_adapter"]
                ],
                dtype=torch.int64,
            ),
        )
        prepared_state.setdefault(
            _state_space_module_state_key(module_name, "fire_mamba_ablation_code"),
            torch.tensor(
                _STATE_SPACE_ABLATION_CODES[target["fire_mamba_ablation"]],
                dtype=torch.int64,
            ),
        )
        prepared_state.setdefault(
            _state_space_module_state_key(module_name, "branch_count_buffer"),
            torch.tensor(target["fire_mamba_branch_count"], dtype=torch.int64),
        )

    return prepared_state, {
        "source_model_metadata": inferred_source_metadata,
        "target_model_metadata": target_metadata,
        "remap_applied": bool(remapped_modules),
        "remapped_branch_modules": remapped_modules,
    }
