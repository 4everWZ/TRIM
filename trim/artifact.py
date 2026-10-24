from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch_pruning as tp

from trim.operators.metadata import (
    collect_state_space_model_metadata,
    diff_state_space_model_metadata,
)


def de_parallel(model):
    """Unwrap PyTorch DataParallel or DistributedDataParallel models."""
    parallel_types = (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    return model.module if type(model) in parallel_types else model


def save_pruned_artifact(
    model, artifact_dir: str | Path, manifest: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Path]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "manifest": artifact_dir / "manifest.json",
        "summary": artifact_dir / "summary.json",
        "tp": artifact_dir / "pruned.tp.pth",
        "full": artifact_dir / "pruned.full.pth",
    }

    module = de_parallel(model)
    torch.save(tp.state_dict(module), paths["tp"])
    torch.save(copy.deepcopy(module).cpu(), paths["full"])

    manifest_payload = dict(manifest)
    manifest_payload.update(
        {
            "artifact_dir": str(artifact_dir),
            "pruned_tp_path": paths["tp"].name,
            "pruned_full_path": paths["full"].name,
            "summary_path": paths["summary"].name,
        }
    )
    paths["manifest"].write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return paths


def load_pruned_artifact(
    base_model,
    artifact_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    summary_path = artifact_dir / "summary.json"
    tp_path = artifact_dir / "pruned.tp.pth"
    full_path = artifact_dir / "pruned.full.pth"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing pruning manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    metadata = {
        "manifest": manifest,
        "summary": summary,
        "loaded_format": None,
        "tp_error": None,
        "artifact_model_metadata": manifest.get("model_metadata", {}),
    }
    base_model_metadata = (
        collect_state_space_model_metadata(base_model) if base_model is not None else {}
    )
    metadata["base_model_metadata"] = base_model_metadata
    compatibility = (
        diff_state_space_model_metadata(
            metadata["artifact_model_metadata"],
            base_model_metadata,
        )
        if base_model is not None
        else {}
    )
    metadata["compatibility"] = compatibility
    metadata["tp_skipped_due_to_scan_mismatch"] = False

    if base_model is not None and tp_path.exists() and not compatibility:
        try:
            tp_state = torch.load(
                tp_path, map_location=map_location, weights_only=False
            )
            tp.load_state_dict(base_model, tp_state)
            metadata["loaded_format"] = "tp"
            return base_model, metadata
        except Exception as exc:  # pragma: no cover - exercised by fallback tests
            metadata["tp_error"] = str(exc)
    elif base_model is not None and compatibility:
        metadata["tp_skipped_due_to_scan_mismatch"] = True

    if not full_path.exists():
        raise FileNotFoundError(f"Missing full fallback artifact: {full_path}")

    model = torch.load(full_path, map_location=map_location, weights_only=False)
    metadata["loaded_format"] = "full"
    return model, metadata
