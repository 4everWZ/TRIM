import sys

import pytest
import torch
from torch import nn

from trim import MLPWidthPruner, SwiGLUWidthPruner, TRIMPruner
from trim.allocation_types import AllocationCandidate, AllocationObject
from trim.operators.feed_forward import FeedForwardMLP, SwiGLUFeedForward
from trim.pruner import UnitCandidate
from trim.repair import budget_excess


def test_import_does_not_initialize_training_or_dataset_modules():
    assert not any(name.startswith(("engine",)) for name in sys.modules)


@pytest.mark.parametrize("kind", ["swiglu", "mlp"])
def test_closed_ffn_pruning_preserves_external_shape(kind):
    if kind == "swiglu":
        module = SwiGLUFeedForward(8, 16, 8)
        pruner = SwiGLUWidthPruner()
    else:
        module = FeedForwardMLP(8, 16, 8, num_layers=3)
        pruner = MLPWidthPruner()
    before = sum(p.numel() for p in module.parameters())
    pruner.prune_out_channels(module, [0, 2, 4, 6])
    assert pruner.get_out_channels(module) == 12
    assert sum(p.numel() for p in module.parameters()) < before
    assert module(torch.randn(2, 5, 8)).shape == (2, 5, 8)


def test_milp_requires_both_budgets_and_preserves_coupled_members():
    pruner = TRIMPruner(
        nn.Identity(),
        nn.Identity(),
        device="cpu",
        input_shape=(1, 3, 4, 4),
        verbose=False,
    )
    dense = UnitCandidate(1.0, 8, [], 0, 0.0, 0.0)
    partial = UnitCandidate(0.75, 6, [0, 1], 40, 0.1, 1.0)
    pruned = UnitCandidate(0.5, 4, [0, 1, 2, 3], 40, 0.4, 2.0)
    # The cheaper candidate meets only the parameter budget. The solver must
    # instead select the joint action satisfying both resource constraints.
    pruner.allocation_objects = [
        AllocationObject(
            name="shared-axis",
            unit_names=["left", "right"],
            kind="coupled",
            candidates=[
                AllocationCandidate(1.0, {"left": dense, "right": dense}, 0, 0.0, 0.0),
                AllocationCandidate(
                    0.75, {"left": partial, "right": partial}, 40, 0.1, 1.0
                ),
                AllocationCandidate(
                    0.5, {"left": pruned, "right": pruned}, 40, 0.4, 2.0
                ),
            ],
        )
    ]
    selected = pruner.solve_allocation(100, 1.0, 0.70, 0.70)
    assert selected == {"left": pruned, "right": pruned}
    with pytest.raises(RuntimeError, match="unreachable"):
        pruner.solve_allocation(100, 1.0, 0.10, 0.10)


def test_budget_repair_checks_measured_resources_separately():
    materialized = {"pruned_stats": {"params": 69, "flops_value": 0.72}}
    param_excess, flop_excess = budget_excess(
        materialized,
        dense_params=100,
        dense_gflops=1.0,
        target_param_ratio=0.70,
        target_gflops_ratio=0.70,
        param_metric="measured",
    )
    assert param_excess == 0
    assert flop_excess == pytest.approx(0.02)


def test_renamed_operator_artifact_roundtrip(tmp_path):
    from trim import load_pruned_artifact, save_pruned_artifact

    model = SwiGLUFeedForward(8, 16, 8).eval()
    inputs = torch.randn(2, 5, 8)
    expected = model(inputs).detach()
    save_pruned_artifact(model, tmp_path, manifest={}, summary={})
    restored, metadata = load_pruned_artifact(None, tmp_path)
    assert metadata["loaded_format"] == "full"
    assert restored.__class__.__module__ == "trim.operators.feed_forward"
    assert list(restored.state_dict()) == list(model.state_dict())
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
