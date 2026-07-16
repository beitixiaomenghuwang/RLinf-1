"""Unit tests for generalized and specialized expert adapters."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from rlinf.models.peft.gse import (
    GSEConfig,
    GSELinear,
    gse_load_balancing_loss,
    gse_orthogonality_loss,
    gse_state_dict,
    inject_gse,
    joint_lora_a,
    load_gse_state_dict,
    mark_only_gse_as_trainable,
    orthogonality_error,
)


def make_config(**overrides: object) -> GSEConfig:
    """Create a small deterministic test configuration."""
    values = {
        "total_rank": 8,
        "lora_alpha": 8.0,
        "num_experts": 4,
        "num_generalized_experts": 1,
        "top_k": 2,
        "init_seed": 17,
    }
    values.update(overrides)
    return GSEConfig(**values)


def test_config_treats_rank_as_layer_total() -> None:
    config = make_config(total_rank=10)

    assert config.expert_ranks == (3, 3, 2, 2)
    assert sum(config.expert_ranks) == config.total_rank


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"total_rank": 3}, "at least rank 1"),
        ({"num_generalized_experts": 4}, "must be in"),
        ({"top_k": 4}, "top_k"),
        ({"top_k": 1}, "task loss can train the router"),
    ],
)
def test_config_rejects_invalid_expert_allocations(
    overrides: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


def test_orthogonal_zero_initialization_preserves_base_output() -> None:
    torch.manual_seed(4)
    base_layer = nn.Linear(12, 7)
    original = deepcopy(base_layer)
    layer = GSELinear(base_layer, make_config())
    inputs = torch.randn(3, 5, 12)

    torch.testing.assert_close(layer(inputs), original(inputs))
    assert not layer.base_layer.weight.requires_grad
    assert orthogonality_error(layer.all_experts).item() < 1e-5

    joint_a = joint_lora_a(layer.all_experts)
    assert joint_a.shape == (layer.config.total_rank, layer.in_features)
    for expert in layer.all_experts:
        assert torch.count_nonzero(expert.lora_b.weight) == 0


def test_sequence_router_makes_one_decision_per_batch_item() -> None:
    inputs = torch.randn(3, 5, 12)
    sequence_layer = GSELinear(
        nn.Linear(12, 7), make_config(routing_granularity="sequence")
    )
    token_layer = GSELinear(nn.Linear(12, 7), make_config(routing_granularity="token"))

    sequence_layer(inputs)
    token_layer(inputs)

    assert sequence_layer.router_stats["num_routing_items"].item() == 3
    assert token_layer.router_stats["num_routing_items"].item() == 15
    assert sequence_layer.router_stats["selection_fraction"].sum() == pytest.approx(1.0)


def test_zero_b_initialization_has_expected_two_stage_gradient_flow() -> None:
    layer = GSELinear(nn.Linear(12, 7), make_config())
    inputs = torch.randn(3, 5, 12)

    layer(inputs).sum().backward()

    assert any(
        torch.count_nonzero(expert.lora_b.weight.grad) > 0
        for expert in layer.all_experts
    )
    assert all(
        torch.count_nonzero(expert.lora_a.weight.grad) == 0
        for expert in layer.all_experts
    )
    assert torch.count_nonzero(layer.router.weight.grad) == 0

    layer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for expert in layer.all_experts:
            expert.lora_b.weight.normal_(std=0.01)
    layer(inputs).square().mean().backward()

    assert any(
        torch.count_nonzero(expert.lora_a.weight.grad) > 0
        for expert in layer.all_experts
        if expert.lora_a.weight.grad is not None
    )
    assert torch.count_nonzero(layer.router.weight.grad) > 0


class ToyModel(nn.Module):
    """Small model with separable VLM and action-expert subtrees."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm = nn.Sequential(nn.Linear(12, 12), nn.ReLU())
        self.action_expert = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 7),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the action path only."""
        return self.action_expert(inputs)


def test_injection_freezing_and_adapter_only_state_round_trip() -> None:
    model = ToyModel()
    report = inject_gse(
        model,
        make_config(),
        target_modules=("action_expert.0",),
    )
    mark_only_gse_as_trainable(model)

    assert report.injected_module_names == ("action_expert.0",)
    assert isinstance(model.action_expert[0], GSELinear)
    assert isinstance(model.vlm[0], nn.Linear)
    assert not model.vlm[0].weight.requires_grad
    assert not model.action_expert[0].base_layer.weight.requires_grad
    assert model.action_expert[0].router.weight.requires_grad

    saved_state = gse_state_dict(model)
    assert saved_state
    assert all("base_layer" not in name for name in saved_state)

    restored = ToyModel()
    inject_gse(
        restored,
        make_config(init_seed=999),
        target_modules=("action_expert.0",),
    )
    load_gse_state_dict(restored, saved_state)
    for name, value in gse_state_dict(restored).items():
        torch.testing.assert_close(value, saved_state[name])


def test_auxiliary_losses_are_finite_and_differentiable() -> None:
    model = ToyModel()
    inject_gse(model, make_config(), target_modules=("action_expert.0",))
    model(torch.randn(3, 5, 12))

    load_balance = gse_load_balancing_loss(model)
    orthogonality = gse_orthogonality_loss(model)

    assert torch.isfinite(load_balance)
    assert torch.isfinite(orthogonality)
    assert load_balance.requires_grad
    assert orthogonality.requires_grad


def test_injection_is_strict_when_no_linear_module_matches() -> None:
    with pytest.raises(ValueError, match="No linear modules matched"):
        inject_gse(ToyModel(), make_config(), target_modules=("missing",))
