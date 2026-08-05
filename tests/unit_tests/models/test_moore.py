"""Unit tests for hidden-state-routed MoORE adapters."""

import torch
from torch import nn

from rlinf.models.peft.moore import (
    MoOREConfig,
    MoORELinear,
    inject_moore,
    load_moore_state_dict,
    mark_only_moore_as_trainable,
    moore_state_dict,
)


def test_svd_initialized_moore_preserves_frozen_base_and_spectrum():
    torch.manual_seed(3)
    base = nn.Linear(12, 7)
    reference = nn.Linear(12, 7)
    reference.load_state_dict(base.state_dict())
    layer = MoORELinear(base, MoOREConfig(rank=4, num_experts=3, init_seed=17))
    inputs = torch.randn(2, 5, 12)

    torch.testing.assert_close(layer(inputs), reference(inputs))
    assert not layer.base_layer.weight.requires_grad
    torch.testing.assert_close(layer.svd_S, layer.adapter.base_svd_S)
    assert torch.count_nonzero(layer.svd_S) > 0
    assert not layer.svd_S.requires_grad
    layer(inputs).sum().backward()
    assert layer.svd_S.grad is None


def test_svd_initialization_matches_source_moore_parameter_conventions():
    torch.manual_seed(13)
    base = nn.Linear(12, 7, bias=False)
    layer = MoORELinear(base, MoOREConfig(rank=4, num_experts=3, init_seed=17))
    _, singular_values, _ = torch.linalg.svd(
        base.weight.detach().float(), full_matrices=False
    )

    torch.testing.assert_close(layer.svd_S, singular_values[:7])
    torch.testing.assert_close(
        layer.adapter.householder[0], layer.adapter.householder[1]
    )
    torch.testing.assert_close(
        layer.adapter.householder[2], layer.adapter.householder[3]
    )
    assert torch.count_nonzero(layer.adapter.router_up.weight) == 0
    assert torch.count_nonzero(layer.adapter.router.weight) > 0


def test_moore_injection_freezes_non_adapter_parameters_and_round_trips_state():
    model = nn.Sequential(nn.Linear(12, 12), nn.Linear(12, 7))
    config = MoOREConfig(rank=4, num_experts=3, init_seed=11)
    report = inject_moore(model, config, target_modules="0")
    mark_only_moore_as_trainable(model)

    assert report.injected_module_names == ("0",)
    assert isinstance(model[0], MoORELinear)
    assert model[0].adapter.router.weight.requires_grad
    assert not model[1].weight.requires_grad
    saved = moore_state_dict(model)
    assert saved and all("base_layer" not in name for name in saved)

    restored = nn.Sequential(nn.Linear(12, 12), nn.Linear(12, 7))
    restored[0].load_state_dict(model[0].base_layer.state_dict())
    inject_moore(restored, config, target_modules="0")
    load_moore_state_dict(restored, saved)
    for name, value in moore_state_dict(restored).items():
        torch.testing.assert_close(value, saved[name])


def test_sequence_router_uses_hidden_state_without_task_ids():
    layer = MoORELinear(
        nn.Linear(12, 7),
        MoOREConfig(rank=4, num_experts=3, routing_granularity="sequence"),
    )
    output = layer(torch.randn(3, 5, 12))
    assert output.shape == (3, 5, 7)
