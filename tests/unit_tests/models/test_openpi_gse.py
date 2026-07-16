"""Integration tests for injecting GSE into the OpenPI action expert."""

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openpi.gse import (
    DEFAULT_ACTION_EXPERT_TARGETS,
    configure_openpi_gse,
    get_action_expert_transformer,
    state_dict_contains_gse,
)
from rlinf.models.peft.gse import GSELinear


class ToyAttention(nn.Module):
    """Attention-shaped module containing OpenPI target names."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(12, 12, bias=False)
        self.k_proj = nn.Linear(12, 12, bias=False)
        self.v_proj = nn.Linear(12, 12, bias=False)
        self.o_proj = nn.Linear(12, 12, bias=False)


class ToyMLP(nn.Module):
    """MLP-shaped module containing OpenPI target names."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(12, 24, bias=False)
        self.up_proj = nn.Linear(12, 24, bias=False)
        self.down_proj = nn.Linear(24, 12, bias=False)


class ToyDecoderLayer(nn.Module):
    """Minimal Gemma decoder layer structure."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()


class ToyOpenPi(nn.Module):
    """Minimal model exposing the π0.5 action-expert path."""

    def __init__(self, *, pi05: bool = True) -> None:
        super().__init__()
        self.pi05 = pi05
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Module()
        self.paligemma_with_expert.gemma_expert.model.layers = nn.ModuleList(
            [ToyDecoderLayer(), ToyDecoderLayer()]
        )
        self.paligemma_with_expert.paligemma = nn.Linear(12, 12)
        self.action_in_proj = nn.Linear(4, 12)
        self.action_out_proj = nn.Linear(12, 4)
        self.time_mlp_in = nn.Linear(12, 12)
        self.time_mlp_out = nn.Linear(12, 12)
        self.value_head = nn.Linear(12, 1)


def make_integration_config(**overrides: object) -> dict[str, object]:
    """Create a small enabled OpenPI GSE configuration."""
    config: dict[str, object] = {
        "enabled": True,
        "total_rank": 8,
        "lora_alpha": 8.0,
        "num_experts": 4,
        "num_generalized_experts": 1,
        "top_k": 2,
        "init_seed": 9,
        "target_modules": list(DEFAULT_ACTION_EXPERT_TARGETS),
        "train_value_head": True,
        "require_pi05": True,
    }
    config.update(overrides)
    return config


def test_configure_openpi_gse_targets_only_action_transformer() -> None:
    model = ToyOpenPi()
    original_q_proj = model.paligemma_with_expert.gemma_expert.model.layers[
        0
    ].self_attn.q_proj
    original_weight = original_q_proj.weight.detach().clone()
    inputs = torch.randn(2, 3, 12)
    expected_outputs = original_q_proj(inputs)

    report = configure_openpi_gse(model, make_integration_config())

    assert len(report.injected_module_names) == 14
    assert all(name.startswith("layers.") for name in report.injected_module_names)
    wrapped_q_proj = model.paligemma_with_expert.gemma_expert.model.layers[
        0
    ].self_attn.q_proj
    assert isinstance(wrapped_q_proj, GSELinear)
    torch.testing.assert_close(wrapped_q_proj.base_layer.weight, original_weight)
    torch.testing.assert_close(wrapped_q_proj(inputs), expected_outputs)

    assert isinstance(model.action_in_proj, nn.Linear)
    assert not model.action_in_proj.weight.requires_grad
    assert not model.paligemma_with_expert.paligemma.weight.requires_grad
    assert model.value_head.weight.requires_grad
    assert wrapped_q_proj.router.weight.requires_grad
    assert not wrapped_q_proj.base_layer.weight.requires_grad
    value_head_parameters = sum(
        parameter.numel() for parameter in model.value_head.parameters()
    )
    assert report.trainable_parameters == (
        report.adapter_parameters + value_head_parameters
    )


def test_openpi_gse_requires_pi05_by_default() -> None:
    with pytest.raises(ValueError, match="requires a pi0.5 model"):
        configure_openpi_gse(ToyOpenPi(pi05=False), make_integration_config())


def test_get_action_expert_reports_incompatible_model() -> None:
    with pytest.raises(ValueError, match="gemma_expert.model"):
        get_action_expert_transformer(nn.Linear(2, 2))


def test_state_dict_detection_distinguishes_sft_and_gse_checkpoints() -> None:
    sft_state = {
        "paligemma_with_expert.gemma_expert.model.layers.0.weight": torch.ones(1)
    }
    gse_state = {
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj."
        "generalized_experts.0.lora_a.weight": torch.ones(1)
    }

    assert not state_dict_contains_gse(sft_state)
    assert state_dict_contains_gse(gse_state)


def test_actor_and_rollout_build_identical_trainable_state_structure() -> None:
    actor = ToyOpenPi()
    rollout = ToyOpenPi()
    config = make_integration_config()
    configure_openpi_gse(actor, config)
    configure_openpi_gse(rollout, config)

    actor_trainable = [
        name for name, parameter in actor.named_parameters() if parameter.requires_grad
    ]
    rollout_trainable = [
        name
        for name, parameter in rollout.named_parameters()
        if parameter.requires_grad
    ]
    assert actor_trainable == rollout_trainable

    with torch.no_grad():
        first_router = next(
            module.router for module in actor.modules() if isinstance(module, GSELinear)
        )
        first_router.weight.add_(0.5)
    actor_state = actor.state_dict()
    assert state_dict_contains_gse(actor_state)
    rollout.load_state_dict(actor_state)
    for name in actor_trainable:
        torch.testing.assert_close(
            actor.get_parameter(name), rollout.get_parameter(name)
        )


def test_openpi_gse_rejects_unknown_configuration_fields() -> None:
    with pytest.raises(ValueError, match="Unknown OpenPI GSE fields"):
        configure_openpi_gse(ToyOpenPi(), make_integration_config(unknown_option=True))
