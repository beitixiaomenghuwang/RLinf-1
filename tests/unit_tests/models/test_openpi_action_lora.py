"""Tests for plain LoRA injection into the OpenPI action expert."""

import pytest
import torch
from torch import nn

pytest.importorskip("peft")

from rlinf.models.embodiment.openpi.lora import (  # noqa: E402
    DEFAULT_ACTION_LORA_TARGETS,
    configure_openpi_action_lora,
    state_dict_contains_action_lora,
)


class ToyBlock(nn.Module):
    """Minimal block containing all action-expert LoRA target names."""

    def __init__(self) -> None:
        super().__init__()
        for name in DEFAULT_ACTION_LORA_TARGETS:
            setattr(self, name, nn.Linear(12, 12, bias=False))


class ToyOpenPi(nn.Module):
    """Minimal Pi0.5-shaped model for action-LoRA integration tests."""

    def __init__(self, *, pi05: bool = True) -> None:
        super().__init__()
        self.pi05 = pi05
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Module()
        self.paligemma_with_expert.gemma_expert.model.layers = nn.ModuleList(
            [ToyBlock(), ToyBlock()]
        )
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.proj = nn.Linear(12, 12)
        self.action_in_proj = nn.Linear(4, 12)
        self.value_head = nn.Linear(12, 1)


def make_config(**overrides: object) -> dict[str, object]:
    """Create an enabled action-LoRA configuration."""
    config: dict[str, object] = {
        "is_lora": True,
        "lora_target": "action_expert",
        "lora_rank": 4,
        "lora_alpha": 4.0,
        "lora_init": "gaussian",
        "lora_target_modules": list(DEFAULT_ACTION_LORA_TARGETS),
        "lora_train_value_head": True,
    }
    config.update(overrides)
    return config


def test_action_lora_is_zero_output_and_freezes_non_adapters() -> None:
    model = ToyOpenPi()
    original = model.paligemma_with_expert.gemma_expert.model.layers[0].q_proj
    inputs = torch.randn(2, 3, 12)
    expected = original(inputs)

    report = configure_openpi_action_lora(model, make_config())

    wrapped = model.paligemma_with_expert.gemma_expert.model.layers[0].q_proj
    torch.testing.assert_close(wrapped(inputs), expected)
    assert len(report.injected_module_names) == 14
    assert report.adapter_parameters > 0
    assert not wrapped.base_layer.weight.requires_grad
    assert wrapped.lora_A["default"].weight.requires_grad
    assert wrapped.lora_B["default"].weight.requires_grad
    torch.testing.assert_close(
        wrapped.lora_B["default"].weight,
        torch.zeros_like(wrapped.lora_B["default"].weight),
    )
    assert not model.action_in_proj.weight.requires_grad
    assert not model.paligemma_with_expert.paligemma.proj.weight.requires_grad
    assert model.value_head.weight.requires_grad


def test_action_lora_state_dict_detection_is_domain_specific() -> None:
    action_state = {
        "paligemma_with_expert.gemma_expert.model.layers.0.q_proj."
        "lora_A.default.weight": torch.ones(1)
    }
    vlm_state = {
        "paligemma_with_expert.paligemma.language_model.layers.0.q_proj."
        "lora_A.default.weight": torch.ones(1)
    }

    assert state_dict_contains_action_lora(action_state)
    assert not state_dict_contains_action_lora(vlm_state)


def test_action_lora_full_state_round_trip() -> None:
    source = ToyOpenPi()
    configure_openpi_action_lora(source, make_config())
    source_layer = source.paligemma_with_expert.gemma_expert.model.layers[0].q_proj
    with torch.no_grad():
        source_layer.lora_B["default"].weight.normal_(std=0.1)
    inputs = torch.randn(2, 3, 12)
    expected = source_layer(inputs)
    state_dict = source.state_dict()

    target = ToyOpenPi()
    configure_openpi_action_lora(target, make_config())
    target.load_state_dict(state_dict)
    target_layer = target.paligemma_with_expert.gemma_expert.model.layers[0].q_proj

    torch.testing.assert_close(target_layer(inputs), expected)


def test_action_lora_requires_pi05_by_default() -> None:
    with pytest.raises(ValueError, match="requires a pi0.5 model"):
        configure_openpi_action_lora(ToyOpenPi(pi05=False), make_config())
