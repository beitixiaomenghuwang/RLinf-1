"""Integration tests for whole-language-model OpenVLA-OFT GSE."""

import torch
from torch import nn

from rlinf.models.embodiment.openvla_oft.rlinf.gse import (
    DEFAULT_LLM_TARGET_MODULES,
    configure_openvla_gse,
)
from rlinf.models.peft.gse import GSELinear


class ToyDecoderLayer(nn.Module):
    """Small decoder layer with the projection names used by Llama."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)


class ToyOpenVLA(nn.Module):
    """Minimal OpenVLA shape with visual and language subtrees."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_backbone = nn.Linear(8, 16)
        self.projector = nn.Linear(16, 16)
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(
            [ToyDecoderLayer(), ToyDecoderLayer()]
        )
        self.language_model.lm_head = nn.Linear(16, 32, bias=False)


def make_config() -> dict[str, object]:
    """Return a compact whole-LLM GSE configuration."""
    return {
        "enabled": True,
        "target_modules": list(DEFAULT_LLM_TARGET_MODULES),
        "total_rank": 4,
        "lora_alpha": 4.0,
        "num_experts": 4,
        "num_generalized_experts": 1,
        "top_k": 2,
        "initialization": "svd",
        "init_seed": 42,
        "freeze_base": True,
    }


def test_openvla_gse_injects_every_llm_projection_and_keeps_oft_trainable() -> None:
    model = ToyOpenVLA()
    original = model.language_model.layers[0].q_proj
    inputs = torch.randn(2, 3, 16)
    expected = original(inputs)

    report = configure_openvla_gse(model, make_config())

    assert len(report.injected_module_names) == 2 * 7 + 1
    assert isinstance(model.language_model.layers[0].q_proj, GSELinear)
    assert isinstance(model.language_model.lm_head, GSELinear)
    torch.testing.assert_close(model.language_model.layers[0].q_proj(inputs), expected)
    assert not model.language_model.layers[0].q_proj.base_layer.weight.requires_grad
    assert model.language_model.layers[0].q_proj.router.weight.requires_grad

    # Non-LLM OpenVLA-OFT weights are intentionally full-finetuned.
    assert model.vision_backbone.weight.requires_grad
    assert model.projector.weight.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in model.language_model.named_parameters()
        if ".adapter." not in name
        and ".generalized_experts." not in name
        and ".specialized_experts." not in name
        and ".router." not in name
    )
