"""OpenPI action-only MoORE integration tests."""

import torch
from torch import nn

from rlinf.models.embodiment.openpi.moore import (
    configure_openpi_moore,
    state_dict_contains_moore,
)
from rlinf.models.peft.moore import MoORELinear


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(12, 12, bias=False)
        self.k_proj = nn.Linear(12, 12, bias=False)
        self.v_proj = nn.Linear(12, 12, bias=False)
        self.o_proj = nn.Linear(12, 12, bias=False)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(12, 24, bias=False),
                "up_proj": nn.Linear(12, 24, bias=False),
                "down_proj": nn.Linear(24, 12, bias=False),
            }
        )


class _OpenPi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pi05 = True
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Module()
        self.paligemma_with_expert.gemma_expert.model.layers = nn.ModuleList(
            [_Layer(), _Layer()]
        )
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.language_model = nn.Module()
        self.paligemma_with_expert.paligemma.language_model.layers = nn.ModuleList(
            [_Layer(), _Layer()]
        )
        self.value_head = nn.Linear(12, 1)


def _config() -> dict[str, object]:
    return {
        "enabled": True,
        "rank": 4,
        "num_experts": 3,
        "init_seed": 7,
        "train_value_head": True,
    }


def test_configure_openpi_moore_targets_action_expert_only() -> None:
    model = _OpenPi()
    reference = model.paligemma_with_expert.gemma_expert.model.layers[
        0
    ].self_attn.q_proj
    inputs = torch.randn(2, 3, 12)
    expected = reference(inputs)
    report = configure_openpi_moore(model, _config())

    assert len(report.injected_module_names) == 14
    assert isinstance(
        model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj,
        MoORELinear,
    )
    torch.testing.assert_close(
        model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj(
            inputs
        ),
        expected,
    )
    assert not model.paligemma_with_expert.paligemma.language_model.layers[
        0
    ].self_attn.q_proj.weight.requires_grad
    assert model.value_head.weight.requires_grad
    assert state_dict_contains_moore(model.state_dict())
