# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for whole-language-model OpenVLA-OFT GSE."""

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openvla_oft.rlinf.gse import (
    ALL_LINEAR_TARGET_MODULES,
    DEFAULT_LLM_TARGET_MODULES,
    WHOLE_MODEL_SCOPE,
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


def test_openvla_gse_injects_every_linear_in_whole_model_scope() -> None:
    model = ToyOpenVLA()
    config = make_config()
    config.update(
        {
            "scope": WHOLE_MODEL_SCOPE,
            "target_modules": ALL_LINEAR_TARGET_MODULES,
        }
    )

    report = configure_openvla_gse(model, config)

    assert len(report.injected_module_names) == 2 * 7 + 3
    assert isinstance(model.vision_backbone, GSELinear)
    assert isinstance(model.projector, GSELinear)
    assert isinstance(model.language_model.lm_head, GSELinear)
    assert model.vision_backbone.gse_domain == "vision"
    assert model.projector.gse_domain == "projector"
    assert model.language_model.lm_head.gse_domain == "llm"
    assert report.trainable_parameters == report.adapter_parameters
    assert all(
        not layer.base_layer.weight.requires_grad
        for layer in model.modules()
        if isinstance(layer, GSELinear)
    )


def test_openvla_gse_accepts_training_diagnostic_fields() -> None:
    config = make_config()
    config.update(
        {
            "load_balancing_loss_coef": 0.0,
            "orthogonality_loss_coef": 0.0,
            "log_router_metrics": True,
            "log_orthogonality": False,
        }
    )

    configure_openvla_gse(ToyOpenVLA(), config)


def test_openvla_task_router_diagnostics_record_assignments() -> None:
    config = make_config()
    config.update(
        {
            "log_task_router_metrics": True,
            "log_layerwise_task_router_metrics": True,
            "task_router_num_tasks": 90,
        }
    )

    model = ToyOpenVLA()
    configure_openvla_gse(model, config)

    wrapped_layers = [
        module for module in model.modules() if isinstance(module, GSELinear)
    ]
    assert wrapped_layers
    assert all(layer.config.record_routing_assignments for layer in wrapped_layers)


def test_openvla_gse_rejects_unknown_fields() -> None:
    config = make_config()
    config["log_orthogonalty"] = False

    with pytest.raises(ValueError, match="log_orthogonalty"):
        configure_openvla_gse(ToyOpenVLA(), config)
