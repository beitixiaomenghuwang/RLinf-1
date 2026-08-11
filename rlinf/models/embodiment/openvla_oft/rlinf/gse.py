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

"""OpenVLA-OFT integration for whole-language-model GSE adapters."""

import logging
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from torch import nn

from rlinf.models.peft.gse import (
    GSEConfig,
    GSEInjectionReport,
    inject_gse,
    iter_gse_layers,
    mark_only_gse_as_trainable,
)

DEFAULT_LLM_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

_INTEGRATION_FIELDS = {
    "enabled",
    "target_modules",
    "exclude_modules",
    "train_value_head",
    "load_balancing_loss_coef",
    "orthogonality_loss_coef",
    "log_router_metrics",
    "log_task_router_metrics",
    "log_layerwise_task_router_metrics",
    "task_router_num_tasks",
    "task_router_informative_nmi_threshold",
    "log_orthogonality",
}
_GSE_CONFIG_FIELDS = {field.name for field in fields(GSEConfig)}


def is_gse_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether an optional GSE configuration is enabled."""
    return config is not None and bool(config.get("enabled", False))


def _build_core_config(config: Mapping[str, Any]) -> GSEConfig:
    unknown_fields = set(config) - _INTEGRATION_FIELDS - _GSE_CONFIG_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown OpenVLA-OFT GSE fields: {sorted(unknown_fields)}")
    values = {
        name: config[name]
        for name in _GSE_CONFIG_FIELDS
        if name in config and config[name] is not None
    }
    # Task-conditioned diagnostics need the per-token routing assignments.
    # Keep this derived from the public logging switches so OpenVLA-OFT has the
    # same behavior as the OpenPI GSE integration.
    values["record_routing_assignments"] = bool(
        config.get("log_task_router_metrics", False)
        or config.get("log_layerwise_task_router_metrics", False)
    )
    return GSEConfig(**values)


def get_language_model(model: nn.Module) -> nn.Module:
    """Return the OpenVLA language model subtree used for GSE injection."""
    try:
        return model.language_model
    except AttributeError as error:
        raise ValueError("OpenVLA-OFT GSE requires model.language_model") from error


def _tag_gse_domain(model: nn.Module, domain: str) -> None:
    for _, layer in iter_gse_layers(model):
        layer.gse_domain = domain


def _existing_gse_report(model: nn.Module) -> GSEInjectionReport:
    layers = tuple(iter_gse_layers(model))
    if not layers:
        raise ValueError("Expected existing OpenVLA-OFT GSE adapters, but found none")
    return GSEInjectionReport(
        injected_module_names=tuple(name for name, _ in layers),
        adapter_parameters=sum(
            parameter.numel()
            for _, layer in layers
            for parameter in layer.adapter.parameters()
        ),
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )


def configure_openvla_gse(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    already_injected: bool = False,
) -> GSEInjectionReport:
    """Inject GSE into the complete LLM and keep non-LLM OFT weights trainable.

    The official dense SFT checkpoint is loaded by ``from_pretrained`` as
    ordinary model weights. This function only adds the new RL adapter and
    never loads or merges the repository's attached PEFT adapter.
    """
    if not is_gse_enabled(config):
        raise ValueError("configure_openvla_gse requires enabled=true")

    core_config = _build_core_config(config)
    language_model = get_language_model(model)
    target_modules = tuple(config.get("target_modules", DEFAULT_LLM_TARGET_MODULES))
    exclude_modules = tuple(config.get("exclude_modules", ()))
    if already_injected:
        report = _existing_gse_report(language_model)
    else:
        report = inject_gse(
            language_model,
            core_config,
            target_modules,
            exclude_modules=exclude_modules,
            strict=True,
        )
    _tag_gse_domain(language_model, "llm")

    # Freeze the original LLM while retaining all adapter parameters. Then
    # unfreeze every non-LLM subtree, which covers the visual encoder and OFT
    # projector/action components used by OpenVLA-OFT.
    mark_only_gse_as_trainable(model)
    for name, parameter in model.named_parameters():
        if not name.startswith("language_model."):
            parameter.requires_grad_(True)
    if bool(config.get("train_value_head", False)) and hasattr(model, "value_head"):
        model.value_head.requires_grad_(True)

    report = replace(
        report,
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    model.gse_injection_report = report
    logging.getLogger(__name__).info(
        "Injected GSE into %d OpenVLA-OFT LLM linear layers (%d adapter parameters; "
        "%d trainable parameters including visual/OFT weights)",
        len(report.injected_module_names),
        report.adapter_parameters,
        report.trainable_parameters,
    )
    return report
