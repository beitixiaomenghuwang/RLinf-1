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

"""OpenPI-specific integration for GSE adapters."""

import logging
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from torch import nn

from rlinf.models.peft.gse import (
    GSEConfig,
    GSEInjectionReport,
    inject_gse,
    mark_only_gse_as_trainable,
)

DEFAULT_ACTION_EXPERT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_INTEGRATION_FIELDS = {
    "enabled",
    "target_modules",
    "exclude_modules",
    "train_value_head",
    "require_pi05",
    "load_balancing_loss_coef",
    "orthogonality_loss_coef",
    "log_router_metrics",
    "log_task_router_metrics",
    "task_router_num_tasks",
    "log_orthogonality",
}
_GSE_CONFIG_FIELDS = {field.name for field in fields(GSEConfig)}


def is_gse_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether an optional GSE configuration is enabled."""
    return config is not None and bool(config.get("enabled", False))


def state_dict_contains_gse(state_dict: Mapping[str, Any]) -> bool:
    """Identify a full checkpoint saved after GSE injection."""
    markers = (".generalized_experts.", ".specialized_experts.")
    return any(any(marker in name for marker in markers) for name in state_dict)


def _build_core_config(config: Mapping[str, Any]) -> GSEConfig:
    unknown_fields = set(config) - _INTEGRATION_FIELDS - _GSE_CONFIG_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown OpenPI GSE fields: {sorted(unknown_fields)}")
    values = {
        name: config[name]
        for name in _GSE_CONFIG_FIELDS
        if name in config and config[name] is not None
    }
    return GSEConfig(**values)


def get_action_expert_transformer(model: nn.Module) -> nn.Module:
    """Return the Gemma transformer used as the OpenPI action expert."""
    try:
        return model.paligemma_with_expert.gemma_expert.model
    except AttributeError as error:
        raise ValueError(
            "OpenPI GSE requires model.paligemma_with_expert.gemma_expert.model"
        ) from error


def configure_openpi_gse(
    model: nn.Module,
    config: Mapping[str, Any],
) -> GSEInjectionReport:
    """Inject GSE into the action transformer and freeze non-GSE actor weights."""
    if not is_gse_enabled(config):
        raise ValueError("configure_openpi_gse requires enabled=true")
    if bool(config.get("require_pi05", True)) and not bool(
        getattr(model, "pi05", False)
    ):
        raise ValueError("This GSE configuration requires a pi0.5 model")

    core_config = _build_core_config(config)
    action_expert = get_action_expert_transformer(model)
    target_modules = tuple(config.get("target_modules", DEFAULT_ACTION_EXPERT_TARGETS))
    exclude_modules = tuple(config.get("exclude_modules", ()))
    report = inject_gse(
        action_expert,
        core_config,
        target_modules,
        exclude_modules=exclude_modules,
        strict=True,
    )

    mark_only_gse_as_trainable(model)
    if bool(config.get("train_value_head", True)) and hasattr(model, "value_head"):
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
        "Injected GSE into %d OpenPI action-expert linear layers (%d parameters)",
        len(report.injected_module_names),
        report.adapter_parameters,
    )
    return report
