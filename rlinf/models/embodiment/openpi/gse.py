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
    iter_gse_layers,
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
DEFAULT_VLM_TARGETS = DEFAULT_ACTION_EXPERT_TARGETS

_INTEGRATION_FIELDS = {
    "enabled",
    "layer_indices",
    "target_modules",
    "exclude_modules",
    "train_action_adapters",
    "train_value_head",
    "require_pi05",
    "load_balancing_loss_coef",
    "orthogonality_loss_coef",
    "log_router_metrics",
    "log_task_router_metrics",
    "log_layerwise_task_router_metrics",
    "task_router_num_tasks",
    "task_router_informative_nmi_threshold",
    "log_orthogonality",
    "vlm",
}
_GSE_CONFIG_FIELDS = {field.name for field in fields(GSEConfig)}


def is_gse_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether an optional GSE configuration is enabled."""
    return config is not None and bool(config.get("enabled", False))


def state_dict_contains_gse(state_dict: Mapping[str, Any]) -> bool:
    """Identify a full checkpoint saved after GSE injection."""
    markers = (".generalized_experts.", ".specialized_experts.")
    return any(any(marker in name for marker in markers) for name in state_dict)


def state_dict_contains_vlm_gse(state_dict: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint contains GSE adapters in the VLM tower."""
    return any(
        ".paligemma." in name
        and (".generalized_experts." in name or ".specialized_experts." in name)
        for name in state_dict
    )


def _semantic_embedding_dim(model: nn.Module) -> int:
    """Return the PaliGemma text embedding width used for semantic routing."""
    try:
        embedding = model.paligemma_with_expert.paligemma.get_input_embeddings()
    except AttributeError as error:
        raise ValueError(
            "OpenPI GSE semantic conditioning requires PaliGemma input embeddings"
        ) from error
    dimension = getattr(embedding, "embedding_dim", None)
    if dimension is None and hasattr(embedding, "weight"):
        dimension = int(embedding.weight.shape[-1])
    if dimension is None:
        raise ValueError("Cannot determine OpenPI text embedding dimension")
    return int(dimension)


def _build_core_config(
    config: Mapping[str, Any],
    semantic_embedding_dim: int | None = None,
) -> GSEConfig:
    unknown_fields = set(config) - _INTEGRATION_FIELDS - _GSE_CONFIG_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown OpenPI GSE fields: {sorted(unknown_fields)}")
    values = {
        name: config[name]
        for name in _GSE_CONFIG_FIELDS
        if name in config and config[name] is not None
    }
    if bool(values.get("semantic_conditioning", False)):
        if semantic_embedding_dim is None:
            raise ValueError(
                "OpenPI GSE semantic_conditioning requires a resolvable text "
                "embedding dimension"
            )
        values["semantic_embedding_dim"] = semantic_embedding_dim
    values["record_routing_assignments"] = bool(
        config.get("log_task_router_metrics", False)
        or config.get("log_layerwise_task_router_metrics", False)
    )
    return GSEConfig(**values)


def get_action_expert_transformer(model: nn.Module) -> nn.Module:
    """Return the Gemma transformer used as the OpenPI action expert."""
    try:
        return model.paligemma_with_expert.gemma_expert.model
    except AttributeError as error:
        raise ValueError(
            "OpenPI GSE requires model.paligemma_with_expert.gemma_expert.model"
        ) from error


def get_vlm_transformer(model: nn.Module) -> nn.Module:
    """Return the Gemma language transformer inside the PaliGemma VLM."""
    try:
        return model.paligemma_with_expert.paligemma.language_model
    except AttributeError as error:
        raise ValueError(
            "OpenPI VLM GSE requires "
            "model.paligemma_with_expert.paligemma.language_model"
        ) from error


def _tag_gse_domain(model: nn.Module, domain: str) -> None:
    for _, layer in iter_gse_layers(model):
        layer.gse_domain = domain


def _resolve_layer_indices(transformer: nn.Module, configured: Any) -> tuple[int, ...]:
    try:
        num_layers = len(transformer.layers)
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "OpenPI VLM transformer must expose a layers sequence"
        ) from error
    raw_indices = (-1,) if configured is None else configured
    if isinstance(raw_indices, int):
        raw_indices = (raw_indices,)
    normalized = []
    for raw_index in raw_indices:
        index = int(raw_index)
        if index < 0:
            index += num_layers
        if not 0 <= index < num_layers:
            raise ValueError(
                f"VLM GSE layer index {raw_index} is outside [0, {num_layers})"
            )
        if index not in normalized:
            normalized.append(index)
    if not normalized:
        raise ValueError("VLM GSE layer_indices must not be empty")
    return tuple(normalized)


def _inject_vlm_layers(
    transformer: nn.Module,
    core_config: GSEConfig,
    layer_indices: tuple[int, ...],
    targets: tuple[str, ...],
    exclusions: tuple[str, ...],
) -> GSEInjectionReport:
    names: tuple[str, ...] = ()
    adapter_parameters = 0
    for position, layer_index in enumerate(layer_indices):
        layer_config = core_config
        if core_config.init_seed is not None:
            layer_config = replace(
                core_config,
                init_seed=core_config.init_seed + 10_000 * position,
            )
        layer_report = inject_gse(
            transformer.layers[layer_index],
            layer_config,
            targets,
            exclude_modules=exclusions,
            strict=True,
        )
        names += tuple(
            f"layers.{layer_index}.{name}"
            for name in layer_report.injected_module_names
        )
        adapter_parameters += layer_report.adapter_parameters
    return GSEInjectionReport(
        injected_module_names=names,
        adapter_parameters=adapter_parameters,
        trainable_parameters=0,
    )


def _merge_reports(
    action_report: GSEInjectionReport,
    vlm_report: GSEInjectionReport | None,
) -> GSEInjectionReport:
    names = tuple(
        f"action_expert.{name}" for name in action_report.injected_module_names
    )
    adapter_parameters = action_report.adapter_parameters
    if vlm_report is not None:
        names += tuple(f"vlm.{name}" for name in vlm_report.injected_module_names)
        adapter_parameters += vlm_report.adapter_parameters
    return GSEInjectionReport(
        injected_module_names=names,
        adapter_parameters=adapter_parameters,
        trainable_parameters=0,
    )


def _existing_gse_report(model: nn.Module) -> GSEInjectionReport:
    layers = tuple(iter_gse_layers(model))
    if not layers:
        raise ValueError("Expected existing action GSE adapters, but found none")
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


def configure_openpi_gse(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    action_already_injected: bool = False,
) -> GSEInjectionReport:
    """Inject action and optional VLM GSE, then select trainable adapters."""
    if not is_gse_enabled(config):
        raise ValueError("configure_openpi_gse requires enabled=true")
    if bool(config.get("require_pi05", True)) and not bool(
        getattr(model, "pi05", False)
    ):
        raise ValueError("This GSE configuration requires a pi0.5 model")

    semantic_conditioning = bool(config.get("semantic_conditioning", False)) or bool(
        is_gse_enabled(config.get("vlm", None))
        and isinstance(config.get("vlm"), Mapping)
        and config["vlm"].get("semantic_conditioning", False)
    )
    semantic_embedding_dim = (
        _semantic_embedding_dim(model) if semantic_conditioning else None
    )
    core_config = _build_core_config(config, semantic_embedding_dim)
    action_expert = get_action_expert_transformer(model)
    target_modules = tuple(config.get("target_modules", DEFAULT_ACTION_EXPERT_TARGETS))
    exclude_modules = tuple(config.get("exclude_modules", ()))
    if action_already_injected:
        action_report = _existing_gse_report(action_expert)
    else:
        action_report = inject_gse(
            action_expert,
            core_config,
            target_modules,
            exclude_modules=exclude_modules,
            strict=True,
        )
    _tag_gse_domain(action_expert, "action")

    vlm_config = config.get("vlm", None)
    vlm_report = None
    if is_gse_enabled(vlm_config):
        if not isinstance(vlm_config, Mapping):
            raise TypeError("OpenPI GSE vlm configuration must be a mapping")
        vlm = get_vlm_transformer(model)
        layer_indices = _resolve_layer_indices(
            vlm, vlm_config.get("layer_indices", (-1,))
        )
        vlm_targets = tuple(vlm_config.get("target_modules", DEFAULT_VLM_TARGETS))
        vlm_exclusions = tuple(vlm_config.get("exclude_modules", ()))
        vlm_report = _inject_vlm_layers(
            vlm,
            _build_core_config(vlm_config, semantic_embedding_dim),
            layer_indices,
            vlm_targets,
            vlm_exclusions,
        )
        _tag_gse_domain(vlm, "vlm")

    report = _merge_reports(action_report, vlm_report)

    mark_only_gse_as_trainable(model)
    if not bool(config.get("train_action_adapters", True)):
        if vlm_report is None:
            raise ValueError("train_action_adapters=false requires an enabled VLM GSE")
        for _, layer in iter_gse_layers(action_expert):
            layer.requires_grad_(False)
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
    model.gse_semantic_conditioning = semantic_conditioning
    logging.getLogger(__name__).info(
        "Injected GSE into %d OpenPI linear layers (%d action, %d VLM; %d parameters)",
        len(report.injected_module_names),
        len(action_report.injected_module_names),
        len(vlm_report.injected_module_names) if vlm_report is not None else 0,
        report.adapter_parameters,
    )
    return report
