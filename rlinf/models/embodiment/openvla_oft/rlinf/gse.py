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

"""OpenVLA-OFT integration for language-only or whole-model GSE adapters."""

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
ALL_LINEAR_TARGET_MODULES = "all-linear"
LANGUAGE_MODEL_SCOPE = "language_model"
WHOLE_MODEL_SCOPE = "whole_model"

_INTEGRATION_FIELDS = {
    "enabled",
    "scope",
    "target_modules",
    "exclude_modules",
    "freeze_vision_backbone",
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


def _build_core_config(
    config: Mapping[str, Any], *, semantic_embedding_dim: int | None = None
) -> GSEConfig:
    unknown_fields = set(config) - _INTEGRATION_FIELDS - _GSE_CONFIG_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown OpenVLA-OFT GSE fields: {sorted(unknown_fields)}")
    values = {
        name: config[name]
        for name in _GSE_CONFIG_FIELDS
        if name in config and config[name] is not None
    }
    if bool(values.get("semantic_conditioning", False)):
        values.setdefault("semantic_embedding_dim", semantic_embedding_dim)
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
        layer.adapter.gse_domain = domain


def _tag_whole_model_gse_domains(model: nn.Module) -> None:
    """Tag whole-model adapters by their top-level OpenVLA component."""
    domain_by_root = {
        "language_model": "llm",
        "vision_backbone": "vision",
        "projector": "projector",
    }
    for name, layer in iter_gse_layers(model):
        root = name.partition(".")[0]
        domain = domain_by_root.get(root, "model")
        layer.gse_domain = domain
        layer.adapter.gse_domain = domain


def _resolve_injection_scope(
    model: nn.Module, config: Mapping[str, Any]
) -> tuple[str, nn.Module]:
    scope = str(config.get("scope", LANGUAGE_MODEL_SCOPE))
    if scope == LANGUAGE_MODEL_SCOPE:
        return scope, get_language_model(model)
    if scope == WHOLE_MODEL_SCOPE:
        return scope, model
    raise ValueError(
        "OpenVLA-OFT GSE scope must be 'language_model' or 'whole_model', "
        f"got {scope!r}"
    )


def _resolve_target_modules(
    injection_root: nn.Module,
    config: Mapping[str, Any],
    *,
    excluded_roots: tuple[str, ...] = (),
) -> tuple[str, ...]:
    configured_targets = config.get("target_modules", DEFAULT_LLM_TARGET_MODULES)
    if configured_targets == ALL_LINEAR_TARGET_MODULES:
        return tuple(
            name
            for name, module in injection_root.named_modules()
            if name
            and isinstance(module, nn.Linear)
            and not any(
                name == root or name.startswith(f"{root}.") for root in excluded_roots
            )
        )
    targets = (
        (configured_targets,)
        if isinstance(configured_targets, str)
        else tuple(configured_targets)
    )
    if not excluded_roots:
        return targets
    return tuple(
        name
        for name, module in injection_root.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and any(name == target or name.endswith(f".{target}") for target in targets)
        and not any(
            name == root or name.startswith(f"{root}.") for root in excluded_roots
        )
    )


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
    """Inject GSE into the configured OpenVLA-OFT scope.

    The official dense SFT checkpoint is loaded by ``from_pretrained`` as
    ordinary model weights. This function only adds the new RL adapter and
    never loads or merges the repository's attached PEFT adapter. The default
    scope preserves the original language-only behavior. ``whole_model`` with
    ``all-linear`` wraps every selected linear layer and freezes every original
    model parameter. With ``freeze_vision_backbone=true``, the vision backbone
    is excluded from GSE injection and remains completely frozen.
    """
    if not is_gse_enabled(config):
        raise ValueError("configure_openvla_gse requires enabled=true")

    semantic_embedding_dim = None
    if bool(config.get("semantic_conditioning", False)):
        embedding_layer = model.get_input_embeddings()
        semantic_embedding_dim = getattr(embedding_layer, "embedding_dim", None)
        if semantic_embedding_dim is None and hasattr(embedding_layer, "weight"):
            semantic_embedding_dim = int(embedding_layer.weight.shape[-1])
    core_config = _build_core_config(
        config, semantic_embedding_dim=semantic_embedding_dim
    )
    scope, injection_root = _resolve_injection_scope(model, config)
    freeze_vision_backbone = bool(config.get("freeze_vision_backbone", False))
    excluded_roots = (
        ("vision_backbone",)
        if scope == WHOLE_MODEL_SCOPE and freeze_vision_backbone
        else ()
    )
    target_modules = _resolve_target_modules(
        injection_root,
        config,
        excluded_roots=excluded_roots,
    )
    exclude_modules = tuple(config.get("exclude_modules", ()))
    if already_injected:
        if freeze_vision_backbone and any(
            name == "vision_backbone" or name.startswith("vision_backbone.")
            for name, _ in iter_gse_layers(model)
        ):
            raise ValueError(
                "freeze_vision_backbone=True is incompatible with an already "
                "injected vision GSE"
            )
        report = _existing_gse_report(injection_root)
    else:
        report = inject_gse(
            injection_root,
            core_config,
            target_modules,
            exclude_modules=exclude_modules,
            strict=True,
        )
    if scope == WHOLE_MODEL_SCOPE:
        _tag_whole_model_gse_domains(model)
    else:
        _tag_gse_domain(injection_root, "llm")

    # Freeze all original parameters while retaining every GSE adapter. The
    # legacy language-only scope full-finetunes non-LLM OpenVLA-OFT weights.
    mark_only_gse_as_trainable(model)
    if scope == LANGUAGE_MODEL_SCOPE:
        for name, parameter in model.named_parameters():
            if not name.startswith("language_model."):
                parameter.requires_grad_(True)
    if freeze_vision_backbone:
        model.vision_backbone.requires_grad_(False)
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
        "Injected GSE into %d OpenVLA-OFT %s linear layers (%d adapter parameters; "
        "%d trainable parameters)",
        len(report.injected_module_names),
        scope,
        report.adapter_parameters,
        report.trainable_parameters,
    )
    return report
