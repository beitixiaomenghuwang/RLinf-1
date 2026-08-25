"""OpenVLA-OFT integration for Ortho-Hydra adapters."""

import logging
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from torch import nn

from rlinf.models.peft.ortho_hydra import (
    OrthoHydraConfig,
    OrthoHydraInjectionReport,
    inject_ortho_hydra,
    iter_ortho_hydra_layers,
    mark_only_ortho_hydra_as_trainable,
)

ALL_LINEAR_TARGET_MODULES = "all-linear"
WHOLE_MODEL_SCOPE = "whole_model"

_INTEGRATION_FIELDS = {
    "enabled",
    "scope",
    "target_modules",
    "exclude_modules",
    "freeze_vision_backbone",
    "train_value_head",
    "load_balancing_loss_coef",
    "log_router_metrics",
    "log_task_router_metrics",
    "log_layerwise_task_router_metrics",
    "task_router_num_tasks",
    "task_router_informative_nmi_threshold",
    "log_orthogonality",
}
_CONFIG_FIELDS = {field.name for field in fields(OrthoHydraConfig)}


def _build_config(
    config: Mapping[str, Any], semantic_embedding_dim: int
) -> OrthoHydraConfig:
    unknown = set(config) - _INTEGRATION_FIELDS - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown OpenVLA Ortho-Hydra fields: {sorted(unknown)}")
    values = {
        name: config[name]
        for name in _CONFIG_FIELDS
        if name in config and config[name] is not None
    }
    values["semantic_embedding_dim"] = semantic_embedding_dim
    values["record_routing_assignments"] = bool(
        config.get("log_task_router_metrics", False)
        or config.get("log_layerwise_task_router_metrics", False)
    )
    return OrthoHydraConfig(**values)


def _resolve_targets(model: nn.Module, freeze_vision_backbone: bool) -> tuple[str, ...]:
    excluded_root = "vision_backbone"
    return tuple(
        name
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and not (
            freeze_vision_backbone
            and (name == excluded_root or name.startswith(f"{excluded_root}."))
        )
    )


def configure_openvla_ortho_hydra(
    model: nn.Module,
    config: Mapping[str, Any],
) -> OrthoHydraInjectionReport:
    """Inject Ortho-Hydra into projector and language linear layers."""
    if not bool(config.get("enabled", False)):
        raise ValueError("configure_openvla_ortho_hydra requires enabled=true")
    if str(config.get("scope", WHOLE_MODEL_SCOPE)) != WHOLE_MODEL_SCOPE:
        raise ValueError("OpenVLA Ortho-Hydra currently requires scope='whole_model'")
    if (
        config.get("target_modules", ALL_LINEAR_TARGET_MODULES)
        != ALL_LINEAR_TARGET_MODULES
    ):
        raise ValueError(
            "OpenVLA Ortho-Hydra currently requires target_modules='all-linear'"
        )

    embedding = model.get_input_embeddings()
    semantic_dim = getattr(embedding, "embedding_dim", None)
    if semantic_dim is None and hasattr(embedding, "weight"):
        semantic_dim = int(embedding.weight.shape[-1])
    if semantic_dim is None:
        raise ValueError("Cannot determine OpenVLA text embedding dimension")
    core_config = _build_config(config, int(semantic_dim))
    freeze_vision = bool(config.get("freeze_vision_backbone", False))
    targets = _resolve_targets(model, freeze_vision)
    exclusions = tuple(config.get("exclude_modules", ()))
    report = inject_ortho_hydra(
        model,
        core_config,
        targets,
        exclude_modules=exclusions,
        strict=True,
    )
    for name, layer in iter_ortho_hydra_layers(model):
        domain = name.partition(".")[0]
        layer.ortho_hydra_domain = domain
        layer.adapter.ortho_hydra_domain = domain

    mark_only_ortho_hydra_as_trainable(model)
    if freeze_vision:
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
    model.ortho_hydra_injection_report = report
    logging.getLogger(__name__).info(
        "Injected Ortho-Hydra into %d OpenVLA linear layers (%d adapter parameters)",
        len(report.injected_module_names),
        report.adapter_parameters,
    )
    return report
