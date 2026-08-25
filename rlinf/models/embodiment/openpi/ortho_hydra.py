"""OpenPI integration for Ortho-Hydra adapters."""

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
    "log_router_metrics",
    "log_task_router_metrics",
    "log_layerwise_task_router_metrics",
    "task_router_num_tasks",
    "task_router_informative_nmi_threshold",
    "log_orthogonality",
}
_CONFIG_FIELDS = {field.name for field in fields(OrthoHydraConfig)}


def is_ortho_hydra_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether Ortho-Hydra is enabled."""
    return config is not None and bool(config.get("enabled", False))


def state_dict_contains_ortho_hydra(state_dict: Mapping[str, Any]) -> bool:
    """Identify a checkpoint containing Ortho-Hydra rotations."""
    return any(
        ".adapter.s_q" in name
        or (
            ".adapter.experts." in name
            and (name.endswith(".s_p") or name.endswith(".s_q"))
        )
        for name in state_dict
    )


def _embedding_dim(model: nn.Module) -> int:
    try:
        embedding = model.paligemma_with_expert.paligemma.get_input_embeddings()
    except AttributeError as error:
        raise ValueError(
            "OpenPI Ortho-Hydra requires PaliGemma input embeddings"
        ) from error
    dimension = getattr(embedding, "embedding_dim", None)
    if dimension is None and hasattr(embedding, "weight"):
        dimension = int(embedding.weight.shape[-1])
    if dimension is None:
        raise ValueError("Cannot determine OpenPI text embedding dimension")
    return int(dimension)


def _build_config(
    config: Mapping[str, Any], semantic_embedding_dim: int
) -> OrthoHydraConfig:
    unknown = set(config) - _INTEGRATION_FIELDS - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown OpenPI Ortho-Hydra fields: {sorted(unknown)}")
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


def configure_openpi_ortho_hydra(
    model: nn.Module,
    config: Mapping[str, Any],
) -> OrthoHydraInjectionReport:
    """Inject Ortho-Hydra into the OpenPI action expert."""
    if not is_ortho_hydra_enabled(config):
        raise ValueError("configure_openpi_ortho_hydra requires enabled=true")
    if bool(config.get("require_pi05", True)) and not bool(
        getattr(model, "pi05", False)
    ):
        raise ValueError("This Ortho-Hydra configuration requires a pi0.5 model")
    try:
        action_expert = model.paligemma_with_expert.gemma_expert.model
    except AttributeError as error:
        raise ValueError("OpenPI Ortho-Hydra requires gemma_expert.model") from error

    core_config = _build_config(config, _embedding_dim(model))
    report = inject_ortho_hydra(
        action_expert,
        core_config,
        tuple(config.get("target_modules", DEFAULT_ACTION_EXPERT_TARGETS)),
        exclude_modules=tuple(config.get("exclude_modules", ())),
        strict=True,
    )
    for _, layer in iter_ortho_hydra_layers(action_expert):
        layer.ortho_hydra_domain = "action"
        layer.adapter.ortho_hydra_domain = "action"

    mark_only_ortho_hydra_as_trainable(model)
    if bool(config.get("train_value_head", True)) and hasattr(model, "value_head"):
        model.value_head.requires_grad_(True)
    report = replace(
        report,
        injected_module_names=tuple(
            f"action_expert.{name}" for name in report.injected_module_names
        ),
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    model.ortho_hydra_injection_report = report
    model.ortho_hydra_semantic_conditioning = True
    logging.getLogger(__name__).info(
        "Injected Ortho-Hydra into %d OpenPI action linear layers "
        "(%d adapter parameters)",
        len(report.injected_module_names),
        report.adapter_parameters,
    )
    return report
