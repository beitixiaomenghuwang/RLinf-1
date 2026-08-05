"""OpenPI-specific integration for action-only MoORE adapters."""

import logging
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from torch import nn

from rlinf.models.embodiment.openpi.gse import (
    DEFAULT_ACTION_EXPERT_TARGETS,
    get_action_expert_transformer,
)
from rlinf.models.peft.moore import (
    MoOREConfig,
    MoOREInjectionReport,
    inject_moore,
    iter_moore_layers,
    mark_only_moore_as_trainable,
)

_INTEGRATION_FIELDS = {
    "enabled",
    "target_modules",
    "exclude_modules",
    "train_action_adapters",
    "train_value_head",
    "require_pi05",
}
_MOORE_CONFIG_FIELDS = {field.name for field in fields(MoOREConfig)}


def is_moore_enabled(config: Mapping[str, Any] | None) -> bool:
    return config is not None and bool(config.get("enabled", False))


def state_dict_contains_moore(state_dict: Mapping[str, Any]) -> bool:
    """Identify a full checkpoint saved after MoORE injection."""
    # ``svd_S`` is a frozen, non-persistent SVD buffer. Keep its old marker for
    # compatibility with checkpoints produced before that buffer was frozen.
    markers = (
        ".adapter.svd_S",
        ".adapter.router.weight",
        ".adapter.router_up.weight",
        ".adapter.householder",
    )
    return any(any(marker in name for marker in markers) for name in state_dict)


def _build_core_config(config: Mapping[str, Any]) -> MoOREConfig:
    unknown = set(config) - _INTEGRATION_FIELDS - _MOORE_CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown OpenPI MoORE fields: {sorted(unknown)}")
    values = {
        name: config[name]
        for name in _MOORE_CONFIG_FIELDS
        if name in config and config[name] is not None
    }
    return MoOREConfig(**values)


def _existing_report(model: nn.Module) -> MoOREInjectionReport:
    layers = tuple(iter_moore_layers(model))
    if not layers:
        raise ValueError("Expected existing action MoORE adapters, but found none")
    return MoOREInjectionReport(
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


def configure_openpi_moore(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    action_already_injected: bool = False,
) -> MoOREInjectionReport:
    """Inject MoORE into the OpenPI action expert and freeze all other weights."""
    if not is_moore_enabled(config):
        raise ValueError("configure_openpi_moore requires enabled=true")
    if bool(config.get("require_pi05", True)) and not bool(
        getattr(model, "pi05", False)
    ):
        raise ValueError("This MoORE configuration requires a pi0.5 model")

    action_expert = get_action_expert_transformer(model)
    core_config = _build_core_config(config)
    targets = tuple(config.get("target_modules", DEFAULT_ACTION_EXPERT_TARGETS))
    exclusions = tuple(config.get("exclude_modules", ()))
    if action_already_injected:
        report = _existing_report(action_expert)
    else:
        report = inject_moore(
            action_expert, core_config, targets, exclude_modules=exclusions, strict=True
        )
    mark_only_moore_as_trainable(model)
    if not bool(config.get("train_action_adapters", True)):
        for _, layer in iter_moore_layers(action_expert):
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
    model.moore_injection_report = report
    logging.getLogger(__name__).info(
        "Injected MoORE into %d OpenPI action linear layers (%d parameters)",
        len(report.injected_module_names),
        report.adapter_parameters,
    )
    return report
