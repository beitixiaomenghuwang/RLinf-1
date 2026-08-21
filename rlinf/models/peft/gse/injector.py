"""Utilities for injecting GSE layers into an existing model."""

from collections.abc import Collection, Iterator
from dataclasses import dataclass, replace

from torch import nn

from .config import GSEConfig
from .layer import GSELinear


@dataclass(frozen=True)
class GSEInjectionReport:
    """Summarize a completed GSE injection."""

    injected_module_names: tuple[str, ...]
    adapter_parameters: int
    trainable_parameters: int


def _normalize_patterns(patterns: Collection[str] | str) -> tuple[str, ...]:
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(patterns)


def _matches(name: str, patterns: Collection[str]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)


def iter_gse_layers(model: nn.Module) -> Iterator[tuple[str, GSELinear]]:
    """Yield all GSE layers and their fully qualified module names."""
    for name, module in model.named_modules():
        if isinstance(module, GSELinear):
            yield name, module


def inject_gse(
    model: nn.Module,
    config: GSEConfig,
    target_modules: Collection[str] | str,
    *,
    exclude_modules: Collection[str] | str = (),
    strict: bool = True,
) -> GSEInjectionReport:
    """Replace matching linear modules with GSE residual wrappers.

    Patterns match either a complete module name or a dotted-name suffix. Passing
    the action-expert subtree instead of the complete VLA is the safest way to
    prevent accidental VLM injection.
    """
    targets = _normalize_patterns(target_modules)
    exclusions = _normalize_patterns(exclude_modules)
    if not targets:
        raise ValueError("target_modules must not be empty")

    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, GSELinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if _matches(name, targets) and not _matches(name, exclusions):
            candidates.append((name, module))

    if strict and not candidates:
        raise ValueError(f"No linear modules matched targets: {sorted(targets)}")

    injected_names: list[str] = []
    for index, (name, base_layer) in enumerate(candidates):
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer_seed = None if config.init_seed is None else config.init_seed + 2 * index
        layer_config = replace(config, init_seed=layer_seed)
        setattr(parent, child_name, GSELinear(base_layer, layer_config))
        injected_names.append(name)

    adapter_parameters = sum(
        parameter.numel()
        for _, layer in iter_gse_layers(model)
        for child_name, child in layer.named_children()
        if child_name != "base_layer"
        for parameter in child.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return GSEInjectionReport(
        injected_module_names=tuple(injected_names),
        adapter_parameters=adapter_parameters,
        trainable_parameters=trainable_parameters,
    )


def mark_only_gse_as_trainable(model: nn.Module) -> None:
    """Freeze the complete model, then enable GSE expert and router parameters."""
    model.requires_grad_(False)
    for _, layer in iter_gse_layers(model):
        layer.generalized_experts.requires_grad_(True)
        layer.specialized_experts.requires_grad_(True)
        if layer.config.routing_mode != "uniform":
            layer.router.requires_grad_(True)
            layer.semantic_router.requires_grad_(True)
