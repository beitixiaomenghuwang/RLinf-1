"""Utilities for injecting Ortho-Hydra into linear layers."""

from collections.abc import Collection, Iterator
from dataclasses import dataclass, replace

from torch import nn

from .config import OrthoHydraConfig
from .layer import OrthoHydraLinear


@dataclass(frozen=True)
class OrthoHydraInjectionReport:
    """Summarize an Ortho-Hydra injection."""

    injected_module_names: tuple[str, ...]
    adapter_parameters: int
    trainable_parameters: int


def iter_ortho_hydra_layers(
    model: nn.Module,
) -> Iterator[tuple[str, OrthoHydraLinear]]:
    """Yield every Ortho-Hydra layer and its qualified name."""
    for name, module in model.named_modules():
        if isinstance(module, OrthoHydraLinear):
            yield name, module


def _normalize(patterns: Collection[str] | str) -> tuple[str, ...]:
    return (patterns,) if isinstance(patterns, str) else tuple(patterns)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)


def inject_ortho_hydra(
    model: nn.Module,
    config: OrthoHydraConfig,
    target_modules: Collection[str] | str,
    *,
    exclude_modules: Collection[str] | str = (),
    strict: bool = True,
) -> OrthoHydraInjectionReport:
    """Replace matching linear modules with Ortho-Hydra wrappers."""
    targets = _normalize(target_modules)
    exclusions = _normalize(exclude_modules)
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and _matches(name, targets)
        and not _matches(name, exclusions)
    ]
    if strict and not candidates:
        raise ValueError(f"No linear modules matched targets: {sorted(targets)}")

    names = []
    for index, (name, base_layer) in enumerate(candidates):
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        seed = None if config.init_seed is None else config.init_seed + 2 * index
        setattr(
            parent,
            child_name,
            OrthoHydraLinear(base_layer, replace(config, init_seed=seed)),
        )
        names.append(name)

    adapter_parameters = sum(
        parameter.numel()
        for _, layer in iter_ortho_hydra_layers(model)
        for parameter in layer.adapter.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return OrthoHydraInjectionReport(
        injected_module_names=tuple(names),
        adapter_parameters=adapter_parameters,
        trainable_parameters=trainable_parameters,
    )


def mark_only_ortho_hydra_as_trainable(model: nn.Module) -> None:
    """Freeze the model and enable only Ortho-Hydra adapters."""
    model.requires_grad_(False)
    for _, layer in iter_ortho_hydra_layers(model):
        layer.adapter.requires_grad_(True)
