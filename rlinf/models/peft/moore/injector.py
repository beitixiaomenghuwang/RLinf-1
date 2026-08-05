"""Utilities for injecting MoORE layers into an existing model."""

from collections.abc import Collection, Iterator
from dataclasses import dataclass, replace

from torch import nn

from .config import MoOREConfig
from .layer import MoORELinear


@dataclass(frozen=True)
class MoOREInjectionReport:
    injected_module_names: tuple[str, ...]
    adapter_parameters: int
    trainable_parameters: int


def _normalize_patterns(patterns: Collection[str] | str) -> tuple[str, ...]:
    return (patterns,) if isinstance(patterns, str) else tuple(patterns)


def _matches(name: str, patterns: Collection[str]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in patterns)


def iter_moore_layers(model: nn.Module) -> Iterator[tuple[str, MoORELinear]]:
    for name, module in model.named_modules():
        if isinstance(module, MoORELinear):
            yield name, module


def inject_moore(
    model: nn.Module,
    config: MoOREConfig,
    target_modules: Collection[str] | str,
    *,
    exclude_modules: Collection[str] | str = (),
    strict: bool = True,
) -> MoOREInjectionReport:
    targets = _normalize_patterns(target_modules)
    exclusions = _normalize_patterns(exclude_modules)
    if not targets:
        raise ValueError("target_modules must not be empty")
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, MoORELinear) or not isinstance(module, nn.Linear):
            continue
        if _matches(name, targets) and not _matches(name, exclusions):
            candidates.append((name, module))
    if strict and not candidates:
        raise ValueError(f"No linear modules matched targets: {sorted(targets)}")
    names: list[str] = []
    for index, (name, base_layer) in enumerate(candidates):
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        seed = None if config.init_seed is None else config.init_seed + 2 * index
        setattr(
            parent,
            child_name,
            MoORELinear(base_layer, replace(config, init_seed=seed)),
        )
        names.append(name)
    adapter_parameters = sum(
        parameter.numel()
        for _, layer in iter_moore_layers(model)
        for parameter in layer.adapter.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return MoOREInjectionReport(tuple(names), adapter_parameters, trainable_parameters)


def mark_only_moore_as_trainable(model: nn.Module) -> None:
    model.requires_grad_(False)
    for _, layer in iter_moore_layers(model):
        layer.adapter.requires_grad_(True)
