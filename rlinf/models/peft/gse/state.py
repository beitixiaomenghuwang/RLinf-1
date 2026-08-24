"""Checkpoint helpers for adapter-only GSE state."""

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn

from .injector import iter_gse_layers


def _adapter_prefixes(model: nn.Module) -> tuple[str, ...]:
    prefixes: list[str] = []
    for name, _ in iter_gse_layers(model):
        root = f"{name}." if name else ""
        prefixes.extend(
            (
                f"{root}adapter.generalized_experts.",
                f"{root}adapter.specialized_experts.",
                f"{root}adapter.router.",
                f"{root}adapter.semantic_router.",
            )
        )
    return tuple(prefixes)


def gse_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return experts, routers, and adapter buffers without base weights."""
    prefixes = _adapter_prefixes(model)
    return OrderedDict(
        (name, value)
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    )


def load_gse_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    """Load an adapter-only state dict and optionally validate its exact keys."""
    expected = set(gse_state_dict(model))
    supplied = set(state_dict)
    if strict and expected != supplied:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise RuntimeError(
            f"GSE state mismatch; missing={missing}, unexpected={unexpected}"
        )
    adapter_state = {
        name: value for name, value in state_dict.items() if name in expected
    }
    model.load_state_dict(adapter_state, strict=False)
