"""Checkpoint helpers for adapter-only MoORE state."""

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn

from .injector import iter_moore_layers


def moore_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    prefixes = tuple(
        f"{name + '.' if name else ''}adapter." for name, _ in iter_moore_layers(model)
    )
    return OrderedDict(
        (name, value)
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    )


def load_moore_state_dict(
    model: nn.Module, state_dict: Mapping[str, torch.Tensor], *, strict: bool = True
) -> None:
    expected = set(moore_state_dict(model))
    supplied = set(state_dict)
    if strict and expected != supplied:
        raise RuntimeError(
            f"MoORE state mismatch; missing={sorted(expected - supplied)}, "
            f"unexpected={sorted(supplied - expected)}"
        )
    model.load_state_dict(
        {name: value for name, value in state_dict.items() if name in expected},
        strict=False,
    )
