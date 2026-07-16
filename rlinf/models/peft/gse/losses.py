"""Auxiliary losses collected from GSE layers."""

from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn

from .injector import iter_gse_layers

Reduction = Literal["mean", "sum"]


def _reduce(
    losses: Iterable[torch.Tensor], reduction: Reduction, model: nn.Module
) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    values = list(losses)
    if not values:
        reference = next(model.parameters(), None)
        return torch.tensor(0.0) if reference is None else reference.new_zeros(())
    stacked = torch.stack(values)
    return stacked.mean() if reduction == "mean" else stacked.sum()


def gse_load_balancing_loss(
    model: nn.Module, reduction: Reduction = "mean"
) -> torch.Tensor:
    """Collect latest router load-balancing losses from all GSE layers."""
    return _reduce(
        (
            layer.load_balancing_loss
            for _, layer in iter_gse_layers(model)
            if layer.load_balancing_loss is not None
        ),
        reduction,
        model,
    )


def gse_orthogonality_loss(
    model: nn.Module, reduction: Reduction = "mean"
) -> torch.Tensor:
    """Collect differentiable A-factor orthogonality losses."""
    return _reduce(
        (layer.orthogonality_loss() for _, layer in iter_gse_layers(model)),
        reduction,
        model,
    )


def reset_gse_auxiliary_state(model: nn.Module) -> None:
    """Clear forward-dependent losses and router statistics."""
    for _, layer in iter_gse_layers(model):
        layer.reset_auxiliary_state()
