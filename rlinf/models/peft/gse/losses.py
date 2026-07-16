"""Auxiliary losses collected from GSE layers."""

import math
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


def gse_router_metrics(model: nn.Module) -> dict[str, torch.Tensor]:
    """Aggregate detached router diagnostics across active GSE layers.

    Layer statistics are weighted by their number of routing items so this also
    behaves sensibly when sequence lengths or batch sizes differ across layers.
    """
    layer_stats = [
        layer.router_stats
        for _, layer in iter_gse_layers(model)
        if layer.router_stats
    ]
    if not layer_stats:
        return {}

    selection_shapes = {
        tuple(stats["selection_fraction"].shape) for stats in layer_stats
    }
    if len(selection_shapes) != 1:
        raise ValueError(
            "All GSE layers must have the same number of specialized experts "
            "to aggregate router metrics"
        )

    reference = layer_stats[0]["entropy"]
    item_counts = torch.stack(
        [stats["num_routing_items"].to(reference) for stats in layer_stats]
    )
    weights = item_counts / item_counts.sum().clamp_min(1.0)
    selections = torch.stack(
        [stats["selection_fraction"].to(reference) for stats in layer_stats]
    )
    probabilities = torch.stack(
        [stats["mean_probability"].to(reference) for stats in layer_stats]
    )
    entropies = torch.stack([stats["entropy"].to(reference) for stats in layer_stats])

    mean_selection = torch.sum(selections * weights[:, None], dim=0)
    mean_probability = torch.sum(probabilities * weights[:, None], dim=0)
    mean_entropy = torch.sum(entropies * weights)
    num_experts = mean_selection.numel()
    normalized_entropy = (
        mean_entropy / math.log(num_experts)
        if num_experts > 1
        else mean_entropy.new_zeros(())
    )

    metrics = {
        "gse/router/active_layers": reference.new_tensor(float(len(layer_stats))),
        "gse/router/entropy": mean_entropy,
        "gse/router/normalized_entropy": normalized_entropy,
        "gse/router/selection_min": mean_selection.min(),
        "gse/router/selection_max": mean_selection.max(),
        "gse/router/selection_std": mean_selection.std(unbiased=False),
        "gse/router/probability_min": mean_probability.min(),
        "gse/router/probability_max": mean_probability.max(),
        "gse/router/probability_std": mean_probability.std(unbiased=False),
    }
    for expert_index in range(num_experts):
        metrics[f"gse/router/expert_{expert_index}_selection"] = mean_selection[
            expert_index
        ]
        metrics[f"gse/router/expert_{expert_index}_probability"] = mean_probability[
            expert_index
        ]
    return {name: value.detach() for name, value in metrics.items()}


def gse_auxiliary_loss(
    model: nn.Module,
    *,
    load_balancing_coefficient: float = 0.0,
    orthogonality_coefficient: float = 0.0,
    log_orthogonality: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build configurable GSE regularization and its training metrics."""
    if load_balancing_coefficient < 0 or orthogonality_coefficient < 0:
        raise ValueError("GSE auxiliary-loss coefficients must be non-negative")

    load_balancing = gse_load_balancing_loss(model)
    if orthogonality_coefficient > 0:
        orthogonality = gse_orthogonality_loss(model)
    elif log_orthogonality:
        with torch.no_grad():
            orthogonality = gse_orthogonality_loss(model)
    else:
        orthogonality = load_balancing.new_zeros(())

    weighted_load_balancing = load_balancing * load_balancing_coefficient
    weighted_orthogonality = orthogonality * orthogonality_coefficient
    auxiliary = weighted_load_balancing + weighted_orthogonality
    metrics = {
        "gse/load_balancing_loss": load_balancing.detach(),
        "gse/orthogonality_loss": orthogonality.detach(),
        "gse/weighted_load_balancing_loss": weighted_load_balancing.detach(),
        "gse/weighted_orthogonality_loss": weighted_orthogonality.detach(),
        "gse/auxiliary_loss": auxiliary.detach(),
    }
    metrics.update(gse_router_metrics(model))
    return auxiliary, metrics


def reset_gse_auxiliary_state(model: nn.Module) -> None:
    """Clear forward-dependent losses and router statistics."""
    for _, layer in iter_gse_layers(model):
        layer.reset_auxiliary_state()
