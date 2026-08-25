"""Losses and routing diagnostics for Ortho-Hydra."""

import math

import torch
from torch import nn

from rlinf.models.peft.gse.losses import (
    gse_layerwise_task_router_metrics,
    gse_task_router_metrics_from_tensor,
)

from .injector import iter_ortho_hydra_layers


def reset_ortho_hydra_auxiliary_state(model: nn.Module) -> None:
    """Clear losses and routing assignments retained by adapted layers."""
    for _, layer in iter_ortho_hydra_layers(model):
        layer.reset_auxiliary_state()


def ortho_hydra_orthogonality_error(model: nn.Module) -> torch.Tensor:
    """Average structural cross-expert subspace overlap."""
    values = [
        layer.adapter.orthogonality_error()
        for _, layer in iter_ortho_hydra_layers(model)
    ]
    if values:
        return torch.stack(values).mean()
    reference = next(model.parameters(), None)
    return torch.tensor(0.0) if reference is None else reference.new_zeros(())


def _router_metrics(model: nn.Module) -> dict[str, torch.Tensor]:
    stats = [
        layer.router_stats
        for _, layer in iter_ortho_hydra_layers(model)
        if layer.router_stats
    ]
    if not stats:
        return {}
    reference = stats[0]["entropy"]
    counts = torch.stack([item["num_routing_items"].to(reference) for item in stats])
    weights = counts / counts.sum().clamp_min(1)
    selections = torch.stack(
        [item["selection_fraction"].to(reference) for item in stats]
    )
    probabilities = torch.stack(
        [item["mean_probability"].to(reference) for item in stats]
    )
    entropies = torch.stack([item["entropy"].to(reference) for item in stats])
    selection = torch.sum(selections * weights[:, None], dim=0)
    probability = torch.sum(probabilities * weights[:, None], dim=0)
    entropy = torch.sum(entropies * weights)
    metrics = {
        "ortho_hydra/router/active_layers": reference.new_tensor(float(len(stats))),
        "ortho_hydra/router/entropy": entropy,
        "ortho_hydra/router/normalized_entropy": entropy
        / math.log(probability.numel()),
        "ortho_hydra/router/selection_min": selection.min(),
        "ortho_hydra/router/selection_max": selection.max(),
        "ortho_hydra/router/probability_min": probability.min(),
        "ortho_hydra/router/probability_max": probability.max(),
    }
    for index in range(probability.numel()):
        metrics[f"ortho_hydra/router/expert_{index}_selection"] = selection[index]
        metrics[f"ortho_hydra/router/expert_{index}_probability"] = probability[index]
    return {name: value.detach() for name, value in metrics.items()}


def ortho_hydra_auxiliary_loss(
    model: nn.Module,
    *,
    load_balancing_coefficient: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return weighted load balancing and detached diagnostics."""
    losses = [
        layer.load_balancing_loss
        for _, layer in iter_ortho_hydra_layers(model)
        if layer.load_balancing_loss is not None
    ]
    if losses:
        load_balance = torch.stack(losses).mean()
    else:
        reference = next(model.parameters(), None)
        load_balance = (
            torch.tensor(0.0) if reference is None else reference.new_zeros(())
        )
    weighted = load_balance * load_balancing_coefficient
    metrics = _router_metrics(model)
    metrics.update(
        {
            "ortho_hydra/load_balancing_loss": load_balance.detach(),
            "ortho_hydra/weighted_load_balancing_loss": weighted.detach(),
        }
    )
    return weighted, metrics


def ortho_hydra_layerwise_task_router_statistics(
    model: nn.Module,
    task_ids: torch.Tensor,
    *,
    num_tasks: int,
    domain: str | None = None,
) -> torch.Tensor:
    """Pack task/router sufficient statistics for distributed reduction."""
    task_ids = task_ids.detach().reshape(-1).to(torch.long)
    if task_ids.numel() == 0:
        return torch.empty(0)
    if torch.any((task_ids < 0) | (task_ids >= num_tasks)):
        raise ValueError(f"task_ids must be in [0, {num_tasks})")
    layer_stats = [
        layer.router_stats
        for _, layer in iter_ortho_hydra_layers(model)
        if layer.router_stats
        and "probabilities" in layer.router_stats
        and (domain is None or getattr(layer, "ortho_hydra_domain", None) == domain)
    ]
    if not layer_stats:
        return torch.empty(0)
    num_experts = int(layer_stats[0]["probabilities"].shape[-1])
    device = layer_stats[0]["probabilities"].device
    packed = torch.zeros(
        len(layer_stats),
        num_tasks,
        2 + 2 * num_experts,
        device=device,
        dtype=torch.float32,
    )
    task_ids = task_ids.to(device)
    for layer_index, stats in enumerate(layer_stats):
        probabilities = stats["probabilities"].to(device)
        selected = stats["selected_experts"].to(device)
        if probabilities.shape[0] != task_ids.numel():
            raise ValueError("Ortho-Hydra uses one routing item per task sequence")
        layer_packed = packed[layer_index]
        ones = torch.ones_like(task_ids, dtype=torch.float32)
        layer_packed[:, 0].index_add_(0, task_ids, ones)
        layer_packed[:, 1].index_add_(0, task_ids, ones * selected.shape[-1])
        layer_packed[:, 2 : 2 + num_experts].index_add_(
            0, task_ids, probabilities.float()
        )
        flat_tasks = task_ids[:, None].expand_as(selected).reshape(-1)
        flat_experts = selected.reshape(-1)
        layer_packed[:, 2 + num_experts :].index_put_(
            (flat_tasks, flat_experts),
            torch.ones_like(flat_tasks, dtype=torch.float32),
            accumulate=True,
        )
    return packed.detach()


def _rename_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        name.replace("gse/task_router", "ortho_hydra/task_router", 1): value
        for name, value in metrics.items()
    }


def ortho_hydra_task_router_metrics_from_tensor(
    statistics: torch.Tensor,
) -> dict[str, float]:
    """Finalize aggregate task/router diagnostics."""
    return _rename_metrics(gse_task_router_metrics_from_tensor(statistics))


def ortho_hydra_layerwise_task_router_metrics(
    statistics: torch.Tensor,
    *,
    informative_nmi_threshold: float = 0.01,
) -> dict[str, float]:
    """Finalize layerwise task/router diagnostics."""
    return _rename_metrics(
        gse_layerwise_task_router_metrics(
            statistics,
            informative_nmi_threshold=informative_nmi_threshold,
        )
    )
