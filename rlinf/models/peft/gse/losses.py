"""Auxiliary losses collected from GSE layers."""

import math
from collections.abc import Iterable, Mapping
from typing import Literal

import torch
from torch import nn

from .injector import iter_gse_layers

Reduction = Literal["mean", "sum"]
TASK_ROUTER_STATS_PREFIX = "gse/task_router_stats"


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


def gse_task_router_statistics(
    model: nn.Module,
    task_ids: torch.Tensor,
    *,
    num_tasks: int,
) -> dict[str, torch.Tensor]:
    """Collect count-weighted task/router sufficient statistics.

    The returned sums can be added across micro-batches and distributed ranks
    before ratios are computed, avoiding bias from uneven per-task batch sizes.
    """
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    task_ids = task_ids.detach().reshape(-1).to(torch.long)
    if task_ids.numel() == 0:
        return {}
    if torch.any((task_ids < 0) | (task_ids >= num_tasks)):
        raise ValueError(f"task_ids must be in [0, {num_tasks})")

    layer_stats = [
        layer.router_stats
        for _, layer in iter_gse_layers(model)
        if layer.router_stats
    ]
    if not layer_stats:
        return {}
    num_experts = int(layer_stats[0]["probabilities"].shape[-1])
    device = layer_stats[0]["probabilities"].device
    probability_sums = torch.zeros(num_tasks, num_experts, device=device)
    selection_counts = torch.zeros(num_tasks, num_experts, device=device)
    routing_counts = torch.zeros(num_tasks, device=device)
    selection_totals = torch.zeros(num_tasks, device=device)

    for stats in layer_stats:
        probabilities = stats["probabilities"].to(device)
        selected_experts = stats["selected_experts"].to(device)
        if probabilities.shape[-1] != num_experts:
            raise ValueError("All GSE layers must use the same expert count")
        if probabilities.shape[0] % task_ids.numel() != 0:
            raise ValueError(
                "Router items must be divisible by the task-id batch size, got "
                f"{probabilities.shape[0]} and {task_ids.numel()}"
            )
        repeats = probabilities.shape[0] // task_ids.numel()
        expanded_task_ids = task_ids.to(device).repeat_interleave(repeats)
        probability_sums.index_add_(0, expanded_task_ids, probabilities.float())
        routing_counts.index_add_(
            0,
            expanded_task_ids,
            torch.ones_like(expanded_task_ids, dtype=torch.float32),
        )
        flat_tasks = expanded_task_ids[:, None].expand_as(selected_experts).reshape(-1)
        flat_experts = selected_experts.reshape(-1)
        flat_indices = flat_tasks * num_experts + flat_experts
        selection_counts.view(-1).index_add_(
            0,
            flat_indices,
            torch.ones_like(flat_indices, dtype=torch.float32),
        )
        selection_totals.index_add_(
            0,
            expanded_task_ids,
            torch.full_like(
                expanded_task_ids,
                selected_experts.shape[-1],
                dtype=torch.float32,
            ),
        )

    statistics: dict[str, torch.Tensor] = {}
    for task_index in range(num_tasks):
        statistics[f"{TASK_ROUTER_STATS_PREFIX}/task_{task_index:02d}/routing_count"] = (
            routing_counts[task_index]
        )
        statistics[
            f"{TASK_ROUTER_STATS_PREFIX}/task_{task_index:02d}/selection_total"
        ] = selection_totals[task_index]
        for expert_index in range(num_experts):
            statistics[
                f"{TASK_ROUTER_STATS_PREFIX}/task_{task_index:02d}/"
                f"expert_{expert_index}_probability_sum"
            ] = probability_sums[task_index, expert_index]
            statistics[
                f"{TASK_ROUTER_STATS_PREFIX}/task_{task_index:02d}/"
                f"expert_{expert_index}_selection_count"
            ] = selection_counts[task_index, expert_index]
    return {name: value.detach() for name, value in statistics.items()}


def gse_task_router_metrics(
    statistics: Mapping[str, float],
    *,
    num_tasks: int,
    num_experts: int,
) -> dict[str, float]:
    """Convert globally summed task/router statistics into diagnostics."""
    task_probabilities = []
    task_selections = []
    task_weights = []
    metrics: dict[str, float] = {}
    for task_index in range(num_tasks):
        prefix = f"{TASK_ROUTER_STATS_PREFIX}/task_{task_index:02d}"
        routing_count = float(statistics.get(f"{prefix}/routing_count", 0.0))
        selection_total = float(statistics.get(f"{prefix}/selection_total", 0.0))
        if routing_count <= 0 or selection_total <= 0:
            continue
        probabilities = torch.tensor(
            [
                float(
                    statistics.get(
                        f"{prefix}/expert_{expert_index}_probability_sum", 0.0
                    )
                )
                / routing_count
                for expert_index in range(num_experts)
            ],
            dtype=torch.float64,
        )
        selections = torch.tensor(
            [
                float(
                    statistics.get(
                        f"{prefix}/expert_{expert_index}_selection_count", 0.0
                    )
                )
                / selection_total
                for expert_index in range(num_experts)
            ],
            dtype=torch.float64,
        )
        task_probabilities.append(probabilities)
        task_selections.append(selections)
        task_weights.append(routing_count)
        metrics[f"gse/task_router/task_{task_index:02d}/routing_items"] = routing_count
        metrics[f"gse/task_router/task_{task_index:02d}/dominant_expert"] = float(
            selections.argmax().item()
        )
        for expert_index in range(num_experts):
            metrics[
                f"gse/task_router/task_{task_index:02d}/"
                f"expert_{expert_index}_selection"
            ] = float(selections[expert_index])
            metrics[
                f"gse/task_router/task_{task_index:02d}/"
                f"expert_{expert_index}_probability"
            ] = float(probabilities[expert_index])

    if not task_probabilities:
        return {}
    probability_matrix = torch.stack(task_probabilities)
    selection_matrix = torch.stack(task_selections)
    weights = torch.tensor(task_weights, dtype=torch.float64)
    weights = weights / weights.sum()
    global_selection = torch.sum(selection_matrix * weights[:, None], dim=0)
    eps = torch.finfo(torch.float64).eps
    mutual_information = torch.sum(
        weights[:, None]
        * selection_matrix
        * (
            selection_matrix.clamp_min(eps).log()
            - global_selection.clamp_min(eps).log()[None]
        )
    )
    mean_distribution = 0.5 * (
        selection_matrix + global_selection.unsqueeze(0)
    )
    js_divergence = 0.5 * torch.sum(
        selection_matrix
        * (
            selection_matrix.clamp_min(eps).log()
            - mean_distribution.clamp_min(eps).log()
        ),
        dim=1,
    ) + 0.5 * torch.sum(
        global_selection.unsqueeze(0)
        * (
            global_selection.clamp_min(eps).log()
            - mean_distribution.clamp_min(eps).log()
        ),
        dim=1,
    )
    normalized_mi = (
        mutual_information / math.log(num_experts)
        if num_experts > 1
        else mutual_information.new_zeros(())
    )
    metrics.update(
        {
            "gse/task_router/covered_tasks": float(len(task_probabilities)),
            "gse/task_router/normalized_mutual_information": float(normalized_mi),
            "gse/task_router/mean_js_divergence": float(
                torch.sum(js_divergence * weights)
            ),
            "gse/task_router/mean_probability_std_across_tasks": float(
                probability_matrix.std(dim=0, unbiased=False).mean()
            ),
            "gse/task_router/mean_selection_std_across_tasks": float(
                selection_matrix.std(dim=0, unbiased=False).mean()
            ),
        }
    )
    metrics.update(
        {
            "gse/task_router/nmi": metrics[
                "gse/task_router/normalized_mutual_information"
            ],
            "gse/task_router/js": metrics[
                "gse/task_router/mean_js_divergence"
            ],
            "gse/task_router/prob_std": metrics[
                "gse/task_router/mean_probability_std_across_tasks"
            ],
            "gse/task_router/select_std": metrics[
                "gse/task_router/mean_selection_std_across_tasks"
            ],
        }
    )
    return metrics


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
