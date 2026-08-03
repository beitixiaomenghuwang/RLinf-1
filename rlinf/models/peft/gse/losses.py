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


def _aggregate_router_metrics(
    layer_stats: list[dict[str, torch.Tensor]], metric_prefix: str
) -> dict[str, torch.Tensor]:
    """Aggregate compatible router statistics under one metric prefix."""
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
        f"{metric_prefix}/active_layers": reference.new_tensor(float(len(layer_stats))),
        f"{metric_prefix}/entropy": mean_entropy,
        f"{metric_prefix}/normalized_entropy": normalized_entropy,
        f"{metric_prefix}/selection_min": mean_selection.min(),
        f"{metric_prefix}/selection_max": mean_selection.max(),
        f"{metric_prefix}/selection_std": mean_selection.std(unbiased=False),
        f"{metric_prefix}/probability_min": mean_probability.min(),
        f"{metric_prefix}/probability_max": mean_probability.max(),
        f"{metric_prefix}/probability_std": mean_probability.std(unbiased=False),
    }
    for expert_index in range(num_experts):
        metrics[f"{metric_prefix}/expert_{expert_index}_selection"] = mean_selection[
            expert_index
        ]
        metrics[f"{metric_prefix}/expert_{expert_index}_probability"] = (
            mean_probability[expert_index]
        )
    return {name: value.detach() for name, value in metrics.items()}


def gse_router_metrics(model: nn.Module) -> dict[str, torch.Tensor]:
    """Aggregate detached router diagnostics, separating adapter domains."""
    layers_by_domain: dict[str, list[dict[str, torch.Tensor]]] = {}
    for _, layer in iter_gse_layers(model):
        if not layer.router_stats:
            continue
        domain = str(getattr(layer, "gse_domain", "default"))
        layers_by_domain.setdefault(domain, []).append(layer.router_stats)
    if not layers_by_domain:
        return {}

    metrics: dict[str, torch.Tensor] = {}
    if "action" in layers_by_domain:
        metrics.update(
            _aggregate_router_metrics(layers_by_domain["action"], "gse/router")
        )
    elif len(layers_by_domain) == 1:
        metrics.update(
            _aggregate_router_metrics(
                next(iter(layers_by_domain.values())), "gse/router"
            )
        )
    for domain, layer_stats in layers_by_domain.items():
        if len(layers_by_domain) > 1:
            metrics.update(
                _aggregate_router_metrics(layer_stats, f"gse/{domain}_router")
            )
    return metrics


def gse_task_router_statistics(
    model: nn.Module,
    task_ids: torch.Tensor,
    *,
    num_tasks: int,
    domain: str | None = None,
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
        and "probabilities" in layer.router_stats
        and (domain is None or getattr(layer, "gse_domain", None) == domain)
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
        statistics[
            f"{TASK_ROUTER_STATS_PREFIX}/task_{task_index:02d}/routing_count"
        ] = routing_counts[task_index]
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


def gse_layerwise_task_router_statistics(
    model: nn.Module,
    task_ids: torch.Tensor,
    *,
    num_tasks: int,
    domain: str | None = None,
) -> torch.Tensor:
    """Collect packed layer/task/router sufficient statistics.

    The result has shape ``[layers, tasks, 2 + 2 * experts]`` and stores
    routing count, selection total, probability sums, and selection counts.
    It can be accumulated across micro-batches and all-reduced once per PPO
    update, unlike a flat dictionary with one key per layer/task/expert.
    """
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    task_ids = task_ids.detach().reshape(-1).to(torch.long)
    if task_ids.numel() == 0:
        return torch.empty(0)
    if torch.any((task_ids < 0) | (task_ids >= num_tasks)):
        raise ValueError(f"task_ids must be in [0, {num_tasks})")

    layer_stats = [
        layer.router_stats
        for _, layer in iter_gse_layers(model)
        if layer.router_stats
        and "probabilities" in layer.router_stats
        and (domain is None or getattr(layer, "gse_domain", None) == domain)
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
        selected_experts = stats["selected_experts"].to(device)
        if probabilities.shape[-1] != num_experts:
            raise ValueError("All GSE layers must use the same expert count")
        if probabilities.shape[0] % task_ids.numel() != 0:
            raise ValueError(
                "Router items must be divisible by the task-id batch size, got "
                f"{probabilities.shape[0]} and {task_ids.numel()}"
            )
        repeats = probabilities.shape[0] // task_ids.numel()
        expanded_task_ids = task_ids.repeat_interleave(repeats)
        layer_packed = packed[layer_index]
        layer_packed[:, 0].index_add_(
            0,
            expanded_task_ids,
            torch.ones_like(expanded_task_ids, dtype=torch.float32),
        )
        layer_packed[:, 1].index_add_(
            0,
            expanded_task_ids,
            torch.full_like(
                expanded_task_ids,
                selected_experts.shape[-1],
                dtype=torch.float32,
            ),
        )
        layer_packed[:, 2 : 2 + num_experts].index_add_(
            0, expanded_task_ids, probabilities.float()
        )
        flat_tasks = expanded_task_ids[:, None].expand_as(selected_experts).reshape(-1)
        flat_experts = selected_experts.reshape(-1)
        flat_indices = flat_tasks * num_experts + flat_experts
        layer_packed[:, 2 + num_experts :].reshape(-1).index_add_(
            0,
            flat_indices,
            torch.ones_like(flat_indices, dtype=torch.float32),
        )
    return packed.detach()


def _distribution_information(
    distributions: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_distribution = torch.sum(distributions * weights[:, None], dim=0)
    eps = torch.finfo(torch.float64).eps
    mutual_information = torch.sum(
        weights[:, None]
        * distributions
        * (
            distributions.clamp_min(eps).log()
            - global_distribution.clamp_min(eps).log()[None]
        )
    )
    mean_distribution = 0.5 * (distributions + global_distribution.unsqueeze(0))
    js_divergence = 0.5 * torch.sum(
        distributions
        * (distributions.clamp_min(eps).log() - mean_distribution.clamp_min(eps).log()),
        dim=1,
    ) + 0.5 * torch.sum(
        global_distribution.unsqueeze(0)
        * (
            global_distribution.clamp_min(eps).log()
            - mean_distribution.clamp_min(eps).log()
        ),
        dim=1,
    )
    return mutual_information, torch.sum(js_divergence * weights)


def _cramers_v(contingency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and finite-sample bias-corrected Cramer's V."""
    contingency = contingency.to(dtype=torch.float64)
    nonempty_rows = contingency.sum(dim=1) > 0
    nonempty_columns = contingency.sum(dim=0) > 0
    contingency = contingency[nonempty_rows][:, nonempty_columns]
    num_rows, num_columns = contingency.shape
    sample_count = contingency.sum()
    if num_rows < 2 or num_columns < 2 or sample_count <= 1:
        zero = contingency.new_zeros(())
        return zero, zero

    expected = (
        contingency.sum(dim=1, keepdim=True)
        * contingency.sum(dim=0, keepdim=True)
        / sample_count
    )
    chi_squared = torch.sum((contingency - expected).square() / expected)
    phi_squared = chi_squared / sample_count
    raw = torch.sqrt(phi_squared / min(num_rows - 1, num_columns - 1))

    bias = (num_rows - 1) * (num_columns - 1) / (sample_count - 1)
    corrected_phi_squared = torch.clamp(phi_squared - bias, min=0.0)
    corrected_rows = num_rows - (num_rows - 1) ** 2 / (sample_count - 1)
    corrected_columns = num_columns - (num_columns - 1) ** 2 / (sample_count - 1)
    corrected_denominator = torch.minimum(corrected_rows - 1, corrected_columns - 1)
    corrected = torch.where(
        corrected_denominator > 0,
        torch.sqrt(corrected_phi_squared / corrected_denominator.clamp_min(1e-12)),
        corrected_phi_squared.new_zeros(()),
    )
    return raw, corrected


def _packed_task_router_information(
    statistics: torch.Tensor,
) -> dict[str, float]:
    if statistics.ndim != 2 or statistics.shape[1] < 4:
        raise ValueError(
            "Packed task-router statistics must have shape [tasks, 2 + 2 * experts]"
        )
    num_experts = (statistics.shape[1] - 2) // 2
    if statistics.shape[1] != 2 + 2 * num_experts:
        raise ValueError("Packed task-router statistics have an invalid width")

    statistics = statistics.to(device="cpu", dtype=torch.float64)
    routing_counts = statistics[:, 0]
    selection_totals = statistics[:, 1]
    covered = (routing_counts > 0) & (selection_totals > 0)
    if not covered.any():
        return {}

    routing_counts = routing_counts[covered]
    selection_totals = selection_totals[covered]
    probabilities = statistics[covered, 2 : 2 + num_experts] / routing_counts[:, None]
    selection_counts = statistics[covered, 2 + num_experts :]
    selections = selection_counts / selection_totals[:, None]
    weights = routing_counts / routing_counts.sum()
    selection_mi, selection_js = _distribution_information(selections, weights)
    probability_mi, probability_js = _distribution_information(probabilities, weights)
    cramers_v, adjusted_cramers_v = _cramers_v(selection_counts)
    normalizer = math.log(num_experts) if num_experts > 1 else 1.0
    return {
        "covered_tasks": float(covered.sum()),
        "nmi": float(selection_mi / normalizer) if num_experts > 1 else 0.0,
        "js": float(selection_js),
        "probability_nmi": float(probability_mi / normalizer)
        if num_experts > 1
        else 0.0,
        "probability_js": float(probability_js),
        "cramers_v": float(cramers_v),
        "adjusted_cramers_v": float(adjusted_cramers_v),
    }


def gse_task_router_metrics_from_tensor(statistics: torch.Tensor) -> dict[str, float]:
    """Convert packed task-router statistics into aggregate diagnostics."""
    if statistics.ndim == 3:
        statistics = statistics.sum(dim=0)
    information = _packed_task_router_information(statistics)
    if not information:
        return {}

    statistics = statistics.to(device="cpu", dtype=torch.float64)
    num_experts = (statistics.shape[1] - 2) // 2
    routing_counts = statistics[:, 0]
    selection_totals = statistics[:, 1]
    covered = (routing_counts > 0) & (selection_totals > 0)
    probabilities = (
        statistics[covered, 2 : 2 + num_experts] / routing_counts[covered, None]
    )
    selections = (
        statistics[covered, 2 + num_experts :] / selection_totals[covered, None]
    )
    metrics = {
        "gse/task_router/covered_tasks": information["covered_tasks"],
        "gse/task_router/normalized_mutual_information": information["nmi"],
        "gse/task_router/mean_js_divergence": information["js"],
        "gse/task_router/probability_normalized_mutual_information": information[
            "probability_nmi"
        ],
        "gse/task_router/probability_mean_js_divergence": information["probability_js"],
        "gse/task_router/nmi": information["nmi"],
        "gse/task_router/js": information["js"],
        "gse/task_router/prob_nmi": information["probability_nmi"],
        "gse/task_router/prob_js": information["probability_js"],
        "gse/task_router/cramers_v": information["cramers_v"],
        "gse/task_router/adjusted_cramers_v": information["adjusted_cramers_v"],
        "gse/task_router/mean_probability_std_across_tasks": float(
            probabilities.std(dim=0, unbiased=False).mean()
        ),
        "gse/task_router/mean_selection_std_across_tasks": float(
            selections.std(dim=0, unbiased=False).mean()
        ),
    }
    metrics["gse/task_router/prob_std"] = metrics[
        "gse/task_router/mean_probability_std_across_tasks"
    ]
    metrics["gse/task_router/select_std"] = metrics[
        "gse/task_router/mean_selection_std_across_tasks"
    ]

    covered_indices = torch.where(covered)[0]
    for row_index, task_index in enumerate(covered_indices.tolist()):
        prefix = f"gse/task_router/task_{task_index:02d}"
        metrics[f"{prefix}/routing_items"] = float(routing_counts[task_index])
        metrics[f"{prefix}/dominant_expert"] = float(selections[row_index].argmax())
        for expert_index in range(num_experts):
            metrics[f"{prefix}/expert_{expert_index}_selection"] = float(
                selections[row_index, expert_index]
            )
            metrics[f"{prefix}/expert_{expert_index}_probability"] = float(
                probabilities[row_index, expert_index]
            )
    return metrics


def gse_layerwise_task_router_metrics(
    statistics: torch.Tensor,
    *,
    informative_nmi_threshold: float = 0.01,
) -> dict[str, float]:
    """Summarize task information retained by individual router layers."""
    if informative_nmi_threshold < 0:
        raise ValueError("informative_nmi_threshold must be non-negative")
    if statistics.ndim != 3 or statistics.shape[0] == 0:
        return {}

    layer_information = [
        _packed_task_router_information(layer_statistics)
        for layer_statistics in statistics
    ]
    valid = [
        (layer_index, information)
        for layer_index, information in enumerate(layer_information)
        if information
    ]
    if not valid:
        return {}

    def summarize(name: str) -> dict[str, float]:
        values = torch.tensor(
            [information[name] for _, information in valid], dtype=torch.float64
        )
        top_index = int(values.argmax())
        return {
            f"gse/task_router/layerwise_{name}_mean": float(values.mean()),
            f"gse/task_router/layerwise_{name}_std": float(values.std(unbiased=False)),
            f"gse/task_router/layerwise_{name}_p90": float(torch.quantile(values, 0.9)),
            f"gse/task_router/layerwise_{name}_max": float(values[top_index]),
            f"gse/task_router/layerwise_{name}_top_layer": float(valid[top_index][0]),
        }

    metrics = {
        "gse/task_router/layerwise_active_layers": float(len(valid)),
        "gse/task_router/layerwise_informative_nmi_threshold": float(
            informative_nmi_threshold
        ),
        "gse/task_router/layerwise_informative_fraction": float(
            sum(
                information["nmi"] >= informative_nmi_threshold
                for _, information in valid
            )
            / len(valid)
        ),
        "gse/task_router/layerwise_probability_informative_fraction": float(
            sum(
                information["probability_nmi"] >= informative_nmi_threshold
                for _, information in valid
            )
            / len(valid)
        ),
    }
    for name in (
        "nmi",
        "js",
        "probability_nmi",
        "probability_js",
        "cramers_v",
        "adjusted_cramers_v",
    ):
        metrics.update(summarize(name))
    return metrics


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
                f"gse/task_router/task_{task_index:02d}/expert_{expert_index}_selection"
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
    mean_distribution = 0.5 * (selection_matrix + global_selection.unsqueeze(0))
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
            "gse/task_router/js": metrics["gse/task_router/mean_js_divergence"],
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
