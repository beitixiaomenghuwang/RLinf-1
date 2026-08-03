# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurable advantage normalization for multi-task embodied RL."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

AdvantageNormalizationMode = Literal["none", "global", "per_task"]
AdvantageNormalizationFallback = Literal["global", "identity"]


@dataclass(frozen=True)
class AdvantageNormalizationConfig:
    """Resolved advantage-normalization settings."""

    mode: AdvantageNormalizationMode
    num_tasks: int | None = None
    min_count: int = 2
    eps: float = 1e-5
    fallback: AdvantageNormalizationFallback = "global"


def resolve_advantage_normalization_config(
    algorithm_cfg: Mapping[str, Any],
) -> AdvantageNormalizationConfig:
    """Resolve new normalization modes while preserving legacy boolean configs."""
    enabled = bool(algorithm_cfg.get("normalize_advantages", True))
    configured_mode = algorithm_cfg.get("advantage_normalization_mode")
    if configured_mode is None:
        mode = "global" if enabled else "none"
    else:
        mode = str(configured_mode).lower()

    valid_modes = {"none", "global", "per_task"}
    if mode not in valid_modes:
        raise ValueError(
            "algorithm.advantage_normalization_mode must be one of "
            f"{sorted(valid_modes)}, got {mode!r}"
        )
    if not enabled and mode != "none":
        raise ValueError(
            "algorithm.normalize_advantages=False is incompatible with "
            f"advantage_normalization_mode={mode!r}"
        )

    fallback = str(
        algorithm_cfg.get("advantage_normalization_fallback", "global")
    ).lower()
    valid_fallbacks = {"global", "identity"}
    if fallback not in valid_fallbacks:
        raise ValueError(
            "algorithm.advantage_normalization_fallback must be one of "
            f"{sorted(valid_fallbacks)}, got {fallback!r}"
        )

    min_count = int(algorithm_cfg.get("advantage_normalization_min_count", 2))
    eps = float(algorithm_cfg.get("advantage_normalization_eps", 1e-5))
    if min_count < 2:
        raise ValueError("algorithm.advantage_normalization_min_count must be >= 2")
    if eps <= 0:
        raise ValueError("algorithm.advantage_normalization_eps must be positive")

    num_tasks = algorithm_cfg.get("advantage_normalization_num_tasks")
    if mode == "per_task":
        if num_tasks is None:
            raise ValueError(
                "algorithm.advantage_normalization_num_tasks is required when "
                "advantage_normalization_mode='per_task'"
            )
        num_tasks = int(num_tasks)
        if num_tasks <= 0:
            raise ValueError(
                "algorithm.advantage_normalization_num_tasks must be positive"
            )
    elif num_tasks is not None:
        num_tasks = int(num_tasks)

    return AdvantageNormalizationConfig(
        mode=mode,
        num_tasks=num_tasks,
        min_count=min_count,
        eps=eps,
        fallback=fallback,
    )


def task_advantage_stats(
    advantages: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    num_tasks: int,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-task ``(count, sum, squared_sum)`` sufficient statistics."""
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")

    flat_advantages = advantages.detach().reshape(-1).to(torch.float64)
    flat_task_ids = (
        task_ids.detach()
        .reshape(-1)
        .to(device=flat_advantages.device, dtype=torch.long)
    )
    if flat_task_ids.numel() != flat_advantages.numel():
        raise ValueError(
            "task_ids and advantages must have the same number of elements, got "
            f"{flat_task_ids.numel()} and {flat_advantages.numel()}"
        )
    if flat_task_ids.numel() > 0 and torch.any(
        (flat_task_ids < 0) | (flat_task_ids >= num_tasks)
    ):
        raise ValueError(f"task_ids must be in [0, {num_tasks})")

    if loss_mask is not None:
        flat_mask = (
            loss_mask.detach()
            .reshape(-1)
            .to(device=flat_advantages.device, dtype=torch.bool)
        )
        if flat_mask.numel() != flat_advantages.numel():
            raise ValueError(
                "loss_mask and advantages must have the same number of elements, got "
                f"{flat_mask.numel()} and {flat_advantages.numel()}"
            )
        flat_advantages = flat_advantages[flat_mask]
        flat_task_ids = flat_task_ids[flat_mask]

    stats = torch.zeros(
        num_tasks, 3, device=flat_advantages.device, dtype=torch.float64
    )
    if flat_advantages.numel() == 0:
        return stats
    stats[:, 0].index_add_(0, flat_task_ids, torch.ones_like(flat_advantages))
    stats[:, 1].index_add_(0, flat_task_ids, flat_advantages)
    stats[:, 2].index_add_(0, flat_task_ids, flat_advantages.square())
    return stats


def normalize_advantages_per_task(
    advantages: torch.Tensor,
    task_ids: torch.Tensor,
    stats: torch.Tensor,
    *,
    min_count: int = 2,
    eps: float = 1e-5,
    fallback: AdvantageNormalizationFallback = "global",
) -> torch.Tensor:
    """Normalize each task and safely handle sparse or zero-variance tasks."""
    flat_advantages = advantages.reshape(-1)
    flat_task_ids = (
        task_ids.detach()
        .reshape(-1)
        .to(device=flat_advantages.device, dtype=torch.long)
    )
    if flat_task_ids.numel() != flat_advantages.numel():
        raise ValueError(
            "task_ids and advantages must have the same number of elements, got "
            f"{flat_task_ids.numel()} and {flat_advantages.numel()}"
        )

    stats = stats.to(device=flat_advantages.device, dtype=torch.float64)
    if stats.ndim != 2 or stats.shape[1] != 3:
        raise ValueError(f"stats must have shape [num_tasks, 3], got {stats.shape}")
    if flat_task_ids.numel() > 0 and torch.any(
        (flat_task_ids < 0) | (flat_task_ids >= stats.shape[0])
    ):
        raise ValueError(f"task_ids must be in [0, {stats.shape[0]})")

    counts = stats[:, 0]
    means = stats[:, 1] / counts.clamp_min(1.0)
    variances = stats[:, 2] / counts.clamp_min(1.0) - means.square()
    variances = variances.clamp_min(0.0)
    usable = (counts >= min_count) & torch.isfinite(variances) & (variances > eps)

    global_stats = stats.sum(dim=0)
    global_count = global_stats[0]
    global_mean = global_stats[1] / global_count.clamp_min(1.0)
    global_variance = (
        global_stats[2] / global_count.clamp_min(1.0) - global_mean.square()
    ).clamp_min(0.0)
    global_usable = bool(
        (global_count >= min_count).item() and (global_variance > eps).item()
    )

    if fallback == "global" and global_usable:
        fallback_means = global_mean.expand_as(means)
        fallback_variances = global_variance.expand_as(variances)
    elif fallback == "identity":
        fallback_means = torch.zeros_like(means)
        fallback_variances = torch.ones_like(variances)
    else:
        fallback_means = torch.zeros_like(means)
        fallback_variances = torch.ones_like(variances)

    effective_means = torch.where(usable, means, fallback_means)
    effective_variances = torch.where(usable, variances, fallback_variances)
    normalized = (
        flat_advantages.to(torch.float64) - effective_means[flat_task_ids]
    ) * torch.rsqrt(effective_variances[flat_task_ids] + eps)
    return normalized.reshape_as(advantages).to(advantages.dtype)


def task_advantage_normalization_metrics(
    stats: torch.Tensor,
    *,
    min_count: int,
    eps: float,
) -> dict[str, float]:
    """Summarize task coverage and fallback behavior for normalization."""
    stats = stats.detach().to(device="cpu", dtype=torch.float64)
    counts = stats[:, 0]
    means = stats[:, 1] / counts.clamp_min(1.0)
    variances = (stats[:, 2] / counts.clamp_min(1.0) - means.square()).clamp_min(0.0)
    covered = counts > 0
    usable = (
        covered & (counts >= min_count) & torch.isfinite(variances) & (variances > eps)
    )
    covered_stds = variances[covered].sqrt()
    std_mean = float(covered_stds.mean()) if covered_stds.numel() else 0.0
    std_cv = (
        float(covered_stds.std(unbiased=False)) / std_mean
        if covered_stds.numel() and std_mean > 0
        else 0.0
    )
    return {
        "advantage_normalization/covered_tasks": float(covered.sum()),
        "advantage_normalization/normalized_tasks": float(usable.sum()),
        "advantage_normalization/fallback_tasks": float((covered & ~usable).sum()),
        "advantage_normalization/pre_std_cv": std_cv,
        "advantage_normalization/pre_mean_abs_max": float(
            means[covered].abs().max() if covered.any() else 0.0
        ),
        "advantage_normalization/min_count": float(min_count),
        "advantage_normalization/eps": float(eps),
    }
