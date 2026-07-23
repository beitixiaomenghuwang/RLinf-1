# Copyright 2025 The RLinf Authors.
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

import json
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed

if TYPE_CHECKING:
    from rlinf.data.embodied_io_struct import Trajectory


def mean_bool_tensor_rate(
    tensors: Sequence[torch.Tensor | None],
    *,
    sum_key: str,
    count_key: str,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> float | None:
    """Mean of flattened bool-like tensors, optionally reduced across ranks."""
    shards = [
        tensor.detach().reshape(-1).to(torch.float32)
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
    ]
    if not shards:
        return None

    local_values = torch.cat(shards, dim=0)
    reduced = {
        sum_key: float(local_values.sum().item()),
        count_key: float(local_values.numel()),
    }
    if reducer is not None:
        reduced = reducer(reduced)
    if reduced[count_key] <= 0:
        return 0.0
    return reduced[sum_key] / reduced[count_key]


def mean_bool_tensor_rate_from_trajectories(
    trajectories: Sequence["Trajectory"],
    tensor_getter: Callable[["Trajectory"], torch.Tensor | None],
    *,
    sum_key: str,
    count_key: str,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> float | None:
    return mean_bool_tensor_rate(
        [tensor_getter(trajectory) for trajectory in trajectories],
        sum_key=sum_key,
        count_key=count_key,
        reducer=reducer,
    )


def trajectory_forward_input_tensor(
    trajectory: "Trajectory", key: str
) -> torch.Tensor | None:
    forward_inputs = trajectory.forward_inputs
    if not isinstance(forward_inputs, dict):
        return None
    value = forward_inputs.get(key)
    return value if isinstance(value, torch.Tensor) else None


def trajectory_has_bool_tensor(tensor: torch.Tensor | None) -> bool:
    return bool(
        isinstance(tensor, torch.Tensor) and tensor.detach().to(torch.bool).any()
    )


def collect_trajectory_replay_metrics(
    trajectories: Sequence["Trajectory"],
    *,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Replay-route diagnostics aggregated from received trajectories."""
    metrics: dict[str, float] = {}
    rate_specs = (
        (
            "replay/record_transition_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "record_transition"
            ),
            "record_transition_sum",
            "record_transition_count",
        ),
        (
            "replay/actor_switch_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "actor_switch"
            ),
            "actor_switch_sum",
            "actor_switch_count",
        ),
        (
            "replay/intervention_requested_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "intervention_requested"
            ),
            "intervention_requested_sum",
            "intervention_requested_count",
        ),
        (
            "replay/intervention_rate",
            lambda trajectory: trajectory.intervene_flags,
            "intervention_sum",
            "intervention_count",
        ),
    )
    for metric_key, tensor_getter, sum_key, count_key in rate_specs:
        rate = mean_bool_tensor_rate_from_trajectories(
            trajectories,
            tensor_getter,
            sum_key=sum_key,
            count_key=count_key,
            reducer=reducer,
        )
        if rate is not None:
            metrics[metric_key] = rate
    return metrics


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def _normalize_metric_shard(shard: object) -> torch.Tensor:
    """One rank's metric -> 1D float tensor on CPU."""
    if shard is None:
        return torch.tensor([], dtype=torch.float32)
    if isinstance(shard, torch.Tensor):
        return shard.detach().cpu().reshape(-1).float()
    if isinstance(shard, list):
        if not shard:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat([x.detach().cpu().reshape(-1).float() for x in shard], dim=0)
    return torch.as_tensor(shard, dtype=torch.float32).cpu().reshape(-1)


def count_trajectories(metrics_dict):
    """
    Count the total number of trajectories from metrics dictionary.

    Args:
        metrics_dict: Dictionary of metrics where each value is a tensor after concatenation.
                     Each tensor's first dimension represents the number of trajectories.

    Returns:
        int: Total number of trajectories. If metrics_dict is empty, returns 0.
    """
    if not metrics_dict:
        return 0

    # Use the first metric tensor to get the trajectory count
    # All metrics should have the same first dimension (number of trajectories)
    first_key = next(iter(metrics_dict.keys()))
    first_tensor = metrics_dict[first_key]

    if isinstance(first_tensor, torch.Tensor):
        return first_tensor.shape[0]
    elif isinstance(first_tensor, list):
        # If it's a list of tensors, sum up all trajectory counts
        return sum(
            t.shape[0] if isinstance(t, torch.Tensor) else len(t) for t in first_tensor
        )
    else:
        raise TypeError(f"Unsupported tensor type: {type(first_tensor)}")


def compute_evaluate_metrics(eval_metrics_list):
    """
    List of evaluate metrics, list length stands for rollout process

    Returns:
        dict: Aggregated metrics with mean values and trajectory count
    """
    if not eval_metrics_list:
        return {}

    all_eval_metrics = {}
    env_info_keys: set[str] = set()
    for eval_metrics in eval_metrics_list:
        env_info_keys.update(eval_metrics.keys())

    # Count trajectories from each process
    trajectory_counts = []
    for eval_metrics in eval_metrics_list:
        count = count_trajectories(eval_metrics)
        trajectory_counts.append(count)

    for env_info_key in env_info_keys:
        metric = [
            eval_metrics[env_info_key]
            for eval_metrics in eval_metrics_list
            if env_info_key in eval_metrics
        ]
        if metric:
            all_eval_metrics[env_info_key] = metric

    task_success_metrics = _compute_task_success_metrics(eval_metrics_list)

    for key in all_eval_metrics:
        if key == "task_id":
            continue
        shards = [_normalize_metric_shard(s) for s in all_eval_metrics[key]]
        stacked = torch.concat(shards).float()
        all_eval_metrics[key] = (
            stacked.mean().detach().cpu().numpy()
            if stacked.numel() > 0
            else np.asarray(0.0, dtype=np.float64)
        )

    all_eval_metrics.pop("task_id", None)
    all_eval_metrics.update(task_success_metrics)

    # Add total trajectory count to metrics
    all_eval_metrics["num_trajectories"] = sum(trajectory_counts)

    return all_eval_metrics


def _compute_task_success_metrics(eval_metrics_list: Sequence[dict]) -> dict[str, float]:
    """Compute macro and worst-task success without losing task pairing."""
    task_shards: list[torch.Tensor] = []
    success_shards: list[torch.Tensor] = []
    for metrics in eval_metrics_list:
        if "task_id" not in metrics or "success_once" not in metrics:
            continue
        task_ids = _normalize_metric_shard(metrics["task_id"]).to(torch.long)
        successes = _normalize_metric_shard(metrics["success_once"])
        if task_ids.numel() != successes.numel():
            raise ValueError(
                "task_id and success_once must contain the same number of episodes, "
                f"got {task_ids.numel()} and {successes.numel()}"
            )
        task_shards.append(task_ids)
        success_shards.append(successes)

    if not task_shards:
        return {}

    task_ids = torch.cat(task_shards)
    successes = torch.cat(success_shards).float()
    per_task: dict[int, float] = {}
    for task_id in torch.unique(task_ids, sorted=True).tolist():
        task_successes = successes[task_ids == task_id]
        per_task[int(task_id)] = float(task_successes.mean().item())

    task_success_values = sorted(per_task.values())
    metrics = {
        f"task_success/task_{task_id:02d}": success
        for task_id, success in per_task.items()
    }
    metrics.update(
        {
            "task_success/covered_tasks": float(len(per_task)),
            "task_success/macro_mean": float(np.mean(task_success_values)),
            "task_success/min": task_success_values[0],
            "task_success/num_above_90": float(
                sum(success >= 0.9 for success in task_success_values)
            ),
            "task_success/worst_5_mean": float(
                np.mean(task_success_values[: min(5, len(task_success_values))])
            ),
            "task_success/worst_10_mean": float(
                np.mean(task_success_values[: min(10, len(task_success_values))])
            ),
        }
    )
    return metrics


def compute_rollout_metrics(data_buffer: dict) -> dict:
    rollout_metrics = {}
    loss_mask = data_buffer.get("loss_mask", None)

    def reduce_metrics(values: torch.Tensor) -> tuple[float, float, float]:
        from rlinf.scheduler.worker.worker import Worker

        device = Worker.torch_platform.current_device()

        if values.numel() == 0:
            count = torch.tensor(0.0, device=device, dtype=torch.float32)
            values_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            min_value = float("inf")
            max_value = float("-inf")
        else:
            values = values.to(device)
            count = torch.tensor(
                values.numel(), device=values.device, dtype=torch.float32
            )
            values_sum = values.to(dtype=torch.float32).sum()
            max_value = torch.max(values).detach().item()
            min_value = torch.min(values).detach().item()

        reduce_sum_count = torch.stack([values_sum, count])
        reduce_min_max = torch.as_tensor(
            [-min_value, max_value],
            device=device,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_sum_count, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(reduce_min_max, op=torch.distributed.ReduceOp.MAX)
        reduced_sum, reduced_count = reduce_sum_count.tolist()
        reduced_min, reduced_max = reduce_min_max.tolist()

        if reduced_count <= 0:
            return float("nan"), float("nan"), float("nan")
        return reduced_sum / reduced_count, -reduced_min, reduced_max

    def valid_values(values: torch.Tensor) -> torch.Tensor:
        if loss_mask is None:
            return values.reshape(-1)
        mask = loss_mask.to(device=values.device, dtype=torch.bool)
        if mask.shape != values.shape:
            mask = torch.broadcast_to(mask, values.shape)
        return values[mask]

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"]
        rewards = valid_values(rewards)
        mean_rewards, _, _ = reduce_metrics(rewards)

        rewards_metrics = {
            "rewards": mean_rewards,
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        advantages = valid_values(advantages)
        mean_adv, min_adv, max_adv = reduce_metrics(advantages)

        advantages_metrics = {
            "advantages_mean": mean_adv,
            "advantages_max": max_adv,
            "advantages_min": min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if data_buffer.get("returns", None) is not None:
        returns = data_buffer["returns"]
        returns = valid_values(returns)
        mean_ret, min_ret, max_ret = reduce_metrics(returns)

        returns_metrics = {
            "returns_mean": mean_ret,
            "returns_max": max_ret,
            "returns_min": min_ret,
        }
        rollout_metrics.update(returns_metrics)

    return rollout_metrics


def _accumulate_task_advantage_stats(
    task_ids: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None,
    num_tasks: int,
) -> torch.Tensor:
    """Scatter per-sample advantages into per-task sufficient statistics.

    Returns a local (unreduced) ``[num_tasks, 3]`` tensor of
    ``(count, advantage_sum, advantage_sumsq)`` per task, safe to sum across
    micro-batches and distributed ranks before computing per-task mean/std.
    """
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    task_ids = task_ids.detach().reshape(-1).to(torch.long)
    advantages = advantages.detach().reshape(-1).to(torch.float32)
    if task_ids.numel() != advantages.numel():
        raise ValueError(
            "task_ids and advantages must have the same number of elements, got "
            f"{task_ids.numel()} and {advantages.numel()}"
        )
    device = advantages.device
    task_ids = task_ids.to(device)
    if task_ids.numel() > 0 and torch.any((task_ids < 0) | (task_ids >= num_tasks)):
        raise ValueError(f"task_ids must be in [0, {num_tasks})")

    if loss_mask is not None:
        mask = loss_mask.detach().reshape(-1).to(device=device, dtype=torch.bool)
        if mask.numel() != advantages.numel():
            raise ValueError(
                "loss_mask must have the same number of elements as advantages, "
                f"got {mask.numel()} and {advantages.numel()}"
            )
        task_ids = task_ids[mask]
        advantages = advantages[mask]

    stats = torch.zeros(num_tasks, 3, device=device, dtype=torch.float32)
    if task_ids.numel() == 0:
        return stats
    stats[:, 0].index_add_(0, task_ids, torch.ones_like(advantages))
    stats[:, 1].index_add_(0, task_ids, advantages)
    stats[:, 2].index_add_(0, task_ids, advantages * advantages)
    return stats


def _finalize_task_advantage_metrics(
    stats: torch.Tensor, num_tasks: int
) -> dict[str, float]:
    """Convert globally summed per-task advantage statistics into metrics."""
    counts = stats[:, 0]
    sums = stats[:, 1]
    sumsqs = stats[:, 2]

    metrics: dict[str, float] = {}
    covered_counts = []
    covered_means = []
    covered_stds = []
    for task_index in range(num_tasks):
        count = float(counts[task_index].item())
        if count <= 0:
            continue
        mean = float(sums[task_index].item()) / count
        variance = max(float(sumsqs[task_index].item()) / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        metrics[f"task_advantage/task_{task_index:02d}/count"] = count
        metrics[f"task_advantage/task_{task_index:02d}/mean"] = mean
        metrics[f"task_advantage/task_{task_index:02d}/std"] = std
        covered_counts.append(count)
        covered_means.append(mean)
        covered_stds.append(std)

    if not covered_counts:
        return {}

    counts_tensor = torch.tensor(covered_counts, dtype=torch.float64)
    means_tensor = torch.tensor(covered_means, dtype=torch.float64)
    stds_tensor = torch.tensor(covered_stds, dtype=torch.float64)

    counts_mean = float(counts_tensor.mean())
    stds_mean = float(stds_tensor.mean())
    metrics.update(
        {
            "task_advantage/covered_tasks": float(len(covered_counts)),
            "task_advantage/count_min": float(counts_tensor.min()),
            "task_advantage/count_max": float(counts_tensor.max()),
            "task_advantage/count_mean": counts_mean,
            "task_advantage/mean_min": float(means_tensor.min()),
            "task_advantage/mean_max": float(means_tensor.max()),
            "task_advantage/std_min": float(stds_tensor.min()),
            "task_advantage/std_max": float(stds_tensor.max()),
            "task_advantage/std_mean": stds_mean,
            # Coefficient of variation across tasks: large count_cv indicates
            # uneven per-task sample counts (evidence for tail-task loss
            # weighting); large std_cv with small count_cv indicates uneven
            # per-task advantage scale (evidence for per-task advantage
            # normalization).
            "task_advantage/count_cv": float(counts_tensor.std(unbiased=False))
            / counts_mean
            if counts_mean > 0
            else 0.0,
            "task_advantage/std_cv": float(stds_tensor.std(unbiased=False)) / stds_mean
            if stds_mean > 0
            else 0.0,
        }
    )
    return metrics


def compute_task_advantage_metrics(data_buffer: dict, num_tasks: int) -> dict:
    """Log per-task PPO sample counts and advantage mean/std.

    Diagnostic for deciding between tail-task loss weighting and per-task
    advantage normalization on MetaWorld MT50: large ``task_advantage/count_cv``
    means per-task sample counts are uneven, large ``task_advantage/std_cv``
    means per-task advantage scale is uneven.
    """
    from rlinf.scheduler.worker.worker import Worker

    forward_inputs = data_buffer.get("forward_inputs", {}) or {}
    task_ids = forward_inputs.get("task_ids", None)
    advantages = data_buffer.get("advantages", None)
    if task_ids is None or advantages is None:
        return {}

    stats = _accumulate_task_advantage_stats(
        task_ids, advantages, data_buffer.get("loss_mask", None), num_tasks
    )
    # The rollout batch may still live on CPU at this point; all_reduce requires
    # a device the process group's backend (e.g. NCCL) supports.
    stats = stats.to(Worker.torch_platform.current_device())
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    return _finalize_task_advantage_metrics(stats, num_tasks)


def append_to_dict(data, new_data):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def scalar_metrics_to_python(metrics: dict) -> dict:
    """Detach scalar tensor metrics so NumPy aggregation is device-safe."""
    converted = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Metric {key!r} must be scalar, got shape {tuple(value.shape)}"
                )
            value = value.detach().item()
        converted[key] = value
    return converted


def scalar_metrics_for_json(metrics: dict) -> dict[str, float]:
    """Convert scalar numeric metrics while ignoring structured values."""
    converted: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                continue
            value = value.item()
        if isinstance(value, (int, float, np.integer, np.floating)):
            converted[key] = float(value)
    return converted


def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(
        -1, actual_bsz
    )  # [(n_chunk_step + 1) * num_action_chunks, rollout_epoch x bsz]
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual-bsz]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[
        :-1
    ]  # [n_steps, actual-bsz]

    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def print_metrics_table(
    step: int,
    total_steps: int,
    start_time: float,
    metrics: dict,
    start_step: int = 0,
    log_path: str | None = None,
):
    """Print training metrics in a simple, fast formatted table.

    The rendered table is written to stdout and, when ``log_path`` is given,
    also appended to ``<log_path>/metrics.log``.
    """
    # Accumulate the table into lines so the exact same rendering goes to both
    # stdout and the log file.
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    # Calculate progress info
    progress = (step + 1) / total_steps * 100
    elapsed_time = time.time() - start_time
    steps_done = step + 1 - start_step
    eta_seconds = (
        elapsed_time / steps_done * (total_steps - step - 1) if steps_done > 0 else 0
    )

    def format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    # Format elapsed time and ETA
    elapsed_str = format_time(elapsed_time)
    eta_str = format_time(eta_seconds)

    # Create progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Print header with progress
    total_width = 120

    def _fit_line(text: str, width: int) -> str:
        if len(text) <= width:
            return text + (" " * (width - len(text)))
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def _fit_cell(text: str, width: int) -> str:
        return _fit_line(text, width)

    def _print_section_title(title: str) -> None:
        title_text = f" {title} "
        padding = total_width - 2 - len(title_text)
        left = padding // 2
        right = padding - left
        emit(f"├{'─' * left}{title_text}{'─' * right}┤")

    emit(f"\n╭{'─' * (total_width - 2)}╮")
    _print_section_title("Metric Table")

    # First line: Global Step and Progress
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line1 = f"│ {step_str} │ {progress_str}"
    line1 = _fit_line(line1, total_width - 2)
    emit(f"{line1} │")

    # Second line: Time information
    elapsed_str_formatted = f"Elapsed: {elapsed_str}"
    eta_str_formatted = f"ETA: {eta_str}"
    step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"
    line2 = f"│ {elapsed_str_formatted} │ {eta_str_formatted} │ {step_time_str}"
    line2 = _fit_line(line2, total_width - 2)
    emit(f"{line2} │")

    # Group metrics by category
    categories = {
        "Time": {},
        "Environment": {},
        "Rollout": {},
        "Evaluation": {},
        "Replay Buffer": {},
        "Training/Actor": {},
        "Training/Critic": {},
        "Training/Other": {},
    }

    for key, value in metrics.items():
        if "/" in key:
            category, metric_name = key.split("/", 1)
            category_map = {
                "time": "Time",
                "env": "Environment",
                "rollout": "Rollout",
                "eval": "Evaluation",
                "replay_buffer": "Replay Buffer",
            }
            if category in category_map:
                if category == "rollout" and metric_name.startswith(
                    "task_advantage/task_"
                ):
                    continue
                categories[category_map[category]][metric_name] = value
            elif category == "train":
                if metric_name.startswith("gse/task_router/task_"):
                    continue
                if metric_name.startswith("actor/"):
                    categories["Training/Actor"][metric_name] = value
                elif metric_name.startswith("critic/"):
                    categories["Training/Critic"][metric_name] = value
                elif metric_name.startswith("replay_buffer/"):
                    categories["Replay Buffer"][
                        metric_name.replace("replay_buffer/", "")
                    ] = value
                else:
                    categories["Training/Other"][metric_name] = value

    # Print metrics by category - 3 metrics per row
    table_width = total_width  # Match header width
    base_col_width = (table_width - 4) // 3
    remainder = (table_width - 4) - (base_col_width * 3)
    col_widths = [
        base_col_width + (1 if remainder > 0 else 0),
        base_col_width + (1 if remainder > 1 else 0),
        base_col_width,
    ]

    for category_name, category_metrics in categories.items():
        if category_metrics:
            _print_section_title(category_name)
            # Blank line before metrics (except Global Step section, which is separate)
            emit(f"│{' ' * (table_width - 2)}│")

            # Sort metrics for consistent output
            sorted_metrics = sorted(category_metrics.items())

            # Print in 3-column layout
            for i in range(0, len(sorted_metrics), 3):
                # Get up to 3 metrics for this row
                row_metrics = []
                for j in range(3):
                    if i + j < len(sorted_metrics):
                        metric_name, metric_value = sorted_metrics[i + j]

                        # Format value
                        if isinstance(metric_value, float):
                            if abs(metric_value) < 0.001 and metric_value != 0:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) < 0.01:
                                formatted_value = f"{metric_value:.4f}"
                            elif abs(metric_value) > 10000:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) > 100:
                                formatted_value = f"{metric_value:.1f}"
                            else:
                                formatted_value = f"{metric_value:.3f}"
                        else:
                            formatted_value = str(metric_value)

                        display = f"{metric_name}={formatted_value}"
                        row_metrics.append(display)
                    else:
                        row_metrics.append("")

                # Create the line with exactly 3 columns
                line = (
                    f"│{_fit_cell(row_metrics[0], col_widths[0])}"
                    f"│{_fit_cell(row_metrics[1], col_widths[1])}"
                    f"│{_fit_cell(row_metrics[2], col_widths[2])}│"
                )
                emit(line)

            # Section separator (minimal)
            emit(f"│{' ' * (table_width - 2)}│")

    # Bottom border
    emit(f"╰{'─' * (table_width - 2)}╯")

    emit()

    table = "\n".join(lines)
    print(table)
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "metrics.log"), "a") as metrics_file:
            metrics_file.write(table + "\n")
        json_metrics = scalar_metrics_for_json(metrics)
        with open(os.path.join(log_path, "metrics.jsonl"), "a") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {"step": int(step), "metrics": json_metrics},
                    sort_keys=True,
                )
                + "\n"
            )
