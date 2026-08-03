# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-free reset-state scheduling for MetaWorld benchmarks."""

import numpy as np


def required_reset_states_per_process(
    num_group: int,
    rollout_epoch: int,
    *,
    is_eval: bool,
) -> int:
    """Return the reset-state capacity required by one environment process.

    Multi-round evaluation must reserve one distinct batch for every rollout
    epoch. Training only needs the next batch because its shuffled pool can be
    regenerated after exhaustion.
    """
    if num_group <= 0:
        raise ValueError(f"num_group must be positive, got {num_group}")
    if rollout_epoch <= 0:
        raise ValueError(f"rollout_epoch must be positive, got {rollout_epoch}")
    return num_group * rollout_epoch if is_eval else num_group


def should_advance_fixed_eval_reset_states(
    *,
    is_eval: bool,
    use_fixed_reset_state_ids: bool,
    dones: np.ndarray,
) -> bool:
    """Return whether a completed eval batch should advance its reset IDs."""
    dones = np.asarray(dones, dtype=bool)
    return (
        is_eval and use_fixed_reset_state_ids and dones.size > 0 and bool(dones.all())
    )


def resolve_egl_device_id(
    seed_offset: int,
    accelerator_rank: int | None,
    device_count: int,
) -> int:
    """Map a logical environment process to a visible EGL device.

    Pipeline stages increase ``seed_offset`` beyond the number of physical
    GPUs. Rendering must remain on the EnvWorker's assigned accelerator while
    the full seed offset continues to partition reset states.

    Args:
        seed_offset: Logical environment-process offset used for sampling.
        accelerator_rank: Physical accelerator rank assigned to the EnvWorker.
        device_count: Number of CUDA devices visible in the worker process.

    Returns:
        A valid zero-based EGL device index.
    """
    if device_count <= 0:
        return 0
    candidate = seed_offset if accelerator_rank is None else accelerator_rank
    return int(candidate) % device_count


def build_balanced_reset_state_matrix(
    trial_id_bins: list[int],
    cumsum_trial_id_bins: np.ndarray,
    total_num_processes: int,
    *,
    generator: np.random.Generator,
    shuffle: bool,
    minimum_states_per_process: int = 0,
) -> np.ndarray:
    """Interleave tasks before distributing reset states across workers."""
    num_tasks = len(trial_id_bins)
    trial_order = np.arange(max(trial_id_bins), dtype=np.int64)
    if shuffle:
        generator.shuffle(trial_order)

    reset_state_ids = []
    for trial_id in trial_order:
        task_order = np.arange(num_tasks, dtype=np.int64)
        if shuffle:
            generator.shuffle(task_order)
        for task_id in task_order:
            if trial_id >= trial_id_bins[task_id]:
                continue
            task_start = cumsum_trial_id_bins[task_id - 1] if task_id > 0 else 0
            reset_state_ids.append(task_start + trial_id)

    reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64)
    divisible_size = len(reset_state_ids) - (len(reset_state_ids) % total_num_processes)
    required_size = minimum_states_per_process * total_num_processes
    valid_size = max(divisible_size, required_size)
    if valid_size > len(reset_state_ids):
        repeats = (valid_size + len(reset_state_ids) - 1) // len(reset_state_ids)
        reset_state_ids = np.tile(reset_state_ids, repeats)
    reset_state_ids = reset_state_ids[:valid_size]
    return reset_state_ids.reshape(-1, total_num_processes).T.copy()


def reset_ids_to_task_and_trial(
    reset_state_ids: np.ndarray,
    cumsum_trial_id_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flattened benchmark reset IDs to task and trial IDs."""
    reset_state_ids = np.asarray(reset_state_ids, dtype=np.int64)
    task_ids = np.searchsorted(cumsum_trial_id_bins, reset_state_ids, side="right")
    task_starts = np.where(
        task_ids > 0,
        cumsum_trial_id_bins[np.maximum(task_ids - 1, 0)],
        0,
    )
    return task_ids, reset_state_ids - task_starts
