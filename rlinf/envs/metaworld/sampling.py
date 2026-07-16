"""Dependency-free reset-state scheduling for MetaWorld benchmarks."""

import numpy as np


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
    divisible_size = len(reset_state_ids) - (
        len(reset_state_ids) % total_num_processes
    )
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
