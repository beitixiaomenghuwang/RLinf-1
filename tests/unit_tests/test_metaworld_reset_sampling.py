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

"""Tests for balanced and deterministic MetaWorld reset-state sampling."""

import numpy as np
import pytest

from rlinf.envs.metaworld.sampling import (
    build_balanced_reset_state_matrix,
    required_reset_states_per_process,
    reset_ids_to_task_and_trial,
    resolve_egl_device_id,
    should_advance_fixed_eval_reset_states,
)


def make_reset_matrix(*, shuffle: bool) -> tuple[np.ndarray, np.ndarray]:
    trial_id_bins = [50] * 50
    cumsum_trial_id_bins = np.cumsum(trial_id_bins)
    matrix = build_balanced_reset_state_matrix(
        trial_id_bins,
        cumsum_trial_id_bins,
        8,
        generator=np.random.default_rng(seed=0),
        shuffle=shuffle,
        minimum_states_per_process=64,
    )
    return matrix, cumsum_trial_id_bins


def test_eval_reset_states_balance_tasks_across_eight_workers() -> None:
    matrix, cumsum_trial_id_bins = make_reset_matrix(shuffle=False)
    reset_ids = matrix[:, :64].reshape(-1)
    task_ids, _ = reset_ids_to_task_and_trial(reset_ids, cumsum_trial_id_bins)
    counts = np.bincount(task_ids, minlength=50)

    assert np.count_nonzero(counts) == 50
    assert counts.max() - counts.min() <= 1


def test_ordered_training_batches_advance_without_reusing_same_states() -> None:
    matrix, _ = make_reset_matrix(shuffle=True)
    first = matrix[3, :8]
    second = matrix[3, 8:16]

    assert len(np.unique(first)) == 8
    assert len(np.unique(second)) == 8
    assert not np.array_equal(first, second)


def test_reset_id_conversion_round_trips_task_and_trial() -> None:
    _, cumsum_trial_id_bins = make_reset_matrix(shuffle=False)
    reset_ids = np.array([0, 49, 50, 127, 2499])

    task_ids, trial_ids = reset_ids_to_task_and_trial(reset_ids, cumsum_trial_id_bins)

    np.testing.assert_array_equal(task_ids, np.array([0, 0, 1, 2, 49]))
    np.testing.assert_array_equal(trial_ids, np.array([0, 49, 0, 27, 49]))


def test_reset_matrix_tiles_when_worker_batch_exceeds_unique_partition() -> None:
    trial_id_bins = [50] * 50
    cumsum_trial_id_bins = np.cumsum(trial_id_bins)

    matrix = build_balanced_reset_state_matrix(
        trial_id_bins,
        cumsum_trial_id_bins,
        40,
        generator=np.random.default_rng(seed=0),
        shuffle=False,
        minimum_states_per_process=64,
    )
    task_ids, _ = reset_ids_to_task_and_trial(
        matrix[:, :64].reshape(-1), cumsum_trial_id_bins
    )
    counts = np.bincount(task_ids, minlength=50)

    assert matrix.shape == (40, 64)
    assert np.count_nonzero(counts) == 50
    assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize(
    ("total_num_envs", "rollout_epoch"),
    [(16, 32), (32, 16)],
)
def test_multiround_eval_covers_all_mt50_tasks(
    total_num_envs: int,
    rollout_epoch: int,
) -> None:
    total_num_processes = 8
    num_group = total_num_envs // total_num_processes
    trial_id_bins = [10] * 50
    cumsum_trial_id_bins = np.cumsum(trial_id_bins)
    states_per_process = required_reset_states_per_process(
        num_group,
        rollout_epoch,
        is_eval=True,
    )
    matrix = build_balanced_reset_state_matrix(
        trial_id_bins,
        cumsum_trial_id_bins,
        total_num_processes,
        generator=np.random.default_rng(seed=0),
        shuffle=False,
        minimum_states_per_process=states_per_process,
    )

    reset_batches = []
    for epoch in range(rollout_epoch):
        start = epoch * num_group
        end = start + num_group
        reset_batches.append(matrix[:, start:end].reshape(-1))
    reset_ids = np.concatenate(reset_batches)
    task_ids, _ = reset_ids_to_task_and_trial(reset_ids, cumsum_trial_id_bins)
    counts = np.bincount(task_ids, minlength=50)

    assert reset_ids.size == 512
    assert np.unique(reset_ids).size == 500
    assert np.count_nonzero(counts) == 50
    assert counts.min() == 10
    assert counts.max() == 11


def test_fixed_eval_auto_reset_advances_only_after_full_batch() -> None:
    assert should_advance_fixed_eval_reset_states(
        is_eval=True,
        use_fixed_reset_state_ids=True,
        dones=np.ones(16, dtype=bool),
    )
    assert not should_advance_fixed_eval_reset_states(
        is_eval=True,
        use_fixed_reset_state_ids=True,
        dones=np.array([True, False]),
    )
    assert not should_advance_fixed_eval_reset_states(
        is_eval=False,
        use_fixed_reset_state_ids=True,
        dones=np.ones(16, dtype=bool),
    )


def test_pipeline_stage_uses_worker_accelerator_for_egl() -> None:
    assert resolve_egl_device_id(6, accelerator_rank=3, device_count=4) == 3
    assert resolve_egl_device_id(7, accelerator_rank=3, device_count=4) == 3


def test_egl_device_fallback_wraps_logical_seed_offset() -> None:
    assert resolve_egl_device_id(6, accelerator_rank=None, device_count=4) == 2
    assert resolve_egl_device_id(6, accelerator_rank=None, device_count=0) == 0
