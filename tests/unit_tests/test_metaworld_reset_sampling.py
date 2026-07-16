"""Tests for balanced and deterministic MetaWorld reset-state sampling."""

import numpy as np

from rlinf.envs.metaworld.sampling import (
    build_balanced_reset_state_matrix,
    reset_ids_to_task_and_trial,
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

    task_ids, trial_ids = reset_ids_to_task_and_trial(
        reset_ids, cumsum_trial_id_bins
    )

    np.testing.assert_array_equal(task_ids, np.array([0, 0, 1, 2, 49]))
    np.testing.assert_array_equal(trial_ids, np.array([0, 49, 0, 27, 49]))
