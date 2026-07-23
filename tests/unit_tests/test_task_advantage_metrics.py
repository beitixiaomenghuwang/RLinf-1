"""Tests for per-task PPO sample-count and advantage-scale diagnostics."""

import math

import pytest
import torch

from rlinf.utils.metric_utils import (
    _accumulate_task_advantage_stats,
    _finalize_task_advantage_metrics,
)


def test_accumulate_and_finalize_report_per_task_count_mean_std() -> None:
    # Task 0: two samples with advantages 1.0 and 3.0 -> mean 2.0, std 1.0.
    # Task 1: one sample with advantage 0.0 -> mean 0.0, std 0.0.
    task_ids = torch.tensor([0, 0, 1])
    advantages = torch.tensor([1.0, 3.0, 0.0])

    stats = _accumulate_task_advantage_stats(
        task_ids, advantages, loss_mask=None, num_tasks=3
    )
    metrics = _finalize_task_advantage_metrics(stats, num_tasks=3)

    assert metrics["task_advantage/task_00/count"] == pytest.approx(2.0)
    assert metrics["task_advantage/task_00/mean"] == pytest.approx(2.0)
    assert metrics["task_advantage/task_00/std"] == pytest.approx(1.0)
    assert metrics["task_advantage/task_01/count"] == pytest.approx(1.0)
    assert metrics["task_advantage/task_01/mean"] == pytest.approx(0.0)
    assert metrics["task_advantage/task_01/std"] == pytest.approx(0.0)
    assert "task_advantage/task_02/count" not in metrics
    assert metrics["task_advantage/covered_tasks"] == pytest.approx(2.0)


def test_accumulate_excludes_masked_out_samples() -> None:
    task_ids = torch.tensor([0, 0, 0])
    advantages = torch.tensor([1.0, 3.0, 100.0])
    loss_mask = torch.tensor([1, 1, 0], dtype=torch.bool)

    stats = _accumulate_task_advantage_stats(
        task_ids, advantages, loss_mask, num_tasks=1
    )
    metrics = _finalize_task_advantage_metrics(stats, num_tasks=1)

    assert metrics["task_advantage/task_00/count"] == pytest.approx(2.0)
    assert metrics["task_advantage/task_00/mean"] == pytest.approx(2.0)


def test_accumulate_rejects_out_of_range_task_ids() -> None:
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        _accumulate_task_advantage_stats(
            torch.tensor([0, 2]),
            torch.tensor([1.0, 1.0]),
            loss_mask=None,
            num_tasks=2,
        )


def test_accumulate_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same number of elements"):
        _accumulate_task_advantage_stats(
            torch.tensor([0, 1]),
            torch.tensor([1.0]),
            loss_mask=None,
            num_tasks=2,
        )


def test_finalize_count_cv_is_zero_when_balanced_and_positive_when_skewed() -> None:
    # Balanced: every task has the same count and the same advantage scale.
    balanced_task_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    balanced_advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    balanced_stats = _accumulate_task_advantage_stats(
        balanced_task_ids, balanced_advantages, loss_mask=None, num_tasks=3
    )
    balanced_metrics = _finalize_task_advantage_metrics(balanced_stats, num_tasks=3)

    assert balanced_metrics["task_advantage/count_cv"] == pytest.approx(0.0, abs=1e-9)
    assert balanced_metrics["task_advantage/std_cv"] == pytest.approx(0.0, abs=1e-9)

    # Skewed: task 0 has far more samples than tasks 1 and 2 (a tail task).
    skewed_task_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2])
    skewed_advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    skewed_stats = _accumulate_task_advantage_stats(
        skewed_task_ids, skewed_advantages, loss_mask=None, num_tasks=3
    )
    skewed_metrics = _finalize_task_advantage_metrics(skewed_stats, num_tasks=3)

    assert skewed_metrics["task_advantage/count_cv"] > 0.5


def test_finalize_std_cv_is_positive_when_advantage_scale_differs() -> None:
    # Same count per task, but task 1's advantages are much larger in scale.
    task_ids = torch.tensor([0, 0, 1, 1])
    advantages = torch.tensor([1.0, -1.0, 10.0, -10.0])

    stats = _accumulate_task_advantage_stats(
        task_ids, advantages, loss_mask=None, num_tasks=2
    )
    metrics = _finalize_task_advantage_metrics(stats, num_tasks=2)

    assert metrics["task_advantage/count_cv"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["task_advantage/std_cv"] > 0.5
    assert metrics["task_advantage/task_00/std"] == pytest.approx(1.0)
    assert metrics["task_advantage/task_01/std"] == pytest.approx(10.0)


def test_finalize_returns_empty_when_no_task_has_samples() -> None:
    stats = torch.zeros(3, 3)
    assert _finalize_task_advantage_metrics(stats, num_tasks=3) == {}


def test_accumulate_handles_empty_input() -> None:
    stats = _accumulate_task_advantage_stats(
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        loss_mask=None,
        num_tasks=2,
    )
    assert stats.shape == (2, 3)
    assert torch.all(stats == 0)
    assert not math.isnan(float(stats.sum()))
