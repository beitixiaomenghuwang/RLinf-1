"""Tests for task-conditioned embodied evaluation metrics."""

import pytest
import torch

from rlinf.utils.metric_utils import compute_evaluate_metrics


def test_compute_evaluate_metrics_reports_per_task_and_worst_task_success() -> None:
    shards = [
        {
            "task_id": torch.tensor([0, 0, 1, 1]),
            "success_once": torch.tensor([1, 0, 1, 1]),
            "return": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        },
        {
            "task_id": torch.tensor([2, 2]),
            "success_once": torch.tensor([0, 0]),
            "return": torch.tensor([0.0, 0.0]),
        },
    ]

    metrics = compute_evaluate_metrics(shards)

    assert "task_id" not in metrics
    assert metrics["success_once"] == pytest.approx(0.5)
    assert metrics["task_success/task_00"] == pytest.approx(0.5)
    assert metrics["task_success/task_01"] == pytest.approx(1.0)
    assert metrics["task_success/task_02"] == pytest.approx(0.0)
    assert metrics["task_success/covered_tasks"] == 3
    assert metrics["task_success/macro_mean"] == pytest.approx(0.5)
    assert metrics["task_success/min"] == pytest.approx(0.0)
    assert metrics["task_success/num_above_90"] == 1
    assert metrics["task_success/worst_5_mean"] == pytest.approx(0.5)
    assert metrics["task_success/worst_10_mean"] == pytest.approx(0.5)


def test_compute_evaluate_metrics_rejects_unpaired_task_success() -> None:
    with pytest.raises(ValueError, match="same number of episodes"):
        compute_evaluate_metrics(
            [
                {
                    "task_id": torch.tensor([0, 1]),
                    "success_once": torch.tensor([1]),
                }
            ]
        )


def test_compute_evaluate_metrics_without_task_ids_is_unchanged() -> None:
    metrics = compute_evaluate_metrics(
        [{"success_once": torch.tensor([1, 0, 1])}]
    )

    assert metrics["success_once"] == pytest.approx(2 / 3)
    assert not any(key.startswith("task_success/") for key in metrics)
