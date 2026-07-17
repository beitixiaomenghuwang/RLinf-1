"""Tests for task-conditioned embodied evaluation metrics."""

import json

import numpy as np
import pytest
import torch

from rlinf.utils.metric_utils import (
    compute_evaluate_metrics,
    print_metrics_table,
    scalar_metrics_for_json,
    scalar_metrics_to_python,
)


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


def test_scalar_metrics_to_python_detaches_tensor_scalars() -> None:
    metric = torch.tensor(1.25, requires_grad=True)

    converted = scalar_metrics_to_python({"tensor": metric, "float": 2.5})

    assert converted == {"tensor": 1.25, "float": 2.5}
    assert isinstance(converted["tensor"], float)


def test_scalar_metrics_to_python_rejects_vector_metrics() -> None:
    with pytest.raises(ValueError, match="must be scalar"):
        scalar_metrics_to_python({"vector": torch.ones(2)})


def test_scalar_metrics_for_json_keeps_numpy_and_tensor_scalars() -> None:
    metrics = scalar_metrics_for_json(
        {
            "python": 1.5,
            "numpy": np.asarray(2.5),
            "tensor": torch.tensor(3.5),
            "vector": torch.ones(2),
        }
    )

    assert metrics == {"python": 1.5, "numpy": 2.5, "tensor": 3.5}


def test_print_metrics_table_writes_exact_jsonl(tmp_path) -> None:
    print_metrics_table(
        step=2,
        total_steps=4,
        start_time=0.0,
        metrics={
            "eval/success_once": np.asarray(0.625),
            "train/gse/task_router/nmi": 0.0123456789,
        },
        log_path=str(tmp_path),
    )

    record = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert record["step"] == 2
    assert record["metrics"]["eval/success_once"] == 0.625
    assert record["metrics"]["train/gse/task_router/nmi"] == 0.0123456789
