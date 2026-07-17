"""Tests for multi-seed embodied evaluation summaries."""

import json

import pytest

from toolkits.embodiment.summarize_multiseed_eval import (
    aggregate_runs,
    compare_tasks,
    load_last_evaluation,
)


def test_load_last_evaluation_skips_training_records(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    records = [
        {"step": 0, "metrics": {"train/loss": 1.0}},
        {"step": 1, "metrics": {"eval/success_once": 0.5}},
        {"step": 2, "metrics": {"eval/success_once": 0.75}},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    assert load_last_evaluation(path)["eval/success_once"] == 0.75


def test_aggregate_and_compare_task_means() -> None:
    candidate = aggregate_runs(
        [
            {
                "eval/success_once": 0.6,
                "eval/task_success/task_00": 0.8,
                "eval/task_success/task_01": 0.2,
            },
            {
                "eval/success_once": 0.8,
                "eval/task_success/task_00": 1.0,
                "eval/task_success/task_01": 0.4,
            },
        ]
    )
    baseline = aggregate_runs(
        [
            {
                "eval/success_once": 0.5,
                "eval/task_success/task_00": 0.7,
                "eval/task_success/task_01": 0.5,
            }
        ]
    )

    assert candidate["summary"]["eval/success_once"]["mean"] == pytest.approx(0.7)
    comparison = compare_tasks(candidate, baseline)
    assert comparison["improved_tasks"] == 1
    assert comparison["regressed_tasks"] == 1
    assert comparison["shared_tasks"] == 2
