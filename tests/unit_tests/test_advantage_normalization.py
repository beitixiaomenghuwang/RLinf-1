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

"""Tests for configurable multi-task advantage normalization."""

import pytest
import torch

from rlinf.algorithms.advantage_normalization import (
    normalize_advantages_per_task,
    resolve_advantage_normalization_config,
    task_advantage_normalization_metrics,
    task_advantage_stats,
)


def test_resolve_preserves_legacy_global_and_disabled_modes() -> None:
    assert (
        resolve_advantage_normalization_config({"normalize_advantages": True}).mode
        == "global"
    )
    assert (
        resolve_advantage_normalization_config({"normalize_advantages": False}).mode
        == "none"
    )


def test_resolve_per_task_requires_task_count_and_enabled_normalization() -> None:
    with pytest.raises(ValueError, match="num_tasks is required"):
        resolve_advantage_normalization_config(
            {
                "normalize_advantages": True,
                "advantage_normalization_mode": "per_task",
            }
        )
    with pytest.raises(ValueError, match="incompatible"):
        resolve_advantage_normalization_config(
            {
                "normalize_advantages": False,
                "advantage_normalization_mode": "per_task",
                "advantage_normalization_num_tasks": 2,
            }
        )


def test_per_task_normalization_centers_and_scales_each_task() -> None:
    advantages = torch.tensor([[1.0, 3.0], [10.0, 14.0]])
    task_ids = torch.tensor([[0, 0], [1, 1]])
    stats = task_advantage_stats(advantages, task_ids, num_tasks=2)

    normalized = normalize_advantages_per_task(
        advantages, task_ids, stats, min_count=2, eps=1e-8
    )

    for task_index in range(2):
        task_values = normalized[task_ids == task_index]
        assert task_values.mean() == pytest.approx(0.0, abs=1e-6)
        assert task_values.std(unbiased=False) == pytest.approx(1.0, abs=1e-6)


def test_sparse_task_falls_back_to_global_statistics() -> None:
    advantages = torch.tensor([1.0, 3.0, 10.0])
    task_ids = torch.tensor([0, 0, 1])
    stats = task_advantage_stats(advantages, task_ids, num_tasks=2)

    normalized = normalize_advantages_per_task(
        advantages,
        task_ids,
        stats,
        min_count=2,
        eps=1e-8,
        fallback="global",
    )
    metrics = task_advantage_normalization_metrics(stats, min_count=2, eps=1e-8)

    assert normalized[-1].abs() > 0
    assert metrics["advantage_normalization/normalized_tasks"] == 1
    assert metrics["advantage_normalization/fallback_tasks"] == 1


def test_task_stats_exclude_masked_values() -> None:
    advantages = torch.tensor([1.0, 100.0, 3.0])
    task_ids = torch.tensor([0, 0, 0])
    mask = torch.tensor([True, False, True])

    stats = task_advantage_stats(advantages, task_ids, num_tasks=1, loss_mask=mask)

    torch.testing.assert_close(
        stats[0], torch.tensor([2.0, 4.0, 10.0], dtype=torch.float64)
    )
