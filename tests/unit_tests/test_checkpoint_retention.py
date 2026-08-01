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

"""Tests for bounded embodied checkpoint retention."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from rlinf.runners.embodied_runner import (
    EmbodiedRunner,
    load_best_macro_mean,
    promote_checkpoint_as_best,
    prune_old_checkpoints,
)


class ConfigNamespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_prune_old_checkpoints_keeps_latest_steps_and_unrelated_paths(tmp_path) -> None:
    for step in (10, 2, 1):
        checkpoint = tmp_path / f"global_step_{step}"
        checkpoint.mkdir()
        (checkpoint / "marker").write_text(str(step))
    unrelated = tmp_path / "best_model"
    unrelated.mkdir()

    removed = prune_old_checkpoints(tmp_path, max_checkpoints_to_keep=2)

    assert removed == (tmp_path / "global_step_1",)
    assert not (tmp_path / "global_step_1").exists()
    assert (tmp_path / "global_step_2").is_dir()
    assert (tmp_path / "global_step_10").is_dir()
    assert unrelated.is_dir()


def test_prune_old_checkpoints_can_be_disabled(tmp_path) -> None:
    (tmp_path / "global_step_1").mkdir()

    assert prune_old_checkpoints(tmp_path, None) == ()
    assert (tmp_path / "global_step_1").is_dir()


def test_prune_old_checkpoints_rejects_zero_retention(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        prune_old_checkpoints(tmp_path, 0)


def test_checkpoint_is_saved_before_evaluation() -> None:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            val_check_interval=10,
            save_interval=10,
            resume_dir=None,
        )
    )
    runner.global_step = 10
    runner.max_steps = 20
    runner.timer = lambda _name: nullcontext()
    calls = []
    runner._save_checkpoint = lambda: calls.append("save")
    runner.update_rollout_weights = lambda: calls.append("sync")
    runner.evaluate = lambda: calls.append("evaluate") or {"success_once": 0.5}
    runner.metric_logger = SimpleNamespace(log=lambda **_kwargs: calls.append("log"))
    runner._maybe_save_best_macro_mean = lambda _metrics: calls.append("best")
    runner._mark_evaluation_complete = lambda _metrics: calls.append("mark")

    metrics = runner._maybe_eval_and_checkpoint(step=9)

    assert metrics == {"eval/success_once": 0.5}
    assert calls == ["save", "sync", "evaluate", "log", "best", "mark"]


def test_resume_evaluates_pending_checkpoint_before_training(tmp_path) -> None:
    resume_dir = tmp_path / "global_step_20"
    resume_dir.mkdir()
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(
        runner=ConfigNamespace(
            resume_dir=str(resume_dir),
            eval_on_resume=True,
            val_check_interval=20,
            save_interval=20,
        )
    )
    runner.global_step = 20
    runner.max_steps = 320

    assert runner._resume_needs_evaluation()

    (resume_dir / "evaluation_complete.json").write_text("{}\n")
    assert not runner._resume_needs_evaluation()


def test_resume_does_not_evaluate_between_intervals(tmp_path) -> None:
    resume_dir = tmp_path / "global_step_21"
    resume_dir.mkdir()
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(
        runner=ConfigNamespace(
            resume_dir=str(resume_dir),
            eval_on_resume=True,
            val_check_interval=20,
            save_interval=20,
        )
    )
    runner.global_step = 21
    runner.max_steps = 320

    assert not runner._resume_needs_evaluation()


def test_best_macro_checkpoint_is_independent_of_retention(tmp_path) -> None:
    source = tmp_path / "global_step_20"
    actor = source / "actor"
    actor.mkdir(parents=True)
    weights = actor / "weights.pt"
    weights.write_bytes(b"checkpoint")

    best = promote_checkpoint_as_best(tmp_path, global_step=20, macro_mean=0.625)
    metadata = json.loads((best / "best_metric.json").read_text())

    assert best == tmp_path / "best_macro_mean"
    assert (best / "actor" / "weights.pt").read_bytes() == b"checkpoint"
    assert (best / "actor" / "weights.pt").stat().st_ino == weights.stat().st_ino
    assert metadata == {
        "global_step": 20,
        "metric": "task_success/macro_mean",
        "value": 0.625,
    }
    assert load_best_macro_mean(tmp_path) == pytest.approx(0.625)

    (tmp_path / "global_step_30").mkdir()
    prune_old_checkpoints(tmp_path, max_checkpoints_to_keep=1)
    assert not source.exists()
    assert best.is_dir()


def test_best_macro_checkpoint_updates_only_on_improvement(tmp_path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    source = checkpoints_dir / "global_step_20" / "actor"
    source.mkdir(parents=True)
    (source / "weights.pt").write_bytes(b"checkpoint")
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            logger=SimpleNamespace(
                log_path=str(tmp_path.parent), experiment_name=tmp_path.name
            )
        )
    )
    runner.global_step = 20
    runner.save_best_macro_mean = True
    runner.best_eval_macro_mean = 0.6
    runner.logger = SimpleNamespace(
        info=lambda *_args: None, warning=lambda *_args: None
    )
    runner._save_checkpoint = lambda: pytest.fail("checkpoint already exists")

    runner._maybe_save_best_macro_mean({"eval/task_success/macro_mean": 0.5})
    assert not (checkpoints_dir / "best_macro_mean").exists()

    runner._maybe_save_best_macro_mean({"eval/task_success/macro_mean": 0.7})
    assert runner.best_eval_macro_mean == pytest.approx(0.7)
    assert load_best_macro_mean(checkpoints_dir) == pytest.approx(0.7)
