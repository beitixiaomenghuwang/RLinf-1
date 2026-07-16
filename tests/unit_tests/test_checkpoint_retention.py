"""Tests for bounded embodied checkpoint retention."""

import pytest

from rlinf.runners.embodied_runner import prune_old_checkpoints


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
