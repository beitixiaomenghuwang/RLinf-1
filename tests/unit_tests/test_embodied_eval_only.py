"""Tests for the embodied runner's evaluation-only control flow."""

from contextlib import contextmanager
from types import SimpleNamespace

from rlinf.runners.embodied_runner import EmbodiedRunner


class FakeWorkerGroup:
    def __init__(self) -> None:
        self.steps = []

    def set_global_step(self, step: int) -> None:
        self.steps.append(step)


class FakeTimer:
    @contextmanager
    def __call__(self, _name):
        yield

    def consume_durations(self):
        return {"eval": 1.5}


class FakeMetricLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, data, step) -> None:
        self.records.append((data, step))


def test_run_evaluation_syncs_weights_and_skips_training() -> None:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(runner={"only_eval": True})
    runner.global_step = 4
    runner.actor = FakeWorkerGroup()
    runner.rollout = FakeWorkerGroup()
    runner.timer = FakeTimer()
    runner.metric_logger = FakeMetricLogger()
    calls = []
    runner.update_rollout_weights = lambda: calls.append("sync")
    runner.evaluate = lambda: {"success_once": 0.5}
    runner.print_metrics_table_async = lambda *args: calls.append("print")
    runner._finish_run = lambda: calls.append("finish")

    metrics = runner.run_evaluation()

    assert metrics == {"eval/success_once": 0.5}
    assert runner.actor.steps == [4]
    assert runner.rollout.steps == [4]
    assert calls == ["sync", "print", "finish"]
    assert runner.metric_logger.records == [
        ({"eval/success_once": 0.5}, 4),
        ({"time/eval": 1.5}, 4),
    ]
