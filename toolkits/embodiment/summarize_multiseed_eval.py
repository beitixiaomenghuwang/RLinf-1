"""Summarize matched multi-seed embodied evaluations from metrics JSONL files."""

import argparse
import json
import math
import statistics
from pathlib import Path

SUMMARY_KEYS = (
    "eval/success_once",
    "eval/success_at_end",
    "eval/task_success/macro_mean",
    "eval/task_success/worst_5_mean",
    "eval/task_success/worst_10_mean",
    "eval/task_success/num_above_90",
)
TASK_PREFIX = "eval/task_success/task_"


def load_last_evaluation(path: Path) -> dict[str, float]:
    """Load the last JSONL record containing evaluation success."""
    selected: dict[str, float] | None = None
    with path.open(encoding="utf-8") as metrics_file:
        for line in metrics_file:
            record = json.loads(line)
            metrics = record.get("metrics", {})
            if "eval/success_once" in metrics:
                selected = {key: float(value) for key, value in metrics.items()}
    if selected is None:
        raise ValueError(f"No evaluation record found in {path}")
    return selected


def summarize(values: list[float]) -> dict[str, float]:
    """Compute sample statistics and a normal-approximation confidence interval."""
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    confidence_radius = (
        1.96 * standard_deviation / math.sqrt(len(values)) if values else 0.0
    )
    return {
        "mean": mean,
        "std": standard_deviation,
        "ci95_low": mean - confidence_radius,
        "ci95_high": mean + confidence_radius,
    }


def aggregate_runs(runs: list[dict[str, float]]) -> dict[str, object]:
    """Aggregate summary and per-task metrics across runs."""
    summary = {
        key: summarize([run[key] for run in runs if key in run])
        for key in SUMMARY_KEYS
        if any(key in run for run in runs)
    }
    task_keys = sorted(
        set.intersection(
            *[
                {key for key in run if key.startswith(TASK_PREFIX)}
                for run in runs
            ]
        )
    )
    per_task = {
        key.removeprefix(TASK_PREFIX): summarize([run[key] for run in runs])
        for key in task_keys
    }
    return {"num_runs": len(runs), "summary": summary, "per_task": per_task}


def compare_tasks(
    candidate: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    """Compare mean task success for tasks shared by both aggregates."""
    candidate_tasks = candidate["per_task"]
    baseline_tasks = baseline["per_task"]
    shared_tasks = sorted(set(candidate_tasks) & set(baseline_tasks))
    deltas = {
        task: candidate_tasks[task]["mean"] - baseline_tasks[task]["mean"]
        for task in shared_tasks
    }
    tolerance = 1e-12
    return {
        "shared_tasks": len(shared_tasks),
        "improved_tasks": sum(delta > tolerance for delta in deltas.values()),
        "regressed_tasks": sum(delta < -tolerance for delta in deltas.values()),
        "unchanged_tasks": sum(abs(delta) <= tolerance for delta in deltas.values()),
        "mean_task_delta": statistics.fmean(deltas.values()) if deltas else 0.0,
        "per_task_delta": deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate",
        nargs="+",
        type=Path,
        help="Candidate metrics.jsonl files, one per seed.",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        type=Path,
        help="Optional baseline metrics.jsonl files.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = aggregate_runs([load_last_evaluation(path) for path in args.candidate])
    result: dict[str, object] = {"candidate": candidate}
    if args.baseline:
        baseline = aggregate_runs([load_last_evaluation(path) for path in args.baseline])
        result["baseline"] = baseline
        result["comparison"] = compare_tasks(candidate, baseline)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
