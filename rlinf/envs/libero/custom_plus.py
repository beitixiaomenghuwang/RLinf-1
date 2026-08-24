"""Manifest-backed LIBERO-90 custom Plus-OOD benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

MANIFEST_ENV = "RLINF_LIBERO90_PLUS_OOD_MANIFEST"


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class LIBERO_90_PLUS_OOD:
    """Expose generated variants through the LIBERO benchmark interface."""

    def __init__(self, task_order_index: int = 0):
        del task_order_index
        manifest_value = os.environ.get(MANIFEST_ENV)
        if not manifest_value:
            raise RuntimeError(
                f"{MANIFEST_ENV} must point to a generated custom Plus-OOD manifest"
            )
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Custom Plus-OOD manifest not found: {manifest_path}"
            )

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("suite") != "libero_90_plus_ood_custom":
            raise ValueError(
                f"Unsupported custom Plus-OOD manifest suite: {manifest.get('suite')!r}"
            )
        self.name = "libero_90_plus_ood"
        self.manifest_path = manifest_path
        self.tasks = []
        self._init_states = []
        state_cache = {}
        for entry in manifest.get("tasks", []):
            task = SimpleNamespace(
                name=entry["id"],
                language=entry["language"],
                problem="Libero",
                problem_folder="libero_90_plus_ood",
                bddl_file=entry["bddl_path"],
                init_states_file=entry["init_states_path"],
            )
            self.tasks.append(task)
            init_path = Path(entry["init_states_path"]).expanduser()
            states = state_cache.get(init_path)
            if states is None:
                states = _load_torch(init_path)
                state_cache[init_path] = states
            if getattr(states, "ndim", 0) == 0:
                states = states.unsqueeze(0)
            self._init_states.append(states[:1])
        self.n_tasks = len(self.tasks)
        if self.n_tasks == 0:
            raise ValueError(
                f"Custom Plus-OOD manifest contains no tasks: {manifest_path}"
            )

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def get_task_problems(self) -> list[str]:
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self) -> list[str]:
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, index: int) -> str:
        return self.tasks[index].bddl_file

    def get_task_demonstration(self, index: int) -> str:
        del index
        raise RuntimeError("Custom Plus-OOD variants do not provide demonstrations")

    def get_task(self, index: int):
        return self.tasks[index]

    def get_task_init_states(self, index: int):
        return self._init_states[index]


def get_custom_benchmark():
    return LIBERO_90_PLUS_OOD
