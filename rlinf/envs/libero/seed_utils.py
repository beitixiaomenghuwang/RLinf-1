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

"""Seed helpers for LIBERO vector environments."""

from collections.abc import Sequence

import numpy as np


def build_libero_env_seeds(
    *,
    base_seed: int,
    seed_offset: int,
    num_envs: int,
    group_size: int,
    env_idx: Sequence[int] | np.ndarray,
    is_eval: bool,
) -> list[int]:
    """Return deterministic simulator seeds for selected local environments.

    Standard LIBERO evaluation follows the reference evaluator and uses one
    fixed seed for every simulator. Training assigns one seed per global
    trajectory group so that all samples in a GRPO group see the same initial
    conditions. The global group index is independent of worker count when the
    total environment count and group size are unchanged.

    Args:
        base_seed: Base simulator seed from the environment configuration.
        seed_offset: Logical environment-worker index.
        num_envs: Number of local environments owned by the worker.
        group_size: Number of trajectories sharing one reset state.
        env_idx: Local environment indices being created or reset.
        is_eval: Whether the environments are used for evaluation.

    Returns:
        One simulator seed per entry in ``env_idx``.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if num_envs <= 0 or num_envs % group_size != 0:
        raise ValueError(
            "num_envs must be positive and divisible by group_size, got "
            f"num_envs={num_envs}, group_size={group_size}"
        )

    local_env_ids = np.asarray(env_idx, dtype=np.int64).reshape(-1)
    if np.any(local_env_ids < 0) or np.any(local_env_ids >= num_envs):
        raise ValueError(
            f"env_idx must be in [0, {num_envs - 1}], got {local_env_ids.tolist()}"
        )

    if is_eval:
        return [int(base_seed)] * len(local_env_ids)

    groups_per_worker = num_envs // group_size
    global_group_ids = (
        int(seed_offset) * groups_per_worker + local_env_ids // group_size
    )
    return (int(base_seed) + global_group_ids).astype(np.int64).tolist()
