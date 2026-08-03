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

"""Composition tests for MetaWorld action-only rank-32 experiments."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[2] / "examples" / "embodiment" / "config"


@pytest.mark.parametrize(
    ("config_name", "normalization_mode"),
    [
        ("metaworld_50_ppo_openpi_pi05_gse_action_r32", "global"),
        (
            "metaworld_50_ppo_openpi_pi05_gse_action_r32_per_task_adv",
            "per_task",
        ),
    ],
)
def test_action_r32_profile_composition(
    config_name: str, normalization_mode: str
) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)

    assert cfg.algorithm.advantage_normalization_mode == normalization_mode
    assert cfg.actor.optim.lr == 5.0e-6
    assert cfg.actor.optim.value_lr == 1.0e-4
    assert cfg.actor.optim.total_training_steps == cfg.runner.max_epochs == 320
    assert cfg.actor.model.gse.total_rank == 32
    assert cfg.actor.model.gse.lora_alpha == 32.0
    assert cfg.actor.model.gse.log_task_router_metrics is True
    assert cfg.actor.model.gse.log_layerwise_task_router_metrics is True
    assert cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch == 256
    assert cfg.actor.global_batch_size == 1024
