# Copyright 2026 The RLinf Authors.
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

import pytest

from rlinf.config import compute_embodied_ppo_samples_per_step


def test_compute_embodied_ppo_samples_per_step() -> None:
    assert compute_embodied_ppo_samples_per_step(96, 2, 100, 5) == 3840
    assert compute_embodied_ppo_samples_per_step(128, 2, 100, 5) == 5120
    assert compute_embodied_ppo_samples_per_step(256, 2, 100, 5) == 10240


def test_compute_embodied_ppo_samples_requires_complete_chunks() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        compute_embodied_ppo_samples_per_step(128, 2, 99, 5)
