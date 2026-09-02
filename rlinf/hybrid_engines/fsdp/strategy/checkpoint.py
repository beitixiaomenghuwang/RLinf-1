# Copyright 2025 The RLinf Authors.
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

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Union

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion, to_local_if_dtensor
from rlinf.utils.utils import get_rng_state, set_rng_state

# Substrings that mark a tensor as belonging to a PEFT adapter rather than to the
# frozen backbone. These only ever ADD tensors to an adapter-only checkpoint: the
# trainable set is derived from optimizer state independently, so a marker that
# fails to match cannot drop a parameter that is being optimized. Their job is to
# retain the adapters' *persistent buffers* (GSE `scaling`, Ortho-Hydra
# `p_basis`/`q_basis`), which carry no optimizer state but are part of the
# adapter's definition.
_ADAPTER_NAME_MARKERS = (".adapter.", "value_head.", "lora_", "router.")


class Checkpoint(Stateful):
    def __init__(
        self,
        model: Union[FSDP, FSDPModule],
        optimizers: Union[Optimizer, Iterable[Optimizer]],
        lr_schedulers: Union[LRScheduler, Iterable[LRScheduler]],
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
        checkpoint_format: str = "dcp",
        adapter_only: bool = False,
    ):
        self.model = model
        self.optimizers = optimizers
        self.lr_schedulers = (
            (lr_schedulers,)
            if isinstance(lr_schedulers, LRScheduler)
            else tuple(lr_schedulers)
        )
        self.opts = opts
        self.fsdp_version = fsdp_version
        self.checkpoint_format = checkpoint_format
        self.adapter_only = adapter_only

    @staticmethod
    def _trainable_fqns(optim_state_dicts) -> set[str]:
        """Collect the parameter names that carry optimizer state.

        In a PEFT run this is exactly the trainable set, because a frozen
        parameter never enters ``optimizer.state``. Only string keys are useful:
        ``get_state_dict`` keys optimizer state by FQN, but a raw
        ``optimizer.state_dict()`` (the ``local_shard`` path) keys it by integer
        index, which says nothing about names. Integer keys are therefore
        ignored, leaving the name markers to carry that format.
        """
        candidates = (
            optim_state_dicts
            if isinstance(optim_state_dicts, (list, tuple))
            else [optim_state_dicts]
        )
        names: set[str] = set()
        for state_dict in candidates:
            if isinstance(state_dict, Mapping):
                names.update(
                    key
                    for key in state_dict.get("state", {})
                    if isinstance(key, str)
                )
        return names

    def _select_adapter_tensors(self, model_state_dict, optim_state_dicts) -> dict:
        """Keep the adapter tensors and drop the frozen backbone.

        A PEFT run re-saves a byte-identical multi-GiB backbone at every step
        while the tensors that actually change are a few tens of MiB. The
        backbone is recoverable from ``model.model_path``, so only the adapter is
        worth persisting.
        """
        trainable = self._trainable_fqns(optim_state_dicts)
        selected = {
            name: value
            for name, value in model_state_dict.items()
            if name in trainable
            or any(marker in name for marker in _ADAPTER_NAME_MARKERS)
        }
        if not selected:
            # Persisting an empty model would produce a checkpoint that resumes
            # silently onto base weights, losing all training. Fail instead.
            raise ValueError(
                "adapter_only checkpointing selected no tensors. Neither "
                "optimizer state nor the adapter name markers "
                f"{_ADAPTER_NAME_MARKERS} matched any of the "
                f"{len(model_state_dict)} model tensors."
            )
        return selected

    def _get_local_optim_state_dicts(self):
        if isinstance(self.optimizers, Optimizer):
            return self.optimizers.state_dict()
        return [opt.state_dict() for opt in self.optimizers]

    def _load_local_optim_state_dicts(self, optim_state_dicts):
        if isinstance(self.optimizers, Optimizer):
            self.optimizers.load_state_dict(optim_state_dicts)
        else:
            for opt, opt_sd in zip(self.optimizers, optim_state_dicts):
                opt.load_state_dict(opt_sd)

    def state_dict(self):
        if self.checkpoint_format == "local_shard":
            model_sd = self.model.state_dict()
            model_sd = {
                key: to_local_if_dtensor(value).cpu()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in model_sd.items()
            }
            optim_sd = self._get_local_optim_state_dicts()
        else:
            model_sd, optim_sd = get_state_dict(
                model=self.model,
                optimizers=self.optimizers,
                options=self.opts,
            )

        if self.adapter_only:
            model_sd = self._select_adapter_tensors(model_sd, optim_sd)

        lr_sched_sd = [lr.state_dict() for lr in self.lr_schedulers]

        return {
            "model": model_sd,
            "optimizers": optim_sd,
            "lr_schedulers": lr_sched_sd,
            "fsdp_version": self.fsdp_version.value,
            "rng": get_rng_state(),
        }

    def load_state_dict(self, state):
        assert "fsdp_version" in state, "Checkpoint is missing FSDP version info."
        ckpt_fsdp_version = FSDPVersion(state["fsdp_version"])
        if ckpt_fsdp_version != self.fsdp_version:
            raise ValueError(
                f"FSDP version mismatch: {ckpt_fsdp_version} != {self.fsdp_version}"
            )

        if self.checkpoint_format == "local_shard":
            # An adapter-only checkpoint deliberately omits the frozen backbone,
            # which the freshly built model already holds.
            self.model.load_state_dict(
                state["model"], strict=not self.adapter_only
            )

            self._load_local_optim_state_dicts(state["optimizers"])

        else:
            opts = replace(self.opts, strict=False) if self.adapter_only else self.opts
            set_state_dict(
                model=self.model,
                optimizers=self.optimizers,
                model_state_dict=state["model"],
                optim_state_dict=state.get("optimizers", state.get("optim")),
                options=opts,
            )

        # lr schedulers
        if "lr_schedulers" in state:
            for lr, lr_sd in zip(self.lr_schedulers, state["lr_schedulers"]):
                lr.load_state_dict(lr_sd)

        if "rng" in state:
            set_rng_state(state["rng"])
