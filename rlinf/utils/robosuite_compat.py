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

"""Compatibility shim for robosuite's import-time EGL device check."""

from __future__ import annotations

import importlib.machinery
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Sequence

_ROBOSUITE_BINDINGS_MODULE = "robosuite.utils.binding_utils"
_MUJOCO_EGL_DEVICE_ID_ENV = "MUJOCO_EGL_DEVICE_ID"


class _EGLDeviceHidingLoader(Loader):
    """Run robosuite's bindings module with the EGL index temporarily unset."""

    def __init__(self, loader: Loader) -> None:
        self._loader = loader

    def create_module(self, spec: ModuleSpec) -> Optional[ModuleType]:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        device_id = os.environ.pop(_MUJOCO_EGL_DEVICE_ID_ENV, None)
        try:
            self._loader.exec_module(module)
        finally:
            if device_id is not None:
                os.environ[_MUJOCO_EGL_DEVICE_ID_ENV] = device_id

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


class _RobosuiteBindingsFinder(MetaPathFinder):
    """Wrap only robosuite's import-time EGL validation."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        if fullname != _ROBOSUITE_BINDINGS_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _EGLDeviceHidingLoader(spec.loader)
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        return spec


def install_robosuite_egl_device_shim() -> None:
    """Allow a real EGL index that differs from ``CUDA_VISIBLE_DEVICES``."""
    if any(isinstance(finder, _RobosuiteBindingsFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _RobosuiteBindingsFinder())
