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

import gc
import multiprocessing
import warnings
from multiprocessing import connection
from typing import Any, Callable, Optional, Union

import gym
import numpy as np

from rlinf.envs.libero.utils import get_libero_type
from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)

# ---------------------------------------------------------------------------
# Dynamic Module Import Logic for Libero Pro / Plus
# ---------------------------------------------------------------------------
libero_type = get_libero_type()

if libero_type == "pro":
    try:
        from liberopro.liberopro.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=pro but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

elif libero_type == "plus":
    try:
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=plus but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

else:
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        try:
            from liberopro.liberopro.envs import OffScreenRenderEnv
        except ImportError:
            try:
                from liberoplus.liberoplus.envs import OffScreenRenderEnv
            except ImportError:
                raise ImportError(
                    "Could not import OffScreenRenderEnv from libero, liberopro, or liberoplus."
                )


gym_old_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)

_ENV_START_TIMEOUT_SECONDS = 180
_ENV_CLOSE_TIMEOUT_SECONDS = 30


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    # The parent waits for this before starting the next environment. Besides
    # surfacing initialization failures immediately, this avoids creating every
    # EGL context assigned to one GPU at the same instant.
    p.send(("ready", None))
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            elif cmd == "reconfigure":
                env.close()
                # robosuite/MuJoCo leave reference cycles behind close();
                # without an explicit collect the long-lived worker process
                # accumulates ~100 MB of host memory per task switch.
                del env
                gc.collect()
                seed = data.pop("seed")
                env = OffScreenRenderEnv(**data)
                env.seed(seed)
                p.send(None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class ReconfigureSubprocEnvWorker(SubprocEnvWorker):
    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        self._wait_until_ready()
        EnvWorker.__init__(self, env_fn)

    def _wait_until_ready(self) -> None:
        """Wait until the child has created its simulator and EGL context."""
        if not self.parent_remote.poll(_ENV_START_TIMEOUT_SECONDS):
            self.process.terminate()
            self.process.join(10)
            raise TimeoutError(
                "LIBERO environment initialization timed out after "
                f"{_ENV_START_TIMEOUT_SECONDS}s; check the NVIDIA driver and EGL logs"
            )
        try:
            message = self.parent_remote.recv()
        except EOFError as exc:
            self.process.join(10)
            raise RuntimeError(
                "LIBERO environment subprocess exited while creating its EGL context"
            ) from exc
        if message != ("ready", None):
            raise RuntimeError(f"Unexpected LIBERO worker startup message: {message!r}")

    def _close_process(self) -> None:
        """Release EGL in the child before using termination as a fallback."""
        if not self.process.is_alive():
            self.process.join(10)
            return

        try:
            self.parent_remote.send(["close", None])
            if self.parent_remote.poll(_ENV_CLOSE_TIMEOUT_SECONDS):
                try:
                    self.parent_remote.recv()
                except EOFError:
                    pass
                self.process.join(10)
        except (BrokenPipeError, EOFError, OSError):
            pass

        if self.process.is_alive():
            # This path is reserved for an already-broken child. Normal task
            # switches must let robosuite destroy the EGL context cleanly.
            self.process.terminate()
            self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(10)

    def reconfigure_env_fn(self, env_fn_param):
        # Respawn the subprocess to reclaim robosuite/MuJoCo native allocations,
        # but close it gracefully first. Sending SIGTERM to a process that owns
        # an EGL context can leave cleanup to the NVIDIA kernel driver; repeated
        # concurrent cleanup triggered a driver general-protection fault on the
        # eight-GPU LIBERO run.
        param = dict(env_fn_param)
        self._close_process()
        self.parent_remote.close()

        def env_fn(param=param):
            import os

            os.environ.setdefault("LIBERO_TYPE", get_libero_type())
            from libero.libero.envs import OffScreenRenderEnv

            seed = param.pop("seed")
            env = OffScreenRenderEnv(**param)
            env.seed(seed)
            return env

        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        self._wait_until_ready()
        return None


class ReconfigureSubprocEnv(SubprocVectorEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> ReconfigureSubprocEnvWorker:
            return ReconfigureSubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])

    def seed(self, seed=None, id=None):
        """Seed only the selected subprocess environments.

        ``BaseVectorEnv.seed`` always starts at worker zero, which makes a
        partial LIBERO auto-reset reseed unrelated environments. This variant
        keeps the seed list aligned with the explicitly selected worker IDs.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        if seed is None:
            seed_list = [None] * len(id)
        elif isinstance(seed, (int, np.integer)):
            seed_list = [int(seed) + i for i in range(len(id))]
        else:
            seed_list = list(seed)

        if len(seed_list) != len(id):
            raise ValueError(
                f"Expected {len(id)} seeds for environment IDs {list(id)}, "
                f"got {len(seed_list)}"
            )
        return [
            self.workers[env_id].seed(value) for env_id, value in zip(id, seed_list)
        ]
