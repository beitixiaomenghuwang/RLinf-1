"""SVD-based hidden-state-routed MoORE adapters."""

from .config import MoOREConfig
from .injector import (
    MoOREInjectionReport,
    inject_moore,
    iter_moore_layers,
    mark_only_moore_as_trainable,
)
from .layer import MoOREAdapter, MoORELinear
from .state import load_moore_state_dict, moore_state_dict

__all__ = [
    "MoOREAdapter",
    "MoOREConfig",
    "MoOREInjectionReport",
    "MoORELinear",
    "inject_moore",
    "iter_moore_layers",
    "load_moore_state_dict",
    "mark_only_moore_as_trainable",
    "moore_state_dict",
]
