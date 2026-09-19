"""Generalized and specialized expert residual adapters."""

from .config import GSEConfig
from .context import (
    get_gse_routing_context,
    gse_routing_context,
    update_gse_routing_context,
)
from .initialization import joint_lora_a, orthogonality_error
from .injector import (
    GSEInjectionReport,
    inject_gse,
    iter_gse_layers,
    mark_only_gse_as_trainable,
)
from .layer import GSEAdapter, GSEExpert, GSELinear
from .losses import (
    collect_gse_gradient_spectrum,
    gse_auxiliary_loss,
    gse_layerwise_task_router_metrics,
    gse_layerwise_task_router_statistics,
    gse_load_balancing_loss,
    gse_orthogonality_loss,
    gse_router_metrics,
    gse_task_router_metrics,
    gse_task_router_metrics_from_tensor,
    gse_task_router_statistics,
    reset_gse_auxiliary_state,
    set_gse_router_stats_enabled,
)
from .state import gse_state_dict, load_gse_state_dict

__all__ = [
    "GSEConfig",
    "GSEAdapter",
    "GSEExpert",
    "GSEInjectionReport",
    "GSELinear",
    "collect_gse_gradient_spectrum",
    "gse_auxiliary_loss",
    "gse_layerwise_task_router_metrics",
    "gse_layerwise_task_router_statistics",
    "gse_load_balancing_loss",
    "gse_orthogonality_loss",
    "gse_router_metrics",
    "gse_task_router_metrics",
    "gse_task_router_metrics_from_tensor",
    "gse_task_router_statistics",
    "gse_state_dict",
    "get_gse_routing_context",
    "gse_routing_context",
    "inject_gse",
    "iter_gse_layers",
    "joint_lora_a",
    "load_gse_state_dict",
    "mark_only_gse_as_trainable",
    "orthogonality_error",
    "reset_gse_auxiliary_state",
    "set_gse_router_stats_enabled",
    "update_gse_routing_context",
]
