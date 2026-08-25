"""Ortho-Hydra parameter-efficient adaptation."""

from .config import OrthoHydraConfig
from .context import (
    get_ortho_hydra_routing_context,
    ortho_hydra_checkpoint_contexts,
    ortho_hydra_routing_context,
)
from .injector import (
    OrthoHydraInjectionReport,
    inject_ortho_hydra,
    iter_ortho_hydra_layers,
    mark_only_ortho_hydra_as_trainable,
)
from .layer import OrthoHydraAdapter, OrthoHydraExpert, OrthoHydraLinear
from .losses import (
    ortho_hydra_auxiliary_loss,
    ortho_hydra_layerwise_task_router_metrics,
    ortho_hydra_layerwise_task_router_statistics,
    ortho_hydra_orthogonality_error,
    ortho_hydra_task_router_metrics_from_tensor,
    reset_ortho_hydra_auxiliary_state,
)

__all__ = [
    "OrthoHydraAdapter",
    "OrthoHydraConfig",
    "OrthoHydraExpert",
    "OrthoHydraInjectionReport",
    "OrthoHydraLinear",
    "get_ortho_hydra_routing_context",
    "inject_ortho_hydra",
    "iter_ortho_hydra_layers",
    "mark_only_ortho_hydra_as_trainable",
    "ortho_hydra_auxiliary_loss",
    "ortho_hydra_checkpoint_contexts",
    "ortho_hydra_layerwise_task_router_metrics",
    "ortho_hydra_layerwise_task_router_statistics",
    "ortho_hydra_orthogonality_error",
    "ortho_hydra_routing_context",
    "ortho_hydra_task_router_metrics_from_tensor",
    "reset_ortho_hydra_auxiliary_state",
]
