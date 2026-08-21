"""Forward-scoped conditioning signals for GSE routers."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator

import torch


@dataclass(frozen=True)
class GSERoutingContext:
    """Per-forward semantic and action-token routing signals."""

    semantic_embeddings: torch.Tensor | None = None
    action_token_mask: torch.Tensor | None = None
    sequence_mask: torch.Tensor | None = None


_ROUTING_CONTEXT: ContextVar[GSERoutingContext | None] = ContextVar(
    "rlinf_gse_routing_context", default=None
)


def get_gse_routing_context() -> GSERoutingContext | None:
    """Return the current forward's routing context, if one is active."""
    return _ROUTING_CONTEXT.get()


@contextmanager
def gse_routing_context(
    semantic_embeddings: torch.Tensor | None,
    action_token_mask: torch.Tensor | None = None,
    sequence_mask: torch.Tensor | None = None,
) -> Iterator[None]:
    """Make frozen semantic features visible to every injected GSE layer."""
    token = _ROUTING_CONTEXT.set(
        GSERoutingContext(
            semantic_embeddings=semantic_embeddings,
            action_token_mask=action_token_mask,
            sequence_mask=sequence_mask,
        )
    )
    try:
        yield
    finally:
        _ROUTING_CONTEXT.reset(token)


def update_gse_routing_context(
    *,
    action_token_mask: torch.Tensor | None = None,
    sequence_mask: torch.Tensor | None = None,
) -> None:
    """Update the action-token mask once multimodal positions are known."""
    current = _ROUTING_CONTEXT.get()
    if current is not None:
        _ROUTING_CONTEXT.set(
            replace(
                current,
                action_token_mask=action_token_mask,
                sequence_mask=sequence_mask,
            )
        )
