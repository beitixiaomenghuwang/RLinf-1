"""Forward-local semantic context for Ortho-Hydra routers."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import ContextManager, Iterator

import torch


@dataclass(frozen=True)
class OrthoHydraRoutingContext:
    """Frozen semantic embeddings associated with the current batch."""

    semantic_embeddings: torch.Tensor


_ROUTING_CONTEXT: ContextVar[OrthoHydraRoutingContext | None] = ContextVar(
    "rlinf_ortho_hydra_routing_context", default=None
)


def get_ortho_hydra_routing_context() -> OrthoHydraRoutingContext | None:
    """Return the active routing context, if any."""
    return _ROUTING_CONTEXT.get()


def ortho_hydra_checkpoint_contexts() -> tuple[
    ContextManager[None], ContextManager[None]
]:
    """Restore semantic routing state during non-reentrant recomputation."""
    routing_context = get_ortho_hydra_routing_context()
    recompute_context = (
        nullcontext()
        if routing_context is None
        else ortho_hydra_routing_context(routing_context.semantic_embeddings)
    )
    return nullcontext(), recompute_context


@contextmanager
def ortho_hydra_routing_context(
    semantic_embeddings: torch.Tensor,
) -> Iterator[None]:
    """Make frozen semantic embeddings visible to every adapted layer."""
    token: Token = _ROUTING_CONTEXT.set(
        OrthoHydraRoutingContext(semantic_embeddings=semantic_embeddings.detach())
    )
    try:
        yield
    finally:
        _ROUTING_CONTEXT.reset(token)
