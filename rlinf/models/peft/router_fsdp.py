"""FSDP plumbing shared by the adapter routers (GSE, Ortho-Hydra).

A dedicated router learning rate (``optim.gse_router_lr``) is matched by
parameter name. Under FSDP1 with ``use_orig_params=False`` a whole adapter is
fused into one ``adapter._fsdp_wrapped_module._flat_param``, which erases the
``.router.`` / ``.semantic_router.`` names the optimizer group needs -- the
group comes out empty and every router silently trains at ``optim.lr``.

Marking each router makes ``get_fsdp_wrap_policy`` give it its own FSDP unit,
so its name survives as
``...adapter._fsdp_wrapped_module.router._fsdp_wrapped_module._flat_param``.
Routers are tiny ``Linear(width, num_specialized)`` modules, so the extra units
cost essentially nothing, and nothing at all under ``NO_SHARD``.
"""

import torch
from torch import nn


def _cast_router_input(module: nn.Module, args: tuple) -> tuple | None:
    """Cast the routing input to the router's dtype at call time.

    Callers cannot do this themselves: once the router is its own FSDP unit,
    reading ``router.weight.dtype`` outside the call happens *before* FSDP
    unshards and casts the parameter, so under mixed precision it reports fp32
    while ``F.linear`` sees bf16. A pre-hook on the inner module runs after the
    unshard, so it always sees the dtype the matmul will actually use. FSDP only
    casts forward inputs automatically for the root unit.
    """
    if not args or not isinstance(args[0], torch.Tensor):
        return None
    return (args[0].to(module.weight.dtype), *args[1:])


def mark_adapter_router(router: nn.Module) -> nn.Module:
    """Tag a router for its own FSDP unit and keep its input dtype correct."""
    if not isinstance(router, nn.Linear):
        # Adapters use ``nn.Identity`` when there is nothing to route over.
        return router
    router._is_adapter_router = True
    router.register_forward_pre_hook(_cast_router_input)
    return router
