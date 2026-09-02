"""Ortho-Hydra residual linear layer."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..router_fsdp import mark_adapter_router
from .config import OrthoHydraConfig
from .context import get_ortho_hydra_routing_context
from .initialization import initialize_router, principal_bases


def _cayley(parameters: torch.Tensor) -> torch.Tensor:
    """Return exact Cayley rotations for a matrix or matrix batch."""
    skew = parameters.float() - parameters.float().transpose(-2, -1)
    identity = torch.eye(skew.shape[-1], device=skew.device, dtype=skew.dtype)
    if skew.ndim == 3:
        identity = identity.unsqueeze(0).expand_as(skew)
    return torch.linalg.solve(identity + skew, identity - skew)


class OrthoHydraExpert(nn.Module):
    """One rank-r expert with disjoint frozen principal bases."""

    def __init__(
        self,
        p_basis: torch.Tensor,
        q_basis: torch.Tensor | None,
    ) -> None:
        super().__init__()
        rank = p_basis.shape[-1]
        self.register_buffer("p_basis", p_basis.contiguous())
        self.s_p = nn.Parameter(torch.zeros(rank, rank))
        if q_basis is None:
            self.register_buffer("q_basis", None)
            self.register_parameter("s_q", None)
        else:
            self.register_buffer("q_basis", q_basis.contiguous())
            self.s_q = nn.Parameter(torch.zeros(rank, rank))


class OrthoHydraAdapter(nn.Module):
    """Group Ortho-Hydra parameters behind one FSDP boundary."""

    _is_peft_adapter = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: OrthoHydraConfig,
        *,
        base_weight: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self._load_balancing_loss: torch.Tensor | None = None
        self._router_stats: dict[str, torch.Tensor] = {}
        self._inference_basis_cache: dict[
            str, tuple[tuple[tuple[int, int], ...], torch.Tensor]
        ] = {}
        # Activation-checkpoint recomputation may run after the outer
        # ContextVar has been restored; retain batch-local routing semantics
        # until the corresponding backward completes.
        self._cached_semantic_embeddings: torch.Tensor | None = None

        left, right = principal_bases(base_weight, config.total_rank)
        rank = config.expert_rank
        experts = []
        for index in range(config.num_experts):
            start = index * rank
            stop = start + rank
            q_basis = (
                right[start:stop] if config.parameterization == "independent" else None
            )
            experts.append(OrthoHydraExpert(left[:, start:stop], q_basis))
        self.experts = nn.ModuleList(experts)

        if config.parameterization == "shared_a":
            # The adapter uses one rank-r Q/A basis shared by all experts. A
            # separate rank-R basis preserves the requested total-rank router
            # feature width without turning the shared adapter into R/r slices.
            self.register_buffer("q_basis", right[:rank].contiguous())
            self.s_q = nn.Parameter(torch.zeros(rank, rank))
            self.register_buffer("routing_q_basis", right.contiguous())
            self.s_routing_q = nn.Parameter(
                torch.zeros(config.total_rank, config.total_rank)
            )
        else:
            self.register_buffer("q_basis", None)
            self.register_parameter("s_q", None)
            self.register_buffer("routing_q_basis", None)
            self.register_parameter("s_routing_q", None)

        lambda_shape = (
            (config.expert_rank,)
            if config.parameterization == "shared_a"
            else (config.num_experts, config.expert_rank)
        )
        self.lambda_layer = nn.Parameter(torch.zeros(lambda_shape))
        self.router = nn.Linear(
            config.total_rank,
            config.num_experts,
            bias=config.router_bias,
        )
        self.semantic_router = nn.Linear(
            config.semantic_embedding_dim,
            config.num_experts,
            bias=False,
        )
        initialize_router(self.router, config.router_init_std, config.init_seed)
        semantic_seed = None if config.init_seed is None else config.init_seed + 1
        initialize_router(self.semantic_router, config.router_init_std, semantic_seed)
        self.dropout = (
            nn.Dropout(config.lora_dropout)
            if config.lora_dropout > 0
            else nn.Identity()
        )
        # Give each router its own FSDP unit so a per-router optimizer group can
        # still find it by name; see rlinf/models/peft/router_fsdp.py.
        mark_adapter_router(self.router)
        mark_adapter_router(self.semantic_router)
        self.to(device=device, dtype=dtype)

    @property
    def load_balancing_loss(self) -> torch.Tensor | None:
        """Return the differentiable balance loss from the latest forward."""
        return self._load_balancing_loss

    @property
    def router_stats(self) -> dict[str, torch.Tensor]:
        """Return detached statistics from the latest forward."""
        return dict(self._router_stats)

    def reset_auxiliary_state(self) -> None:
        """Discard diagnostics saved by the latest forward."""
        self._load_balancing_loss = None
        self._router_stats = {}
        self._cached_semantic_embeddings = None

    def orthogonality_error(self) -> torch.Tensor:
        """Measure cross-expert overlap of the frozen output subspaces."""
        bases = torch.stack([expert.p_basis.float() for expert in self.experts])
        cross = torch.einsum("eor,fos->efrs", bases, bases)
        mask = ~torch.eye(
            self.config.num_experts, device=cross.device, dtype=torch.bool
        )
        error = cross[mask].square().mean()
        return error + self.lambda_layer.sum() * 0.0

    def _effective_basis(
        self,
        name: str,
        parameters: tuple[torch.Tensor, ...],
        build: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Cache a Cayley-transformed basis only for inference forwards."""
        if torch.is_grad_enabled():
            return build()
        signature = tuple(
            (parameter._version, parameter.data_ptr()) for parameter in parameters
        )
        cached = self._inference_basis_cache.get(name)
        if cached is None or cached[0] != signature:
            cached = (signature, build())
            self._inference_basis_cache[name] = cached
        return cached[1]

    def _bottleneck(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adapter_inputs = self.dropout(inputs)
        if self.config.parameterization == "shared_a":
            assert self.s_q is not None and self.q_basis is not None
            assert self.s_routing_q is not None and self.routing_q_basis is not None
            q_effective = self._effective_basis(
                "shared_q",
                (self.s_q,),
                lambda: _cayley(self.s_q).to(self.q_basis) @ self.q_basis,
            )
            hidden = F.linear(adapter_inputs.to(q_effective.dtype), q_effective)
            routing_q = self._effective_basis(
                "routing_q",
                (self.s_routing_q,),
                lambda: (
                    _cayley(self.s_routing_q).to(self.routing_q_basis)
                    @ self.routing_q_basis
                ),
            )
            routing_hidden = F.linear(adapter_inputs.to(routing_q.dtype), routing_q)
            return hidden, routing_hidden

        q_parameters = tuple(expert.s_q for expert in self.experts)
        q_effective = self._effective_basis(
            "independent_q",
            q_parameters,
            lambda: torch.stack(
                [
                    _cayley(expert.s_q).to(expert.q_basis) @ expert.q_basis
                    for expert in self.experts
                ]
            ),
        )
        hidden = torch.einsum(
            "...i,eri->...er", adapter_inputs.to(q_effective.dtype), q_effective
        )
        return hidden, hidden.flatten(start_dim=-2)

    def _routing_probabilities(self, routing_hidden: torch.Tensor) -> torch.Tensor:
        if routing_hidden.ndim < 3:
            pooled = routing_hidden.reshape(1, self.config.total_rank).abs()
        else:
            pooled = routing_hidden.reshape(
                routing_hidden.shape[0], -1, self.config.total_rank
            ).float()
            pooled = pooled.square().mean(dim=1).sqrt()
        # The dtype cast lives in the router's forward pre-hook, not here; see
        # peft/router_fsdp.py for why the caller cannot read weight.dtype.
        logits = self.router(pooled)

        context = get_ortho_hydra_routing_context()
        if context is not None:
            self._cached_semantic_embeddings = context.semantic_embeddings
        semantic = self._cached_semantic_embeddings
        if semantic is None:
            raise RuntimeError(
                "Ortho-Hydra requires frozen text embeddings in its routing context"
            )
        if semantic.shape[0] != logits.shape[0]:
            raise ValueError("Text embedding batch does not match router batch")
        semantic_logits = self.semantic_router(semantic)
        logits = logits + self.config.semantic_router_scale * semantic_logits
        return F.softmax(logits.float(), dim=-1)

    def _record_routing(self, probabilities: torch.Tensor) -> None:
        mean_probabilities = probabilities.mean(dim=0)
        top_experts = probabilities.argmax(dim=-1)
        counts = torch.bincount(top_experts, minlength=self.config.num_experts).float()
        selection_fraction = counts / top_experts.numel()
        self._load_balancing_loss = (
            self.config.num_experts * mean_probabilities.square().sum()
        )
        entropy = -torch.sum(
            probabilities * probabilities.clamp_min(1e-12).log(), dim=-1
        ).mean()
        self._router_stats = {
            "selection_fraction": selection_fraction.detach(),
            "mean_probability": mean_probabilities.detach(),
            "entropy": entropy.detach(),
            "num_routing_items": torch.tensor(
                probabilities.shape[0], device=probabilities.device
            ),
        }
        if self.config.record_routing_assignments:
            self._router_stats.update(
                {
                    "probabilities": probabilities.detach(),
                    "selected_experts": top_experts[:, None].detach(),
                }
            )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply all eight experts under a dense per-sequence softmax gate."""
        squeeze_batch = inputs.ndim == 1
        routed_inputs = inputs.unsqueeze(0) if squeeze_batch else inputs
        hidden, routing_hidden = self._bottleneck(routed_inputs)
        probabilities = self._routing_probabilities(routing_hidden)
        self._record_routing(probabilities)

        p_parameters = tuple(expert.s_p for expert in self.experts)
        p_effective = self._effective_basis(
            "p",
            p_parameters,
            lambda: torch.stack(
                [
                    expert.p_basis @ _cayley(expert.s_p).to(expert.p_basis)
                    for expert in self.experts
                ]
            ),
        )
        scaled_hidden = hidden * self.lambda_layer.to(hidden)
        if self.config.parameterization == "shared_a":
            expert_outputs = torch.einsum(
                "b...r,eor->b...eo", scaled_hidden, p_effective
            )
        else:
            expert_outputs = torch.einsum(
                "b...er,eor->b...eo", scaled_hidden, p_effective
            )
        gate_shape = (probabilities.shape[0],) + (1,) * (expert_outputs.ndim - 3)
        gates = probabilities.reshape(*gate_shape, self.config.num_experts, 1)
        residual = (expert_outputs * gates.to(expert_outputs)).sum(dim=-2)
        residual = residual * self.config.scaling
        if squeeze_batch:
            residual = residual.squeeze(0)
        return residual.to(inputs.dtype), self._load_balancing_loss


class OrthoHydraLinear(nn.Module):
    """Wrap a frozen linear layer with an Ortho-Hydra residual."""

    def __init__(self, base_layer: nn.Linear, config: OrthoHydraConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("OrthoHydraLinear only supports torch.nn.Linear")
        config.validate_for_layer(base_layer.in_features, base_layer.out_features)
        self.base_layer = base_layer
        self.config = config
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.adapter = OrthoHydraAdapter(
            self.in_features,
            self.out_features,
            config,
            base_weight=base_layer.weight,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self._load_balancing_loss: torch.Tensor | None = None
        if config.freeze_base:
            self.base_layer.requires_grad_(False)

    @property
    def weight(self) -> nn.Parameter:
        """Expose the wrapped weight for linear-layer compatibility."""
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        """Expose the wrapped bias for linear-layer compatibility."""
        return self.base_layer.bias

    @property
    def load_balancing_loss(self) -> torch.Tensor | None:
        """Return the latest differentiable balance loss."""
        return self._load_balancing_loss

    @property
    def router_stats(self) -> dict[str, torch.Tensor]:
        """Return latest routing diagnostics."""
        return self.adapter.router_stats

    def reset_auxiliary_state(self) -> None:
        """Clear auxiliary state before another forward."""
        self._load_balancing_loss = None
        self.adapter.reset_auxiliary_state()

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the base linear layer and Ortho-Hydra residual."""
        base_output = self.base_layer(inputs, *args, **kwargs)
        residual, self._load_balancing_loss = self.adapter(inputs)
        return base_output + residual.to(base_output.dtype)
