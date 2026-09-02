"""Core GSE residual linear layer."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..router_fsdp import mark_adapter_router
from .config import GSEConfig
from .context import get_gse_routing_context
from .initialization import initialize_expert_factors, initialize_router


class GSEExpert(nn.Module):
    """One low-rank residual expert."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        dropout: float,
        scaling: float,
    ) -> None:
        super().__init__()
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("scaling", torch.tensor(float(scaling)))
        self.register_buffer("initial_lora_a", None, persistent=False)
        self.register_buffer("initial_lora_b", None, persistent=False)

    @torch.no_grad()
    def preserve_initial_output(self) -> None:
        """Freeze the initialized factors as a non-persistent output baseline."""
        self.initial_lora_a = self.lora_a.weight.detach().clone()
        self.initial_lora_b = self.lora_b.weight.detach().clone()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank residual transformation."""
        residual, _ = self.forward_components(inputs)
        return residual

    def forward_components(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the trainable residual and its frozen initial component."""
        adapter_inputs = inputs.to(self.lora_a.weight.dtype)
        adapter_inputs = self.dropout(adapter_inputs)
        outputs = self.lora_b(self.lora_a(adapter_inputs))
        initial_outputs = None
        if self.initial_lora_a is not None and self.initial_lora_b is not None:
            initial_hidden = F.linear(adapter_inputs, self.initial_lora_a)
            initial_outputs = F.linear(initial_hidden, self.initial_lora_b)
            outputs = outputs - initial_outputs
        scaled_initial = (
            None if initial_outputs is None else initial_outputs * self.scaling
        )
        return outputs * self.scaling, scaled_initial


class GSEAdapter(nn.Module):
    """Group all trainable GSE parameters behind one callable FSDP boundary."""

    _is_gse_adapter = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: GSEConfig,
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
        self._cached_semantic_embeddings: torch.Tensor | None = None
        self._cached_action_token_mask: torch.Tensor | None = None
        self._cached_sequence_mask: torch.Tensor | None = None

        ranks = config.expert_ranks
        experts = [
            GSEExpert(
                in_features,
                out_features,
                rank,
                config.lora_dropout,
                config.scaling_for_rank(rank, in_features),
            )
            for rank in ranks
        ]
        split = config.num_generalized_experts
        self.generalized_experts = nn.ModuleList(experts[:split])
        self.specialized_experts = nn.ModuleList(experts[split:])
        rank_expert_indices = torch.repeat_interleave(
            torch.arange(len(experts)), torch.tensor(ranks)
        )
        self.register_buffer(
            "_rank_expert_indices", rank_expert_indices, persistent=False
        )
        router_in_features = (
            config.total_rank if config.router_input == "rank_rms" else in_features
        )
        self.router = (
            nn.Linear(
                router_in_features,
                config.num_specialized_experts,
                bias=config.router_bias,
            )
            if config.num_specialized_experts > 0
            else nn.Identity()
        )
        self.semantic_router = (
            nn.Linear(
                config.semantic_embedding_dim,
                config.num_specialized_experts,
                bias=False,
            )
            if config.semantic_conditioning
            and config.semantic_embedding_dim is not None
            and config.num_specialized_experts > 0
            else nn.Identity()
        )

        initialize_expert_factors(
            [expert.lora_a for expert in experts],
            [expert.lora_b for expert in experts],
            method=config.initialization,
            seed=config.init_seed,
            orthogonal_gain=config.orthogonal_gain,
            base_weight=base_weight,
            scalings=[float(expert.scaling) for expert in experts],
            svd_rho=config.svd_rho,
        )
        if config.initialization == "svd":
            if config.preserve_svd_output:
                for expert in experts:
                    expert.preserve_initial_output()
            else:
                self._adjust_base_weight_for_svd(base_weight, experts, split)
        if config.num_specialized_experts > 0:
            router_seed = None if config.init_seed is None else config.init_seed + 1
            initialize_router(
                self.router,
                method=config.router_initialization,
                standard_deviation=config.router_init_std,
                seed=router_seed,
            )
            if config.routing_mode == "uniform":
                self.router.requires_grad_(False)
                self.semantic_router.requires_grad_(False)
            if config.semantic_conditioning and config.num_specialized_experts > 0:
                initialize_router(
                    self.semantic_router,
                    method=config.router_initialization,
                    standard_deviation=config.router_init_std,
                    seed=None if router_seed is None else router_seed + 1,
                )
        # Give each router its own FSDP unit so optim.gse_router_lr can still
        # find it by name; see rlinf/models/peft/router_fsdp.py.
        mark_adapter_router(self.router)
        mark_adapter_router(self.semantic_router)
        self.to(device=device, dtype=dtype)

    @staticmethod
    @torch.no_grad()
    def _adjust_base_weight_for_svd(
        base_weight: torch.Tensor,
        experts: list[GSEExpert],
        num_generalized_experts: int,
    ) -> None:
        """Subtract the expected initialized GSE contribution from the base.

        This legacy mode averages the always-on generalized experts and the
        routed experts under a uniform router prior. Exact per-sample
        preservation uses frozen expert baselines instead.
        """
        num_specialized = len(experts) - num_generalized_experts
        residual = torch.zeros_like(base_weight, dtype=torch.float32)
        if num_generalized_experts:
            generalized = sum(
                expert.scaling.float()
                * (expert.lora_b.weight.float() @ expert.lora_a.weight.float())
                for expert in experts[:num_generalized_experts]
            )
            residual.add_(generalized / num_generalized_experts)
        if num_specialized:
            specialized = sum(
                expert.scaling.float()
                * (expert.lora_b.weight.float() @ expert.lora_a.weight.float())
                for expert in experts[num_generalized_experts:]
            )
            residual.add_(specialized / num_specialized)
        base_weight.sub_(
            residual.to(device=base_weight.device, dtype=base_weight.dtype)
        )

    @property
    def all_experts(self) -> tuple[GSEExpert, ...]:
        """Return generalized and specialized experts in rank-allocation order."""
        return tuple(self.generalized_experts) + tuple(self.specialized_experts)

    @property
    def load_balancing_loss(self) -> torch.Tensor | None:
        """Return the load-balancing loss from the most recent forward pass."""
        return self._load_balancing_loss

    @property
    def router_stats(self) -> dict[str, torch.Tensor]:
        """Return detached routing diagnostics from the latest forward pass."""
        return dict(self._router_stats)

    def reset_auxiliary_state(self) -> None:
        """Discard losses and diagnostics saved by the latest forward pass."""
        self._load_balancing_loss = None
        self._router_stats = {}
        self._cached_semantic_embeddings = None
        self._cached_action_token_mask = None
        self._cached_sequence_mask = None

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalize correlation between rows of all expert A factors."""
        weights = torch.cat([expert.lora_a.weight for expert in self.all_experts])
        normalized = F.normalize(weights.float(), p=2, dim=1)
        gram = normalized @ normalized.mT
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        off_diagonal = gram - identity
        denominator = max(gram.numel() - gram.shape[0], 1)
        return off_diagonal.square().sum() / denominator

    def _generalized_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.generalized_experts:
            output_shape = (*inputs.shape[:-1], self.out_features)
            return inputs.new_zeros(output_shape)
        residual = sum(expert(inputs) for expert in self.generalized_experts)
        return residual / len(self.generalized_experts)

    def _routing_probabilities(self, routing_inputs: torch.Tensor) -> torch.Tensor:
        # The dtype cast lives in the router's forward pre-hook, not here: once
        # the router is its own FSDP unit, reading router.weight.dtype from the
        # caller sees the pre-unshard dtype. See peft/router_fsdp.py.
        logits = self.router(routing_inputs)
        if self.config.semantic_conditioning:
            context = get_gse_routing_context()
            if context is not None and context.semantic_embeddings is not None:
                self._cached_semantic_embeddings = context.semantic_embeddings.detach()
            semantic = self._cached_semantic_embeddings
            if semantic is None:
                raise RuntimeError(
                    "semantic_conditioning=True requires an active GSE routing context"
                )
            semantic_logits = self.semantic_router(semantic)
            if semantic_logits.shape[0] != logits.shape[0]:
                if logits.shape[0] % semantic_logits.shape[0] != 0:
                    raise ValueError(
                        "Semantic routing batch does not match GSE token batch"
                    )
                semantic_logits = semantic_logits.repeat_interleave(
                    logits.shape[0] // semantic_logits.shape[0], dim=0
                )
            logits = logits + self.config.semantic_router_scale * semantic_logits
        return F.softmax(logits.float(), dim=-1)

    def _routing_inputs_for_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use an action-state pooled context for action tokens in LLM layers."""
        context = get_gse_routing_context()
        if context is not None and context.action_token_mask is not None:
            self._cached_action_token_mask = context.action_token_mask.detach()
        if context is not None and context.sequence_mask is not None:
            self._cached_sequence_mask = context.sequence_mask.detach()
        action_token_mask = self._cached_action_token_mask
        if (
            not self.config.action_sequence_routing
            or action_token_mask is None
            or getattr(self, "gse_domain", None) != "llm"
            or inputs.ndim < 3
            or tuple(action_token_mask.shape) != tuple(inputs.shape[:2])
        ):
            return inputs.reshape(-1, self.in_features)
        mask = action_token_mask.to(device=inputs.device, dtype=torch.bool)
        sequence_mask = self._cached_sequence_mask
        if sequence_mask is not None and tuple(sequence_mask.shape) == tuple(
            inputs.shape[:2]
        ):
            sequence_mask = sequence_mask.to(device=inputs.device, dtype=torch.bool)
            pooled = (inputs * sequence_mask.unsqueeze(-1)).sum(
                dim=1
            ) / sequence_mask.sum(dim=1, keepdim=True).clamp_min(1).to(inputs.dtype)
        else:
            pooled = inputs.mean(dim=1)
        routing_inputs = inputs.clone()
        routing_inputs[mask] = pooled.unsqueeze(1).expand_as(inputs)[mask]
        return routing_inputs.reshape(-1, self.in_features)

    def _uniform_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        """Average every expert without consulting a router."""
        if self._can_fuse_experts(self.all_experts):
            weights = inputs.new_full(
                (len(self.all_experts),), 1.0 / len(self.all_experts)
            )
            return self._fused_expert_residual(
                inputs,
                self.all_experts,
                weights,
                self._rank_expert_indices,
            )
        residual = sum(expert(inputs) for expert in self.all_experts)
        return residual / len(self.all_experts)

    def _select_experts(
        self, probabilities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.routing_mode == "all":
            weights = probabilities
            indices = torch.arange(
                probabilities.shape[-1], device=probabilities.device
            ).expand_as(probabilities)
        else:
            weights, indices = torch.topk(
                probabilities,
                k=self.config.top_k,
                dim=-1,
            )
            if self.config.normalize_topk:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self._record_routing(probabilities, indices)
        return weights, indices

    def _can_fuse_experts(self, experts: Any) -> bool:
        return bool(experts) and self.config.lora_dropout == 0

    def _fused_expert_residual(
        self,
        inputs: torch.Tensor,
        experts: tuple[GSEExpert, ...],
        expert_weights: torch.Tensor,
        rank_expert_indices: torch.Tensor,
        rank_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate weighted experts with two fused low-rank projections."""
        lora_b = torch.cat([expert.lora_b.weight for expert in experts], dim=1)
        if rank_hidden is None:
            lora_a = torch.cat([expert.lora_a.weight for expert in experts], dim=0)
            rank_hidden = F.linear(inputs.to(lora_a.dtype), lora_a)

        rank_weights = expert_weights.to(rank_hidden.dtype).index_select(
            -1, rank_expert_indices
        )
        while rank_weights.ndim < rank_hidden.ndim:
            rank_weights = rank_weights.unsqueeze(-2)
        rank_scaling = torch.stack([expert.scaling for expert in experts]).index_select(
            0, rank_expert_indices
        )
        weighted_hidden = rank_hidden * rank_weights * rank_scaling
        residual = F.linear(weighted_hidden, lora_b)

        if experts[0].initial_lora_a is not None:
            initial_lora_a = torch.cat(
                [expert.initial_lora_a for expert in experts], dim=0
            )
            initial_lora_b = torch.cat(
                [expert.initial_lora_b for expert in experts], dim=1
            )
            initial_hidden = F.linear(inputs.to(initial_lora_a.dtype), initial_lora_a)
            # Numerically subtract the same routed initialization, but detach
            # only its gate weights. This preserves the SFT function exactly
            # while giving the router the live-expert gradient used by GSE.
            weighted_initial_hidden = (
                initial_hidden * rank_weights.detach() * rank_scaling
            )
            residual = residual - F.linear(weighted_initial_hidden, initial_lora_b)

        return residual

    def _fused_routed_residual(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        rank_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.routing_mode == "all":
            specialized_weights = weights
        else:
            specialized_weights = weights.new_zeros(
                weights.shape[0], len(self.specialized_experts)
            )
            specialized_weights.scatter_(-1, indices, weights)
        if self.generalized_experts:
            generalized_weights = weights.new_full(
                (weights.shape[0], len(self.generalized_experts)),
                1.0 / len(self.generalized_experts),
            )
            expert_weights = torch.cat(
                (generalized_weights, specialized_weights), dim=-1
            )
        else:
            expert_weights = specialized_weights
        return self._fused_expert_residual(
            inputs,
            self.all_experts,
            expert_weights,
            self._rank_expert_indices,
            rank_hidden,
        )

    def _record_routing(
        self,
        probabilities: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> None:
        num_experts = len(self.specialized_experts)
        counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
        fractions = counts.float() / selected_experts.numel()
        expert_loads = counts.float() / probabilities.shape[0]
        mean_probabilities = probabilities.mean(dim=0)
        self._load_balancing_loss = num_experts * torch.sum(
            expert_loads.to(mean_probabilities) * mean_probabilities
        )
        entropy = -torch.sum(
            probabilities * probabilities.clamp_min(1e-12).log(), dim=-1
        ).mean()
        self._router_stats = {
            "selection_fraction": fractions.detach(),
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
                    "selected_experts": selected_experts.detach(),
                }
            )

    def _token_routed_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.reshape(-1, self.in_features)
        routing_inputs = self._routing_inputs_for_tokens(inputs)
        probabilities = self._routing_probabilities(routing_inputs)
        weights, indices = self._select_experts(probabilities)
        if self._can_fuse_experts(self.all_experts):
            residual = self._fused_routed_residual(flattened, weights, indices)
            return residual.to(inputs.dtype).reshape(
                *inputs.shape[:-1], self.out_features
            )
        residual = flattened.new_zeros(flattened.shape[0], self.out_features)

        for expert_index, expert in enumerate(self.specialized_experts):
            item_indices, slots = torch.where(indices == expert_index)
            if item_indices.numel() == 0:
                continue
            expert_outputs, initial_outputs = expert.forward_components(
                flattened[item_indices]
            )
            selected_weights = weights[item_indices, slots, None].to(residual.dtype)
            weighted_outputs = selected_weights * expert_outputs.to(residual.dtype)
            if initial_outputs is not None:
                weighted_outputs += (
                    selected_weights - selected_weights.detach()
                ) * initial_outputs.to(residual.dtype)
            residual[item_indices] += weighted_outputs
        specialized = residual.reshape(*inputs.shape[:-1], self.out_features)
        return self._generalized_residual(inputs).to(inputs.dtype) + specialized

    def _sequence_context(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim <= 2:
            return inputs.reshape(-1, self.in_features)
        sequences = inputs.reshape(inputs.shape[0], -1, self.in_features)
        if self.config.sequence_pooling == "first":
            return sequences[:, 0]
        if self.config.sequence_pooling == "last":
            return sequences[:, -1]
        return sequences.mean(dim=1)

    def _rank_space_hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        lora_a = torch.cat([expert.lora_a.weight for expert in self.all_experts], dim=0)
        return F.linear(inputs.to(lora_a.dtype), lora_a)

    def _rank_space_rms_context(self, rank_hidden: torch.Tensor) -> torch.Tensor:
        """RMS-pool concatenated expert rank projections per sequence."""
        hidden = rank_hidden.float()
        if rank_hidden.ndim <= 2:
            return hidden.reshape(-1, self.config.total_rank).abs()
        sequences = hidden.reshape(hidden.shape[0], -1, self.config.total_rank)
        return sequences.square().mean(dim=1).sqrt()

    def _sequence_routed_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        rank_hidden = None
        if self.config.router_input == "rank_rms":
            rank_hidden = self._rank_space_hidden(inputs)
            context = self._rank_space_rms_context(rank_hidden)
        else:
            context = self._sequence_context(inputs)
        probabilities = self._routing_probabilities(context)
        weights, indices = self._select_experts(probabilities)
        if self._can_fuse_experts(self.all_experts):
            fused_inputs = inputs if inputs.ndim > 1 else inputs.unsqueeze(0)
            if rank_hidden is not None and inputs.ndim == 1:
                rank_hidden = rank_hidden.unsqueeze(0)
            fused_residual = self._fused_routed_residual(
                fused_inputs, weights, indices, rank_hidden
            ).to(inputs.dtype)
            return fused_residual if inputs.ndim > 1 else fused_residual.squeeze(0)
        residual = inputs.new_zeros(*inputs.shape[:-1], self.out_features)

        routed_inputs = inputs if inputs.ndim > 1 else inputs.unsqueeze(0)
        routed_residual = residual if inputs.ndim > 1 else residual.unsqueeze(0)
        for expert_index, expert in enumerate(self.specialized_experts):
            sequence_indices, slots = torch.where(indices == expert_index)
            if sequence_indices.numel() == 0:
                continue
            expert_outputs, initial_outputs = expert.forward_components(
                routed_inputs[sequence_indices]
            )
            shape = (sequence_indices.shape[0],) + (1,) * (expert_outputs.ndim - 1)
            selected_weights = (
                weights[sequence_indices, slots]
                .reshape(shape)
                .to(routed_residual.dtype)
            )
            weighted_outputs = selected_weights * expert_outputs.to(
                routed_residual.dtype
            )
            if initial_outputs is not None:
                weighted_outputs += (
                    selected_weights - selected_weights.detach()
                ) * initial_outputs.to(routed_residual.dtype)
            routed_residual[sequence_indices] += weighted_outputs
        specialized = routed_residual if inputs.ndim > 1 else routed_residual.squeeze(0)
        return self._generalized_residual(inputs).to(inputs.dtype) + specialized

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute the residual and expose its differentiable auxiliary loss.

        Returning the load-balancing loss is required for FSDP. FSDP only
        installs its pre-backward unshard hook on tensors returned by the
        wrapped module; keeping this loss solely as module state leaves its
        router-parameter branch outside that boundary.
        """
        if self.config.routing_mode == "uniform":
            residual = self._uniform_residual(inputs)
        elif self.config.routing_granularity == "token":
            residual = self._token_routed_residual(inputs)
        else:
            residual = self._sequence_routed_residual(inputs)
        return residual.to(inputs.dtype), self._load_balancing_loss


class GSELinear(nn.Module):
    """Wrap a linear layer with a frozen base and grouped GSE adapter."""

    def __init__(self, base_layer: nn.Linear, config: GSEConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("GSELinear only supports torch.nn.Linear")
        config.validate_for_layer(base_layer.in_features, base_layer.out_features)

        self.base_layer = base_layer
        self.config = config
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self._load_balancing_loss: torch.Tensor | None = None
        self.adapter = GSEAdapter(
            self.in_features,
            self.out_features,
            config,
            base_weight=base_layer.weight,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
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
    def generalized_experts(self) -> nn.ModuleList:
        """Expose always-active experts for inspection and freezing utilities."""
        return self.adapter.generalized_experts

    @property
    def specialized_experts(self) -> nn.ModuleList:
        """Expose routed experts for inspection and freezing utilities."""
        return self.adapter.specialized_experts

    @property
    def router(self) -> nn.Linear:
        """Expose the specialized-expert router."""
        return self.adapter.router

    @property
    def semantic_router(self) -> nn.Module:
        """Expose the frozen-instruction conditioning projection."""
        return self.adapter.semantic_router

    @property
    def all_experts(self) -> tuple[GSEExpert, ...]:
        """Return generalized and specialized experts in rank-allocation order."""
        return self.adapter.all_experts

    @property
    def load_balancing_loss(self) -> torch.Tensor | None:
        """Return the load-balancing loss from the most recent forward pass."""
        return self._load_balancing_loss

    @property
    def router_stats(self) -> dict[str, torch.Tensor]:
        """Return detached routing diagnostics from the latest forward pass."""
        return self.adapter.router_stats

    def reset_auxiliary_state(self) -> None:
        """Discard losses and diagnostics saved by the latest forward pass."""
        self._load_balancing_loss = None
        self.adapter.reset_auxiliary_state()

    def orthogonality_loss(self) -> torch.Tensor:
        """Return the adapter A-factor orthogonality loss."""
        return self.adapter.orthogonality_loss()

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the frozen base layer plus a GSE residual update."""
        base_outputs = self.base_layer(inputs, *args, **kwargs)
        residual, self._load_balancing_loss = self.adapter(inputs)
        return base_outputs + residual.to(base_outputs.dtype)
