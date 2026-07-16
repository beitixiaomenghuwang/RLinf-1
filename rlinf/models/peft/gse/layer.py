"""Core GSE residual linear layer."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import GSEConfig
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank residual transformation."""
        adapter_inputs = inputs.to(self.lora_a.weight.dtype)
        outputs = self.lora_b(self.lora_a(self.dropout(adapter_inputs)))
        return outputs * self.scaling


class GSEAdapter(nn.Module):
    """Group all trainable GSE parameters behind one callable FSDP boundary."""

    _is_gse_adapter = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: GSEConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self._load_balancing_loss: torch.Tensor | None = None
        self._router_stats: dict[str, torch.Tensor] = {}

        ranks = config.expert_ranks
        experts = [
            GSEExpert(
                in_features,
                out_features,
                rank,
                config.lora_dropout,
                config.scaling_for_rank(rank),
            )
            for rank in ranks
        ]
        split = config.num_generalized_experts
        self.generalized_experts = nn.ModuleList(experts[:split])
        self.specialized_experts = nn.ModuleList(experts[split:])
        self.router = nn.Linear(
            in_features,
            config.num_specialized_experts,
            bias=config.router_bias,
        )

        initialize_expert_factors(
            [expert.lora_a for expert in experts],
            [expert.lora_b for expert in experts],
            method=config.initialization,
            seed=config.init_seed,
            orthogonal_gain=config.orthogonal_gain,
        )
        router_seed = None if config.init_seed is None else config.init_seed + 1
        initialize_router(
            self.router,
            standard_deviation=config.router_init_std,
            seed=router_seed,
        )
        self.to(device=device, dtype=dtype)

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
        logits = self.router(routing_inputs.to(self.router.weight.dtype))
        return F.softmax(logits.float(), dim=-1)

    def _select_experts(
        self, probabilities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, indices = torch.topk(
            probabilities,
            k=self.config.top_k,
            dim=-1,
        )
        if self.config.normalize_topk:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self._record_routing(probabilities, indices)
        return weights, indices

    def _record_routing(
        self,
        probabilities: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> None:
        num_experts = len(self.specialized_experts)
        counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
        fractions = counts.float() / selected_experts.numel()
        mean_probabilities = probabilities.mean(dim=0)
        self._load_balancing_loss = num_experts * torch.sum(
            fractions.to(mean_probabilities) * mean_probabilities
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

    def _token_routed_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.reshape(-1, self.in_features)
        probabilities = self._routing_probabilities(flattened)
        weights, indices = self._select_experts(probabilities)
        residual = flattened.new_zeros(flattened.shape[0], self.out_features)

        for expert_index, expert in enumerate(self.specialized_experts):
            item_indices, slots = torch.where(indices == expert_index)
            if item_indices.numel() == 0:
                continue
            expert_outputs = expert(flattened[item_indices]).to(residual.dtype)
            residual[item_indices] += (
                weights[item_indices, slots, None].to(residual.dtype) * expert_outputs
            )
        return residual.reshape(*inputs.shape[:-1], self.out_features)

    def _sequence_context(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim <= 2:
            return inputs.reshape(-1, self.in_features)
        sequences = inputs.reshape(inputs.shape[0], -1, self.in_features)
        if self.config.sequence_pooling == "first":
            return sequences[:, 0]
        if self.config.sequence_pooling == "last":
            return sequences[:, -1]
        return sequences.mean(dim=1)

    def _sequence_routed_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        context = self._sequence_context(inputs)
        probabilities = self._routing_probabilities(context)
        weights, indices = self._select_experts(probabilities)
        residual = inputs.new_zeros(*inputs.shape[:-1], self.out_features)

        routed_inputs = inputs if inputs.ndim > 1 else inputs.unsqueeze(0)
        routed_residual = residual if inputs.ndim > 1 else residual.unsqueeze(0)
        for expert_index, expert in enumerate(self.specialized_experts):
            sequence_indices, slots = torch.where(indices == expert_index)
            if sequence_indices.numel() == 0:
                continue
            expert_outputs = expert(routed_inputs[sequence_indices]).to(
                routed_residual.dtype
            )
            shape = (sequence_indices.shape[0],) + (1,) * (expert_outputs.ndim - 1)
            selected_weights = weights[sequence_indices, slots].reshape(shape)
            routed_residual[sequence_indices] += (
                selected_weights.to(routed_residual.dtype) * expert_outputs
            )
        return routed_residual if inputs.ndim > 1 else routed_residual.squeeze(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the generalized plus routed specialized residual."""
        generalized = self._generalized_residual(inputs).to(inputs.dtype)
        if self.config.routing_granularity == "token":
            specialized = self._token_routed_residual(inputs)
        else:
            specialized = self._sequence_routed_residual(inputs)
        return generalized + specialized.to(inputs.dtype)


class GSELinear(nn.Module):
    """Wrap a linear layer with a frozen base and grouped GSE adapter."""

    def __init__(self, base_layer: nn.Linear, config: GSEConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("GSELinear only supports torch.nn.Linear")
        config.validate_for_layer(base_layer.in_features)

        self.base_layer = base_layer
        self.config = config
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.adapter = GSEAdapter(
            self.in_features,
            self.out_features,
            config,
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
    def all_experts(self) -> tuple[GSEExpert, ...]:
        """Return generalized and specialized experts in rank-allocation order."""
        return self.adapter.all_experts

    @property
    def load_balancing_loss(self) -> torch.Tensor | None:
        """Return the load-balancing loss from the most recent forward pass."""
        return self.adapter.load_balancing_loss

    @property
    def router_stats(self) -> dict[str, torch.Tensor]:
        """Return detached routing diagnostics from the latest forward pass."""
        return self.adapter.router_stats

    def reset_auxiliary_state(self) -> None:
        """Discard losses and diagnostics saved by the latest forward pass."""
        self.adapter.reset_auxiliary_state()

    def orthogonality_loss(self) -> torch.Tensor:
        """Return the adapter A-factor orthogonality loss."""
        return self.adapter.orthogonality_loss()

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the frozen base layer plus a GSE residual update."""
        base_outputs = self.base_layer(inputs, *args, **kwargs)
        return base_outputs + self.adapter(inputs).to(base_outputs.dtype)
