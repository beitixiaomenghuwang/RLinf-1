"""SVD and Householder MoORE linear layers."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoOREConfig


def _seeded_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def _initialize_householder(vectors: nn.Parameter, *, seed: int | None) -> None:
    """Use the source MoORE Kaiming initialization with paired reflections."""
    rank, in_features = vectors.shape
    generator = _seeded_generator(seed)
    half = torch.empty(in_features, rank // 2, dtype=torch.float32, device="cpu")
    # ``nn.init.kaiming_uniform_(tensor, a=sqrt(5))`` used by the original
    # implementation has bound 1/sqrt(fan_in), where fan_in is rank/2 here.
    bound = 1.0 / math.sqrt(rank // 2)
    half.uniform_(-bound, bound, generator=generator)
    values = torch.repeat_interleave(half.mT, 2, dim=0)
    vectors.copy_(values.to(device=vectors.device, dtype=vectors.dtype))


class MoOREAdapter(nn.Module):
    """Trainable MoORE residual, routed from hidden states only."""

    _is_gse_adapter = True  # Reuse RLinf's FSDP adapter wrapping policy.

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        config: MoOREConfig,
    ) -> None:
        super().__init__()
        config.validate_for_layer(in_features)
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.dim_s = min(in_features, out_features)
        device = base_weight.device
        dtype = base_weight.dtype

        with torch.no_grad():
            u, singular_values, vh = torch.linalg.svd(
                base_weight.detach().float(), full_matrices=False
            )
            # SVD vectors are sign-ambiguous. Canonicalize each right vector so
            # actor and rollout reconstruct identical non-persistent buffers.
            pivot = vh.abs().argmax(dim=1)
            signs = torch.sign(vh[torch.arange(vh.shape[0], device=vh.device), pivot])
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            u = u * signs.unsqueeze(0)
            vh = vh * signs.unsqueeze(1)
            u = u[:, : self.dim_s].transpose(0, 1).contiguous()
            vh = vh[: self.dim_s].transpose(0, 1).contiguous()
        self.register_buffer(
            "svd_U", u.to(device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "svd_Vh", vh.to(device=device, dtype=dtype), persistent=False
        )
        # The source MoORE keeps the base singular values frozen. Routing
        # deltas, rather than the SVD spectrum itself, are trainable.
        base_svd_s = singular_values[: self.dim_s].to(device=device, dtype=dtype)
        self.register_buffer("base_svd_S", base_svd_s.clone(), persistent=False)
        self.register_buffer("svd_S", base_svd_s.clone(), persistent=False)

        self.router = nn.Linear(
            in_features,
            config.num_experts,
            bias=config.router_bias,
            device=device,
            dtype=dtype,
        )
        router_generator = _seeded_generator(
            None if config.init_seed is None else config.init_seed + 1
        )
        with torch.no_grad():
            weight = torch.empty_like(
                self.router.weight, device="cpu", dtype=torch.float32
            )
            # Matches source MoORE's nn.init.kaiming_uniform_(weight, a=sqrt(5)).
            bound = 1.0 / math.sqrt(self.in_features)
            weight.uniform_(-bound, bound, generator=router_generator)
            self.router.weight.copy_(weight.to(device=device, dtype=dtype))
            if self.router.bias is not None:
                self.router.bias.zero_()
        self.router_up = nn.Linear(
            config.num_experts,
            self.dim_s,
            bias=False,
            device=device,
            dtype=dtype,
        )
        nn.init.zeros_(self.router_up.weight)

        self.householder = nn.Parameter(
            torch.empty(config.rank, in_features, device=device, dtype=dtype)
        )
        _initialize_householder(
            self.householder,
            seed=config.init_seed,
        )
        self.dropout = (
            nn.Dropout(config.lora_dropout) if config.lora_dropout else nn.Identity()
        )

    def _sequence_context(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim <= 2:
            return inputs.reshape(-1, self.in_features)
        sequences = inputs.reshape(inputs.shape[0], -1, self.in_features)
        if self.config.sequence_pooling == "first":
            return sequences[:, 0]
        if self.config.sequence_pooling == "last":
            return sequences[:, -1]
        return sequences.mean(dim=1)

    def _rotate(self, inputs: torch.Tensor) -> torch.Tensor:
        result = inputs.float()
        vectors = self.householder.float()
        for vector in vectors:
            vector = F.normalize(vector, dim=0).unsqueeze(1)
            result = result - 2 * (result @ vector) @ vector.transpose(0, 1)
        return result.to(inputs.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_dtype = inputs.dtype
        hidden = self.dropout(inputs).to(self.svd_Vh.dtype)
        reflected = self._rotate(hidden)
        coordinates = reflected @ self.svd_Vh

        if self.config.routing_granularity == "token":
            context = hidden.reshape(-1, self.in_features)
            logits = self.router(context)
            modulation = self.router_up(logits).reshape(*inputs.shape[:-1], self.dim_s)
        else:
            context = self._sequence_context(hidden)
            logits = self.router(context)
            sequence_modulation = self.router_up(logits)
            if inputs.ndim == 1:
                modulation = sequence_modulation.squeeze(0)
            elif inputs.ndim == 2:
                modulation = sequence_modulation
            else:
                modulation = sequence_modulation[:, None, :].expand(
                    *inputs.shape[:-1], self.dim_s
                )

        # The source MoORE keeps singular values non-negative. Hidden-state
        # Hidden-state routing replaces the source task-conditioned delta;
        # ``svd_S`` remains the frozen base SVD spectrum.
        singular_modulation = F.relu(modulation + self.svd_S)
        dynamic = (coordinates * singular_modulation) @ self.svd_U
        baseline = (hidden @ self.svd_Vh * self.base_svd_S) @ self.svd_U
        residual = dynamic - baseline
        return residual.to(input_dtype)


class MoORELinear(nn.Module):
    """Wrap a frozen linear layer with an SVD-initialized MoORE residual."""

    def __init__(self, base_layer: nn.Linear, config: MoOREConfig) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("MoORELinear only supports torch.nn.Linear")
        self.base_layer = base_layer
        self.config = config
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.adapter = MoOREAdapter(
            self.in_features, self.out_features, base_layer.weight, config
        )
        if config.freeze_base:
            self.base_layer.requires_grad_(False)

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    @property
    def router(self) -> nn.Linear:
        return self.adapter.router

    @property
    def svd_S(self) -> torch.Tensor:
        return self.adapter.svd_S

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        base_outputs = self.base_layer(inputs, *args, **kwargs)
        return base_outputs + self.adapter(inputs).to(base_outputs.dtype)
