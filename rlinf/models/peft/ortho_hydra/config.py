"""Configuration for Ortho-Hydra residual adapters."""

from dataclasses import dataclass
from typing import Literal

Parameterization = Literal["shared_a", "independent"]


@dataclass(frozen=True)
class OrthoHydraConfig:
    """Describe a total-rank-matched Ortho-Hydra adapter."""

    total_rank: int = 32
    lora_alpha: float = 32.0
    num_experts: int = 8
    parameterization: Parameterization = "shared_a"
    lora_dropout: float = 0.0
    router_bias: bool = True
    router_init_std: float = 0.01
    semantic_embedding_dim: int | None = None
    semantic_router_scale: float = 1.0
    freeze_base: bool = True
    init_seed: int | None = None
    record_routing_assignments: bool = False

    def __post_init__(self) -> None:
        """Validate model-independent fields."""
        if self.parameterization not in {"shared_a", "independent"}:
            raise ValueError(
                "parameterization must be 'shared_a' or 'independent', got "
                f"{self.parameterization!r}"
            )
        if self.total_rank <= 0:
            raise ValueError("total_rank must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.total_rank % self.num_experts != 0:
            raise ValueError("total_rank must be divisible by num_experts")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.router_init_std < 0:
            raise ValueError("router_init_std must be non-negative")
        if self.semantic_embedding_dim is None or self.semantic_embedding_dim <= 0:
            raise ValueError("semantic_embedding_dim must be positive")
        if self.semantic_router_scale < 0:
            raise ValueError("semantic_router_scale must be non-negative")

    @property
    def expert_rank(self) -> int:
        """Return the rank assigned to each expert."""
        return self.total_rank // self.num_experts

    @property
    def scaling(self) -> float:
        """Return the paper-style alpha divided by per-expert rank scaling."""
        return self.lora_alpha / self.expert_rank

    def validate_for_layer(self, in_features: int, out_features: int) -> None:
        """Validate the disjoint SVD construction for one linear layer."""
        max_rank = min(in_features, out_features)
        if self.total_rank > max_rank:
            raise ValueError(
                "Ortho-Hydra requires total_rank <= min(in_features, "
                f"out_features), got {self.total_rank} > {max_rank}"
            )
