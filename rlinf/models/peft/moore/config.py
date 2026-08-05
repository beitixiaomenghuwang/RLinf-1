"""Configuration for SVD-based MoORE residual adapters."""

from dataclasses import dataclass
from typing import Literal

RoutingGranularity = Literal["sequence", "token"]
SequencePooling = Literal["mean", "first", "last"]
Initialization = Literal["svd"]


@dataclass(frozen=True)
class MoOREConfig:
    """Describe a hidden-state-routed SVD residual adapter.

    The base linear weight is decomposed once at injection time. ``svd_U`` and
    ``svd_Vh`` and ``svd_S`` remain frozen buffers, while the hidden-state
    router and Householder vectors are trainable. ``svd_S`` is initialized from
    the base singular values, following the source MoORE implementation.
    """

    rank: int = 32
    num_experts: int = 8
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    routing_granularity: RoutingGranularity = "sequence"
    sequence_pooling: SequencePooling = "mean"
    initialization: Initialization = "svd"
    router_bias: bool = False
    freeze_base: bool = True
    init_seed: int | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.rank % 2:
            raise ValueError("MoORE rank must be a positive even number")
        if self.num_experts <= 0:
            raise ValueError("MoORE num_experts must be positive")
        if self.routing_granularity not in {"sequence", "token"}:
            raise ValueError("routing_granularity must be 'sequence' or 'token'")
        if self.sequence_pooling not in {"mean", "first", "last"}:
            raise ValueError("sequence_pooling must be 'mean', 'first', or 'last'")
        if self.initialization != "svd":
            raise ValueError("MoORE initialization must be 'svd'")
        if self.router_bias:
            raise ValueError("Source MoORE router initialization does not use a bias")
        if self.lora_dropout < 0 or self.lora_dropout >= 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")

    def validate_for_layer(self, in_features: int) -> None:
        """Validate the Householder width for a concrete linear layer."""
        if self.rank // 2 > in_features:
            raise ValueError(
                "MoORE Householder rank requires rank/2 <= in_features, got "
                f"{self.rank // 2} > {in_features}"
            )
