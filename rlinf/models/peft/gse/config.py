"""Configuration for generalized and specialized experts."""

from dataclasses import dataclass
from typing import Literal

RoutingGranularity = Literal["sequence", "token"]
SequencePooling = Literal["mean", "first", "last"]
Initialization = Literal["orthogonal_zero", "kaiming_zero", "svd"]
ScalingMode = Literal["total_rank", "expert_rank"]


@dataclass(frozen=True)
class GSEConfig:
    """Describe a parameter-matched GSE residual adapter.

    ``total_rank`` is divided across all generalized and specialized experts.
    It is not the rank of each expert.
    """

    total_rank: int = 64
    lora_alpha: float = 64.0
    num_experts: int = 8
    num_generalized_experts: int = 2
    top_k: int = 2
    lora_dropout: float = 0.0
    routing_granularity: RoutingGranularity = "sequence"
    sequence_pooling: SequencePooling = "mean"
    initialization: Initialization = "orthogonal_zero"
    scaling_mode: ScalingMode = "total_rank"
    normalize_topk: bool = True
    router_bias: bool = False
    router_init_std: float = 0.02
    orthogonal_gain: float = 1.0
    freeze_base: bool = True
    init_seed: int | None = None
    record_routing_assignments: bool = False

    def __post_init__(self) -> None:
        """Validate model-independent configuration fields."""
        valid_values = {
            "routing_granularity": ({"sequence", "token"}, self.routing_granularity),
            "sequence_pooling": ({"mean", "first", "last"}, self.sequence_pooling),
            "initialization": (
                {"orthogonal_zero", "kaiming_zero", "svd"},
                self.initialization,
            ),
            "scaling_mode": ({"total_rank", "expert_rank"}, self.scaling_mode),
        }
        for field_name, (choices, value) in valid_values.items():
            if value not in choices:
                raise ValueError(
                    f"{field_name} must be one of {sorted(choices)}, got {value!r}"
                )
        if self.total_rank <= 0:
            raise ValueError("total_rank must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 0 <= self.num_generalized_experts < self.num_experts:
            raise ValueError("num_generalized_experts must be in [0, num_experts)")
        if self.total_rank < self.num_experts:
            raise ValueError("total_rank must allocate at least rank 1 per expert")
        if not 1 <= self.top_k <= self.num_specialized_experts:
            raise ValueError("top_k must be in [1, num_specialized_experts]")
        if self.top_k == 1 and self.normalize_topk:
            raise ValueError(
                "normalize_topk must be False when top_k=1 so the task loss can "
                "train the router"
            )
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if self.router_init_std < 0:
            raise ValueError("router_init_std must be non-negative")
        if self.orthogonal_gain <= 0:
            raise ValueError("orthogonal_gain must be positive")

    @property
    def num_specialized_experts(self) -> int:
        """Return the number of routed experts."""
        return self.num_experts - self.num_generalized_experts

    @property
    def expert_ranks(self) -> tuple[int, ...]:
        """Split total rank deterministically and exactly across experts."""
        base_rank, remainder = divmod(self.total_rank, self.num_experts)
        return tuple(
            base_rank + int(index < remainder) for index in range(self.num_experts)
        )

    def scaling_for_rank(self, expert_rank: int) -> float:
        """Return the residual scaling for one expert."""
        denominator = (
            self.total_rank if self.scaling_mode == "total_rank" else expert_rank
        )
        return self.lora_alpha / denominator

    def validate_for_layer(
        self, in_features: int, out_features: int | None = None
    ) -> None:
        """Validate constraints that depend on the wrapped linear layer."""
        if self.initialization == "orthogonal_zero" and self.total_rank > in_features:
            raise ValueError(
                "orthogonal_zero requires total_rank <= in_features, got "
                f"{self.total_rank} > {in_features}"
            )
        if self.initialization == "svd" and out_features is not None:
            max_rank = min(in_features, out_features)
            if self.total_rank > max_rank:
                raise ValueError(
                    "svd initialization requires total_rank <= min(in_features, "
                    f"out_features), got {self.total_rank} > {max_rank}"
                )
