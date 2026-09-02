"""Configuration for generalized and specialized experts."""

from dataclasses import dataclass
from typing import Literal

RoutingGranularity = Literal["sequence", "token"]
SequencePooling = Literal["mean", "first", "last"]
Initialization = Literal["orthogonal_zero", "kaiming_zero", "svd"]
ScalingMode = Literal["total_rank", "expert_rank", "gse"]
RoutingMode = Literal["topk", "all", "uniform"]
RouterInput = Literal["hidden", "rank_rms"]
RouterInitialization = Literal["default", "normal", "kaiming"]


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
    routing_mode: RoutingMode = "topk"
    router_input: RouterInput = "hidden"
    initialization: Initialization = "orthogonal_zero"
    scaling_mode: ScalingMode = "total_rank"
    gse_eta: float = 1.0
    svd_rho: float = 1.0
    preserve_svd_output: bool = False
    normalize_topk: bool = True
    router_bias: bool = False
    # "default" reproduces the official GSE repo, which leaves the router at
    # nn.Linear's own reset_parameters: kaiming_uniform_(a=sqrt(5)), i.e.
    # uniform(+/-1/sqrt(in_features)) with std 1/sqrt(3*in_features). The scale
    # therefore tracks in_features automatically and router_init_std is unused.
    # "normal" uses router_init_std verbatim at every width.
    router_initialization: RouterInitialization = "normal"
    router_init_std: float = 0.02
    semantic_conditioning: bool = False
    semantic_embedding_dim: int | None = None
    semantic_router_scale: float = 1.0
    action_sequence_routing: bool = False
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
            "scaling_mode": (
                {"total_rank", "expert_rank", "gse"},
                self.scaling_mode,
            ),
            "routing_mode": ({"topk", "all", "uniform"}, self.routing_mode),
            "router_input": ({"hidden", "rank_rms"}, self.router_input),
            "router_initialization": (
                {"default", "normal", "kaiming"},
                self.router_initialization,
            ),
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
        if not 0 <= self.num_generalized_experts <= self.num_experts or (
            self.routing_mode != "uniform"
            and self.num_generalized_experts == self.num_experts
        ):
            raise ValueError("num_generalized_experts must be in [0, num_experts)")
        if self.total_rank < self.num_experts:
            raise ValueError("total_rank must allocate at least rank 1 per expert")
        if self.routing_mode == "topk" and not (
            1 <= self.top_k <= self.num_specialized_experts
        ):
            raise ValueError("top_k must be in [1, num_specialized_experts]")
        if self.routing_mode == "topk" and self.top_k == 1 and self.normalize_topk:
            raise ValueError(
                "normalize_topk must be False when top_k=1 so the task loss can "
                "train the router"
            )
        if self.router_input == "rank_rms" and self.routing_granularity != "sequence":
            raise ValueError("rank_rms router_input requires sequence routing")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if self.gse_eta <= 0:
            raise ValueError("gse_eta must be positive")
        if self.svd_rho <= 0:
            raise ValueError("svd_rho must be positive")
        if self.router_init_std < 0:
            raise ValueError("router_init_std must be non-negative")
        if self.semantic_conditioning and (
            self.semantic_embedding_dim is None or self.semantic_embedding_dim <= 0
        ):
            raise ValueError(
                "semantic_embedding_dim must be positive when semantic_conditioning=True"
            )
        if self.semantic_router_scale < 0:
            raise ValueError("semantic_router_scale must be non-negative")
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

    def scaling_for_rank(self, expert_rank: int, in_features: int) -> float:
        """Return the residual scaling for one expert."""
        if self.scaling_mode == "gse":
            return (3.0 * self.gse_eta * in_features / expert_rank) ** 0.5
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
