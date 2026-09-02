"""Unit tests for generalized and specialized expert adapters."""

import math
from copy import deepcopy

import pytest
import torch
from torch import nn

from rlinf.models.peft.gse import (
    GSEConfig,
    GSELinear,
    gse_auxiliary_loss,
    gse_layerwise_task_router_metrics,
    gse_layerwise_task_router_statistics,
    gse_load_balancing_loss,
    gse_orthogonality_loss,
    gse_router_metrics,
    gse_routing_context,
    gse_state_dict,
    gse_task_router_metrics,
    gse_task_router_metrics_from_tensor,
    gse_task_router_statistics,
    inject_gse,
    joint_lora_a,
    load_gse_state_dict,
    mark_only_gse_as_trainable,
    orthogonality_error,
)


def make_config(**overrides: object) -> GSEConfig:
    """Create a small deterministic test configuration."""
    values = {
        "total_rank": 8,
        "lora_alpha": 8.0,
        "num_experts": 4,
        "num_generalized_experts": 1,
        "top_k": 2,
        "init_seed": 17,
    }
    values.update(overrides)
    return GSEConfig(**values)


def test_config_treats_rank_as_layer_total() -> None:
    config = make_config(total_rank=10)

    assert config.expert_ranks == (3, 3, 2, 2)
    assert sum(config.expert_ranks) == config.total_rank


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"total_rank": 3}, "at least rank 1"),
        ({"num_generalized_experts": 4}, "must be in"),
        ({"top_k": 4}, "top_k"),
        ({"top_k": 1}, "task loss can train the router"),
        (
            {"router_input": "rank_rms", "routing_granularity": "token"},
            "requires sequence routing",
        ),
    ],
)
def test_config_rejects_invalid_expert_allocations(
    overrides: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


def test_orthogonal_zero_initialization_preserves_base_output() -> None:
    torch.manual_seed(4)
    base_layer = nn.Linear(12, 7)
    original = deepcopy(base_layer)
    layer = GSELinear(base_layer, make_config())
    inputs = torch.randn(3, 5, 12)

    torch.testing.assert_close(layer(inputs), original(inputs))
    assert not layer.base_layer.weight.requires_grad
    assert orthogonality_error(layer.all_experts).item() < 1e-5

    joint_a = joint_lora_a(layer.all_experts)
    assert joint_a.shape == (layer.config.total_rank, layer.in_features)
    for expert in layer.all_experts:
        assert torch.count_nonzero(expert.lora_b.weight) == 0


def test_uniform_routing_averages_all_experts_without_router() -> None:
    layer = GSELinear(
        nn.Linear(12, 7),
        make_config(
            routing_mode="uniform",
            num_experts=4,
            num_generalized_experts=4,
            top_k=1,
        ),
    )
    with torch.no_grad():
        for expert in layer.all_experts:
            expert.lora_b.weight.normal_(std=0.01)
    inputs = torch.randn(3, 5, 12)

    output = layer(inputs) - layer.base_layer(inputs)
    expected = sum(expert(inputs) for expert in layer.all_experts) / 4

    torch.testing.assert_close(output, expected)
    assert not list(layer.router.parameters())


def test_only_specialized_experts_can_use_all_experts_for_topk() -> None:
    config = make_config(num_generalized_experts=0, num_experts=4, top_k=2)

    assert config.num_specialized_experts == 4


@pytest.mark.parametrize(
    ("routing_granularity", "routing_mode", "top_k", "total_rank"),
    [
        ("sequence", "topk", 2, 7),
        ("token", "topk", 2, 7),
        ("sequence", "all", 2, 4),
        ("token", "all", 2, 4),
    ],
)
def test_fused_expert_path_matches_sparse_path(
    routing_granularity: str,
    routing_mode: str,
    top_k: int,
    total_rank: int,
) -> None:
    torch.manual_seed(9)
    config = make_config(
        initialization="svd",
        total_rank=total_rank,
        num_experts=4,
        num_generalized_experts=0 if routing_mode == "all" else 1,
        top_k=top_k,
        routing_granularity=routing_granularity,
        routing_mode=routing_mode,
    )
    fused = GSELinear(nn.Linear(12, 7), config)
    sparse = deepcopy(fused)
    sparse.adapter._can_fuse_experts = lambda experts: False
    fused_inputs = torch.randn(3, 5, 12, requires_grad=True)
    sparse_inputs = fused_inputs.detach().clone().requires_grad_(True)

    fused_output = fused(fused_inputs)
    sparse_output = sparse(sparse_inputs)
    fused_output.square().mean().backward()
    sparse_output.square().mean().backward()

    torch.testing.assert_close(fused_output, sparse_output)
    torch.testing.assert_close(fused_inputs.grad, sparse_inputs.grad)
    for (fused_name, fused_parameter), (sparse_name, sparse_parameter) in zip(
        fused.named_parameters(), sparse.named_parameters(), strict=True
    ):
        assert fused_name == sparse_name
        fused_grad = (
            torch.zeros_like(fused_parameter)
            if fused_parameter.grad is None
            else fused_parameter.grad
        )
        sparse_grad = (
            torch.zeros_like(sparse_parameter)
            if sparse_parameter.grad is None
            else sparse_parameter.grad
        )
        torch.testing.assert_close(fused_grad, sparse_grad)


def test_all_routing_ignores_inherited_topk_and_selects_every_expert() -> None:
    layer = GSELinear(
        nn.Linear(12, 7),
        make_config(
            routing_mode="all",
            num_generalized_experts=0,
            top_k=2,
            record_routing_assignments=True,
        ),
    )

    layer(torch.randn(3, 5, 12))

    selected = layer.router_stats["selected_experts"]
    assert selected.shape == (3, 4)
    assert torch.equal(selected, torch.arange(4).expand(3, 4))


def test_rank_rms_router_uses_expert_a_projection_energy() -> None:
    layer = GSELinear(
        nn.Linear(12, 7),
        make_config(
            initialization="svd",
            total_rank=4,
            num_generalized_experts=0,
            routing_mode="all",
            router_input="rank_rms",
            record_routing_assignments=True,
        ),
    )
    inputs = torch.randn(3, 5, 12)

    layer(inputs)

    rank_hidden = torch.nn.functional.linear(inputs, joint_lora_a(layer.all_experts))
    rank_rms = rank_hidden.float().square().mean(dim=1).sqrt()
    expected = torch.softmax(layer.adapter.router(rank_rms).float(), dim=-1)
    assert layer.adapter.router.in_features == layer.config.total_rank
    torch.testing.assert_close(layer.router_stats["probabilities"], expected)

    layer.zero_grad(set_to_none=True)
    layer(inputs).square().mean().backward()
    assert torch.count_nonzero(layer.adapter.router.weight.grad) > 0


def test_dense_svd_uniform_router_preserves_output_and_trains_router() -> None:
    torch.manual_seed(9)
    base_layer = nn.Linear(12, 9, bias=False)
    original = deepcopy(base_layer)
    layer = GSELinear(
        base_layer,
        make_config(
            initialization="svd",
            total_rank=8,
            num_experts=8,
            num_generalized_experts=0,
            routing_mode="all",
            router_input="rank_rms",
            router_init_std=0.0,
        ),
    )
    inputs = torch.randn(3, 5, 12)

    outputs = layer(inputs)

    torch.testing.assert_close(outputs, original(inputs), rtol=1e-5, atol=1e-6)
    probabilities = layer.router_stats["mean_probability"]
    torch.testing.assert_close(
        probabilities,
        torch.full_like(probabilities, 1.0 / len(layer.specialized_experts)),
    )

    outputs.square().mean().backward()
    assert layer.router.weight.grad is not None
    assert torch.count_nonzero(layer.router.weight.grad) > 0


def test_svd_initialization_uses_balanced_factors_and_preserves_output() -> None:
    torch.manual_seed(8)
    base_layer = nn.Linear(12, 7, bias=False)
    original = deepcopy(base_layer)
    layer = GSELinear(
        base_layer,
        make_config(
            initialization="svd",
            total_rank=4,
            num_experts=2,
            num_generalized_experts=1,
            top_k=1,
            normalize_topk=False,
            scaling_mode="gse",
            svd_rho=10.0,
            preserve_svd_output=True,
        ),
    )
    left, singular_values, right = torch.linalg.svd(
        original.weight.detach().float(), full_matrices=False
    )
    inputs = torch.randn(2, 12)

    effective_weight = torch.zeros_like(original.weight, dtype=torch.float32)
    offset = 0
    for expert in layer.all_experts:
        rank = expert.lora_a.out_features
        expected_norms = singular_values[offset : offset + rank] / (
            expert.scaling.float() * layer.config.svd_rho
        )
        torch.testing.assert_close(
            expert.lora_a.weight.float().square().sum(dim=1), expected_norms
        )
        torch.testing.assert_close(
            expert.lora_b.weight.float().square().sum(dim=0), expected_norms
        )
        effective_weight += expert.scaling.float() * (
            expert.lora_b.weight.float() @ expert.lora_a.weight.float()
        )
        offset += rank
    expected_weight = (
        (left[:, :4] * singular_values[:4].unsqueeze(0)) @ right[:4]
    ) / layer.config.svd_rho
    torch.testing.assert_close(effective_weight, expected_weight)
    torch.testing.assert_close(layer(inputs), original(inputs), rtol=1e-5, atol=1e-6)


def test_exact_svd_initialization_trains_factors_and_router_from_step_zero() -> None:
    layer = GSELinear(
        nn.Linear(12, 7),
        make_config(
            initialization="svd",
            total_rank=6,
            num_experts=3,
            num_generalized_experts=1,
            top_k=2,
            lora_dropout=0.05,
            routing_granularity="token",
            scaling_mode="gse",
            svd_rho=10.0,
            preserve_svd_output=True,
        ),
    )
    inputs = torch.randn(3, 5, 12)

    layer(inputs).square().mean().backward()

    assert all(
        expert.lora_a.weight.grad is not None
        and torch.count_nonzero(expert.lora_a.weight.grad) > 0
        for expert in layer.all_experts
    )
    assert all(
        expert.lora_b.weight.grad is not None
        and torch.count_nonzero(expert.lora_b.weight.grad) > 0
        for expert in layer.all_experts
    )
    assert layer.router.weight.grad is not None
    assert torch.count_nonzero(layer.router.weight.grad) > 0


def test_kaiming_router_initialization_is_deterministic_and_nonzero() -> None:
    config = make_config(router_initialization="kaiming")

    first = GSELinear(nn.Linear(12, 7), config)
    second = GSELinear(nn.Linear(12, 7), config)

    torch.testing.assert_close(first.router.weight, second.router.weight)
    assert torch.count_nonzero(first.router.weight) > 0


def test_sequence_router_makes_one_decision_per_batch_item() -> None:
    inputs = torch.randn(3, 5, 12)
    sequence_layer = GSELinear(
        nn.Linear(12, 7), make_config(routing_granularity="sequence")
    )
    token_layer = GSELinear(nn.Linear(12, 7), make_config(routing_granularity="token"))

    sequence_layer(inputs)
    token_layer(inputs)

    assert sequence_layer.router_stats["num_routing_items"].item() == 3
    assert token_layer.router_stats["num_routing_items"].item() == 15
    assert sequence_layer.router_stats["selection_fraction"].sum() == pytest.approx(1.0)


def test_zero_b_initialization_has_expected_two_stage_gradient_flow() -> None:
    layer = GSELinear(nn.Linear(12, 7), make_config())
    inputs = torch.randn(3, 5, 12)

    layer(inputs).sum().backward()

    assert any(
        torch.count_nonzero(expert.lora_b.weight.grad) > 0
        for expert in layer.all_experts
    )
    assert all(
        torch.count_nonzero(expert.lora_a.weight.grad) == 0
        for expert in layer.all_experts
    )
    assert torch.count_nonzero(layer.router.weight.grad) == 0

    layer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for expert in layer.all_experts:
            expert.lora_b.weight.normal_(std=0.01)
    layer(inputs).square().mean().backward()

    assert any(
        torch.count_nonzero(expert.lora_a.weight.grad) > 0
        for expert in layer.all_experts
        if expert.lora_a.weight.grad is not None
    )
    assert torch.count_nonzero(layer.router.weight.grad) > 0


class ToyModel(nn.Module):
    """Small model with separable VLM and action-expert subtrees."""

    def __init__(self) -> None:
        super().__init__()
        self.vlm = nn.Sequential(nn.Linear(12, 12), nn.ReLU())
        self.action_expert = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 7),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the action path only."""
        return self.action_expert(inputs)


def test_injection_freezing_and_adapter_only_state_round_trip() -> None:
    model = ToyModel()
    report = inject_gse(
        model,
        make_config(),
        target_modules=("action_expert.0",),
    )
    mark_only_gse_as_trainable(model)

    assert report.injected_module_names == ("action_expert.0",)
    assert isinstance(model.action_expert[0], GSELinear)
    assert isinstance(model.vlm[0], nn.Linear)
    assert not model.vlm[0].weight.requires_grad
    assert not model.action_expert[0].base_layer.weight.requires_grad
    assert model.action_expert[0].router.weight.requires_grad

    saved_state = gse_state_dict(model)
    assert saved_state
    assert all("base_layer" not in name for name in saved_state)

    restored = ToyModel()
    inject_gse(
        restored,
        make_config(init_seed=999),
        target_modules=("action_expert.0",),
    )
    load_gse_state_dict(restored, saved_state)
    for name, value in gse_state_dict(restored).items():
        torch.testing.assert_close(value, saved_state[name])


def test_auxiliary_losses_are_finite_and_differentiable() -> None:
    model = ToyModel()
    inject_gse(model, make_config(), target_modules=("action_expert.0",))
    model(torch.randn(3, 5, 12))

    load_balance = gse_load_balancing_loss(model)
    orthogonality = gse_orthogonality_loss(model)

    assert torch.isfinite(load_balance)
    assert torch.isfinite(orthogonality)
    assert load_balance.requires_grad
    assert orthogonality.requires_grad


def test_load_balancing_loss_is_exposed_as_adapter_forward_output() -> None:
    layer = GSELinear(nn.Linear(12, 7), make_config())
    adapter_outputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    hook = layer.adapter.register_forward_hook(
        lambda _module, _inputs, output: adapter_outputs.append(output)
    )

    layer(torch.randn(3, 5, 12))
    hook.remove()

    assert len(adapter_outputs) == 1
    residual, load_balancing_loss = adapter_outputs[0]
    assert residual.shape == (3, 5, 7)
    assert load_balancing_loss is layer.load_balancing_loss
    assert load_balancing_loss is not None
    assert load_balancing_loss.requires_grad


def test_topk_load_balancing_loss_matches_official_selection_scale() -> None:
    layer = GSELinear(
        nn.Linear(12, 7),
        make_config(
            num_experts=4,
            num_generalized_experts=0,
            top_k=2,
            router_init_std=0.0,
        ),
    )

    layer(torch.randn(3, 5, 12))

    torch.testing.assert_close(layer.load_balancing_loss, torch.tensor(2.0))
    torch.testing.assert_close(
        layer.router_stats["selection_fraction"].sum(), torch.tensor(1.0)
    )


def test_router_metrics_aggregate_expert_utilization() -> None:
    model = ToyModel()
    inject_gse(
        model,
        make_config(),
        target_modules=("action_expert.0", "action_expert.2"),
    )
    model(torch.randn(3, 5, 12))

    metrics = gse_router_metrics(model)

    assert metrics["gse/router/active_layers"].item() == 2
    assert 0 <= metrics["gse/router/normalized_entropy"].item() <= 1
    expert_selection = torch.stack(
        [
            metrics[f"gse/router/expert_{index}_selection"]
            for index in range(make_config().num_specialized_experts)
        ]
    )
    expert_probability = torch.stack(
        [
            metrics[f"gse/router/expert_{index}_probability"]
            for index in range(make_config().num_specialized_experts)
        ]
    )
    torch.testing.assert_close(expert_selection.sum(), torch.tensor(1.0))
    torch.testing.assert_close(expert_probability.sum(), torch.tensor(1.0))


def test_router_metrics_separate_incompatible_adapter_domains() -> None:
    model = ToyModel()
    inject_gse(
        model.action_expert,
        make_config(record_routing_assignments=True),
        target_modules=("0",),
    )
    inject_gse(
        model.vlm,
        make_config(
            total_rank=4,
            num_experts=2,
            num_generalized_experts=1,
            top_k=1,
            normalize_topk=False,
        ),
        target_modules=("0",),
    )
    model.action_expert[0].gse_domain = "action"
    model.vlm[0].gse_domain = "vlm"
    inputs = torch.randn(3, 5, 12)
    model.action_expert[0](inputs)
    model.vlm[0](inputs)

    metrics = gse_router_metrics(model)

    assert metrics["gse/router/active_layers"].item() == 1
    assert metrics["gse/action_router/active_layers"].item() == 1
    assert metrics["gse/vlm_router/active_layers"].item() == 1
    assert "gse/vlm_router/expert_0_probability" in metrics
    assert "gse/vlm_router/expert_1_probability" not in metrics

    statistics = gse_task_router_statistics(
        model,
        torch.tensor([0, 1, 1]),
        num_tasks=3,
        domain="action",
    )
    assert statistics["gse/task_router_stats/task_00/routing_count"].item() == 1


def test_task_router_statistics_preserve_task_counts() -> None:
    model = ToyModel()
    config = make_config(record_routing_assignments=True)
    inject_gse(
        model,
        config,
        target_modules=("action_expert.0", "action_expert.2"),
    )
    model(torch.randn(3, 5, 12))

    statistics = gse_task_router_statistics(
        model,
        torch.tensor([0, 1, 1]),
        num_tasks=3,
    )

    prefix = "gse/task_router_stats"
    assert statistics[f"{prefix}/task_00/routing_count"].item() == 2
    assert statistics[f"{prefix}/task_01/routing_count"].item() == 4
    assert statistics[f"{prefix}/task_02/routing_count"].item() == 0
    assert statistics[f"{prefix}/task_00/selection_total"].item() == 4
    assert statistics[f"{prefix}/task_01/selection_total"].item() == 8
    num_specialized = config.num_specialized_experts
    for task_index in (0, 1):
        probability_sum = sum(
            statistics[
                f"{prefix}/task_{task_index:02d}/expert_{expert_index}_probability_sum"
            ]
            for expert_index in range(num_specialized)
        )
        selection_count = sum(
            statistics[
                f"{prefix}/task_{task_index:02d}/expert_{expert_index}_selection_count"
            ]
            for expert_index in range(num_specialized)
        )
        torch.testing.assert_close(
            probability_sum,
            statistics[f"{prefix}/task_{task_index:02d}/routing_count"],
        )
        torch.testing.assert_close(
            selection_count,
            statistics[f"{prefix}/task_{task_index:02d}/selection_total"],
        )


def test_layerwise_task_router_statistics_are_compact_and_count_preserving() -> None:
    model = ToyModel()
    config = make_config(record_routing_assignments=True)
    inject_gse(
        model,
        config,
        target_modules=("action_expert.0", "action_expert.2"),
    )
    model(torch.randn(3, 5, 12))

    statistics = gse_layerwise_task_router_statistics(
        model, torch.tensor([0, 1, 1]), num_tasks=3
    )

    assert statistics.shape == (2, 3, 2 + 2 * config.num_specialized_experts)
    torch.testing.assert_close(statistics[:, :, 0].sum(dim=1), torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        statistics[:, :, 1].sum(dim=1),
        torch.tensor([6.0, 6.0]),
    )
    torch.testing.assert_close(
        statistics[:, :, 2 + config.num_specialized_experts :].sum(dim=(1, 2)),
        statistics[:, :, 1].sum(dim=1),
    )


def test_layerwise_metrics_detect_information_hidden_in_one_layer() -> None:
    # Layout: routing count, selection total, two probability sums, and two
    # selection counts. Layer 0 is task-specialized while layer 1 is uniform.
    statistics = torch.tensor(
        [
            [[10, 10, 10, 0, 10, 0], [10, 10, 0, 10, 0, 10]],
            [[10, 10, 5, 5, 5, 5], [10, 10, 5, 5, 5, 5]],
        ],
        dtype=torch.float32,
    )

    aggregate = gse_task_router_metrics_from_tensor(statistics)
    layerwise = gse_layerwise_task_router_metrics(
        statistics, informative_nmi_threshold=0.5
    )

    assert aggregate["gse/task_router/prob_nmi"] > 0
    assert layerwise["gse/task_router/layerwise_nmi_max"] == pytest.approx(1.0)
    assert layerwise["gse/task_router/layerwise_nmi_top_layer"] == 0
    assert layerwise[
        "gse/task_router/layerwise_adjusted_cramers_v_max"
    ] == pytest.approx(1.0)
    assert layerwise["gse/task_router/layerwise_informative_fraction"] == pytest.approx(
        0.5
    )


def test_task_router_metrics_detect_complete_task_specialization() -> None:
    prefix = "gse/task_router_stats"
    statistics = {
        f"{prefix}/task_00/routing_count": 10.0,
        f"{prefix}/task_00/selection_total": 10.0,
        f"{prefix}/task_00/expert_0_probability_sum": 10.0,
        f"{prefix}/task_00/expert_1_probability_sum": 0.0,
        f"{prefix}/task_00/expert_0_selection_count": 10.0,
        f"{prefix}/task_00/expert_1_selection_count": 0.0,
        f"{prefix}/task_01/routing_count": 10.0,
        f"{prefix}/task_01/selection_total": 10.0,
        f"{prefix}/task_01/expert_0_probability_sum": 0.0,
        f"{prefix}/task_01/expert_1_probability_sum": 10.0,
        f"{prefix}/task_01/expert_0_selection_count": 0.0,
        f"{prefix}/task_01/expert_1_selection_count": 10.0,
    }

    metrics = gse_task_router_metrics(statistics, num_tasks=2, num_experts=2)

    assert metrics["gse/task_router/covered_tasks"] == 2
    assert metrics["gse/task_router/normalized_mutual_information"] == pytest.approx(
        1.0
    )
    assert metrics["gse/task_router/mean_js_divergence"] > 0
    assert (
        metrics["gse/task_router/nmi"]
        == metrics["gse/task_router/normalized_mutual_information"]
    )
    assert (
        metrics["gse/task_router/js"] == metrics["gse/task_router/mean_js_divergence"]
    )
    assert (
        metrics["gse/task_router/prob_std"]
        == metrics["gse/task_router/mean_probability_std_across_tasks"]
    )
    assert (
        metrics["gse/task_router/select_std"]
        == metrics["gse/task_router/mean_selection_std_across_tasks"]
    )
    assert metrics["gse/task_router/task_00/dominant_expert"] == 0
    assert metrics["gse/task_router/task_01/dominant_expert"] == 1


def test_configurable_auxiliary_loss_preserves_zero_coefficient_objective() -> None:
    model = ToyModel()
    inject_gse(model, make_config(), target_modules=("action_expert.0",))
    model(torch.randn(3, 5, 12))

    disabled_loss, disabled_metrics = gse_auxiliary_loss(model)
    enabled_loss, enabled_metrics = gse_auxiliary_loss(
        model,
        load_balancing_coefficient=0.01,
        orthogonality_coefficient=0.1,
    )

    torch.testing.assert_close(disabled_loss, torch.tensor(0.0))
    assert enabled_loss.requires_grad
    assert enabled_loss.item() >= 0
    assert disabled_metrics["gse/auxiliary_loss"].item() == 0
    assert enabled_metrics["gse/weighted_load_balancing_loss"].item() > 0


def test_auxiliary_loss_rejects_negative_coefficients() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        gse_auxiliary_loss(ToyModel(), load_balancing_coefficient=-0.1)


def test_injection_is_strict_when_no_linear_module_matches() -> None:
    with pytest.raises(ValueError, match="No linear modules matched"):
        inject_gse(ToyModel(), make_config(), target_modules=("missing",))


def _semantic_layer(
    embedding_dim: int, scale: float, router_input: str = "hidden"
) -> GSELinear:
    """Build a sequence-routed layer with joint hidden/semantic routing.

    Uses ``svd`` initialization because ``orthogonal_zero`` leaves every
    ``lora_b`` at zero, which makes the whole expert residual -- and therefore
    every router gradient -- identically zero.
    """
    base = nn.Linear(embedding_dim, embedding_dim, bias=False)
    return GSELinear(
        base,
        make_config(
            routing_mode="topk",
            routing_granularity="sequence",
            sequence_pooling="mean",
            router_input=router_input,
            router_initialization="normal",
            router_init_std=0.02,
            initialization="svd",
            scaling_mode="gse",
            svd_rho=10.0,
            preserve_svd_output=True,
            semantic_conditioning=True,
            semantic_embedding_dim=embedding_dim,
            semantic_router_scale=scale,
            record_routing_assignments=True,
        ),
    )


def test_router_init_std_is_ignored_unless_initialization_is_normal() -> None:
    """`router_init_std` only reaches `normal`; the others discard it."""
    kaiming = GSELinear(
        nn.Linear(16, 16, bias=False),
        make_config(router_initialization="kaiming", router_init_std=1e-8),
    )
    default = GSELinear(
        nn.Linear(16, 16, bias=False),
        make_config(router_initialization="default", router_init_std=1e-8),
    )
    normal = GSELinear(
        nn.Linear(16, 16, bias=False),
        make_config(router_initialization="normal", router_init_std=1e-8),
    )
    assert normal.adapter.router.weight.std().item() < 1e-6
    assert kaiming.adapter.router.weight.std().item() > 1e-3
    assert default.adapter.router.weight.std().item() > 1e-3


def test_default_router_initialization_scales_with_in_features() -> None:
    """`default` reproduces the official repo: nn.Linear's own reset_parameters.

    The official GSE never initializes its router, so it keeps
    ``kaiming_uniform_(a=sqrt(5))`` -- ``uniform(+/-1/sqrt(in_features))`` with
    standard deviation ``1/sqrt(3*in_features)``. The scale must therefore track
    the router width instead of staying at a fixed ``router_init_std``.
    """
    for width in (32, 256, 2048):
        layer = GSELinear(
            nn.Linear(width, width, bias=False),
            make_config(router_initialization="default"),
        )
        weight = layer.adapter.router.weight
        bound = 1.0 / math.sqrt(width)
        assert weight.abs().max().item() <= bound + 1e-6
        assert weight.std().item() == pytest.approx(
            1.0 / math.sqrt(3 * width), rel=0.15
        )

    narrow = GSELinear(
        nn.Linear(32, 32, bias=False), make_config(router_initialization="default")
    )
    wide = GSELinear(
        nn.Linear(2048, 2048, bias=False), make_config(router_initialization="default")
    )
    assert narrow.adapter.router.weight.std() > 4 * wide.adapter.router.weight.std()


def test_semantic_conditioning_requires_an_active_routing_context() -> None:
    layer = _semantic_layer(32, 1.0)
    with pytest.raises(RuntimeError, match="requires an active GSE routing context"):
        layer(torch.randn(3, 4, 32))


def test_semantic_conditioning_routes_by_task_when_hidden_states_match() -> None:
    """Identical hidden states plus distinct prompts must still differentiate.

    This is the pi0.5 case: the action expert's suffix carries only action
    tokens, so mean-pooled hidden states are nearly task-invariant and the
    hidden-only router cannot separate tasks.
    """
    torch.manual_seed(0)
    dim, num_tasks = 32, 12
    prompts = torch.randn(num_tasks, dim)
    shared_hidden = torch.randn(1, 4, dim).expand(num_tasks, 4, dim).contiguous()

    def route(scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        layer = _semantic_layer(dim, scale)
        with gse_routing_context(prompts):
            layer(shared_hidden)
        stats = layer.adapter.router_stats
        return stats["probabilities"], stats["selected_experts"]

    hidden_probabilities, hidden_choices = route(0.0)
    joint_probabilities, joint_choices = route(1.0)

    # Task-invariant hidden states give every task an identical distribution.
    assert torch.allclose(
        hidden_probabilities, hidden_probabilities[0].expand_as(hidden_probabilities)
    )
    assert len({tuple(row.tolist()) for row in hidden_choices}) == 1

    # The semantic branch makes the distribution, and the ranking, task-specific.
    assert not torch.allclose(
        joint_probabilities, joint_probabilities[0].expand_as(joint_probabilities)
    )
    assert len({tuple(row.tolist()) for row in joint_choices}) > 1


def test_semantic_router_receives_gradient() -> None:
    torch.manual_seed(0)
    layer = _semantic_layer(32, 1.0)
    with gse_routing_context(torch.randn(3, 32)):
        layer(torch.randn(3, 4, 32)).square().mean().backward()
    gradient = layer.adapter.semantic_router.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm().item() > 0.0


def test_semantic_logits_expand_to_token_routing_batches() -> None:
    """Token routing has more router items than sequences; they must align."""
    torch.manual_seed(0)
    dim, batch, tokens = 32, 3, 4
    layer = GSELinear(
        nn.Linear(dim, dim, bias=False),
        make_config(
            routing_mode="topk",
            routing_granularity="token",
            router_initialization="normal",
            semantic_conditioning=True,
            semantic_embedding_dim=dim,
            record_routing_assignments=True,
        ),
    )
    with gse_routing_context(torch.randn(batch, dim)):
        output = layer(torch.randn(batch, tokens, dim))
    assert output.shape == (batch, tokens, dim)
    probabilities = layer.adapter.router_stats["probabilities"]
    assert probabilities.shape[0] == batch * tokens


def test_nonzero_lora_dropout_disables_expert_fusion() -> None:
    """Fusion is a throughput property; dropout silently switches the kernel."""
    fused = GSELinear(nn.Linear(16, 16, bias=False), make_config(lora_dropout=0.0))
    looped = GSELinear(nn.Linear(16, 16, bias=False), make_config(lora_dropout=0.05))
    assert fused.adapter._can_fuse_experts(fused.adapter.all_experts)
    assert not looped.adapter._can_fuse_experts(looped.adapter.all_experts)


def test_router_parameters_are_separable_for_their_own_optimizer_group() -> None:
    """A dedicated router lr needs the name filter to catch every router.

    ``gse_router_lr`` splits routers into their own AdamW group via
    ``_is_router_parameter``. If that filter missed a router, the router would
    silently keep the base lr; if it caught an expert factor, that factor would
    jump to the router lr. Both are invisible at runtime, so pin the split.
    """
    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import _is_router_parameter

    layers = 3
    config = make_config(
        routing_mode="topk",
        routing_granularity="sequence",
        router_input="hidden",
        router_initialization="default",
        initialization="svd",
        scaling_mode="gse",
        svd_rho=10.0,
        preserve_svd_output=True,
        semantic_conditioning=True,
        semantic_embedding_dim=16,
    )
    model = nn.Sequential(
        *[GSELinear(nn.Linear(16, 16, bias=False), config) for _ in range(layers)]
    )

    names = [name for name, _ in model.named_parameters()]
    routers = [name for name in names if _is_router_parameter(name)]
    others = [name for name in names if not _is_router_parameter(name)]

    # One router plus one semantic router per layer, and nothing else.
    assert len(routers) == 2 * layers
    assert all(name.endswith(("router.weight", "router.bias")) for name in routers)
    assert not any("router" in name for name in others)
    assert any("lora_a" in name for name in others)
    assert any("lora_b" in name for name in others)

    # The split must actually produce the intended lr ratio through AdamW.
    router_parameters = [
        p for n, p in model.named_parameters() if _is_router_parameter(n)
    ]
    expert_parameters = [
        p for n, p in model.named_parameters() if not _is_router_parameter(n)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": expert_parameters, "lr": 5e-6},
            {"params": router_parameters, "lr": 1e-3, "weight_decay": 0.5},
        ],
        weight_decay=0.01,
    )
    # A per-group weight_decay must override the constructor default.
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.01, 0.5]

    before = [p.detach().clone() for p in router_parameters + expert_parameters]
    with gse_routing_context(torch.randn(4, 16)):
        model(torch.randn(4, 3, 16)).sum().backward()
    optimizer.step()
    moved = [
        (p.detach() - b).abs().max().item()
        for p, b in zip(router_parameters + expert_parameters, before)
    ]
    router_moves = moved[: len(router_parameters)]
    expert_moves = moved[len(router_parameters) :]
    assert max(router_moves) > 10 * max(expert_moves)


def test_gse_router_lr_fails_loudly_when_no_router_is_visible() -> None:
    """``gse_router_lr`` is name-based, so an invisible router must not pass.

    Under FSDP1 with ``use_orig_params=False`` a ``GSEAdapter`` collapses into
    one ``adapter._fsdp_wrapped_module._flat_param``, which carries no router
    name. If that happens the routers quietly fall back to ``optim.lr`` and the
    experiment is silently voided, so the empty group has to raise.
    """
    from omegaconf import OmegaConf

    from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
        FSDPModelManager,
        _is_router_parameter,
    )

    # A flattened adapter exposes no router name at all.
    assert not _is_router_parameter("layers.0.adapter._fsdp_wrapped_module._flat_param")
    # A separately wrapped router keeps its name even when flattened.
    assert _is_router_parameter(
        "layers.0.adapter._fsdp_wrapped_module.router._fsdp_wrapped_module._flat_param"
    )
    assert _is_router_parameter(
        "layers.0.adapter._fsdp_wrapped_module.semantic_router."
        "_fsdp_wrapped_module._flat_param"
    )

    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create(
        {
            "optim": {
                "lr": 5e-6,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "gse_router_lr": 1e-3,
                "gse_router_weight_decay": 0.01,
            },
            "fsdp_config": {"use_orig_params": False},
        }
    )
    manager.store_requires_grad_param_name = []

    with pytest.raises(ValueError, match="no router parameter was found"):
        manager.build_optimizer(model=nn.Linear(4, 4))


def test_adapter_routers_are_marked_for_their_own_fsdp_unit() -> None:
    """Both adapter families must tag routers so the wrap policy finds them.

    The ``_is_adapter_router`` marker is what keeps ``.router.`` /
    ``.semantic_router.`` in the parameter path under ``use_orig_params=False``.
    Without it the router optimizer group comes out empty.
    """
    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.models.peft.ortho_hydra import OrthoHydraConfig, OrthoHydraLinear

    gse_layer = GSELinear(
        nn.Linear(16, 16),
        GSEConfig(
            total_rank=8,
            num_experts=4,
            num_generalized_experts=1,
            semantic_conditioning=True,
            semantic_embedding_dim=8,
            init_seed=0,
        ),
    )
    ortho_layer = OrthoHydraLinear(
        nn.Linear(16, 16),
        OrthoHydraConfig(
            total_rank=8,
            num_experts=4,
            semantic_embedding_dim=8,
            init_seed=0,
        ),
    )

    for layer in (gse_layer, ortho_layer):
        marked = {
            name
            for name, module in layer.named_modules()
            if getattr(module, "_is_adapter_router", False)
        }
        assert any(name.endswith("router") for name in marked), (
            f"{type(layer).__name__} marked no hidden router: {marked}"
        )
        assert any(name.endswith("semantic_router") for name in marked), (
            f"{type(layer).__name__} marked no semantic router: {marked}"
        )
        # The wrap policy must accept the marked routers as their own units.
        policy = get_fsdp_wrap_policy(layer, model_type="openpi")
        assert policy is not None
        for name, module in layer.named_modules():
            if getattr(module, "_is_adapter_router", False):
                assert policy(
                    module=module, recurse=False, nonwrapped_numel=module.weight.numel()
                ), f"policy refused to wrap {name}"


def test_router_casts_its_input_dtype_inside_its_own_forward() -> None:
    """The cast must live in the router, not the caller.

    Once the router is its own FSDP unit, reading ``router.weight.dtype`` from
    the caller happens before FSDP unshards and casts the parameter, so under
    mixed precision the caller sees fp32 while the matmul runs in bf16. Routing
    a mismatched input through the router directly has to still work.
    """
    layer = GSELinear(
        nn.Linear(16, 16),
        GSEConfig(total_rank=8, num_experts=4, num_generalized_experts=1, init_seed=0),
    )
    router = layer.adapter.router
    router.to(torch.float64)

    logits = router(torch.randn(2, 16, dtype=torch.float32))
    assert logits.dtype == torch.float64


def test_rank_rms_pooling_matches_ortho_hydra_reduction() -> None:
    """GSE's `rank_rms` context must be Ortho-Hydra's pooling, not mean pooling.

    Ortho-Hydra reduces a sequence with ``sqrt(mean(x^2))`` over the
    concatenated rank projections. Mean pooling cancels opposite-signed token
    activations, so the two disagree on exactly the signal that matters.
    """
    torch.manual_seed(0)
    dim = 32
    layer = _semantic_layer(dim, 1.0, router_input="rank_rms")
    adapter = layer.adapter
    inputs = torch.randn(3, 5, dim)

    rank_hidden = adapter._rank_space_hidden(inputs)
    context = adapter._rank_space_rms_context(rank_hidden)

    sequences = rank_hidden.float().reshape(
        rank_hidden.shape[0], -1, adapter.config.total_rank
    )
    expected = sequences.square().mean(dim=1).sqrt()
    assert torch.allclose(context, expected)

    # Router width collapses to total_rank, and RMS output is non-negative.
    assert adapter.router.in_features == adapter.config.total_rank
    assert bool((context >= 0).all())
    # A sign-flipped sequence pools identically under RMS but not under mean.
    flipped = adapter._rank_space_rms_context(
        rank_hidden * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]).reshape(1, -1, 1)
    )
    assert torch.allclose(context, flipped)


def test_rank_rms_routing_still_honours_semantic_conditioning() -> None:
    """The production main-method combination: rank_rms plus frozen prompts.

    ``rank_rms`` feeds the router a task-invariant magnitude summary, so the
    semantic branch must remain the thing that separates tasks, expert fusion
    must stay enabled, and SVD output preservation must survive.
    """
    torch.manual_seed(0)
    dim, num_tasks = 32, 12
    prompts = torch.randn(num_tasks, dim)
    shared_hidden = torch.randn(1, 4, dim).expand(num_tasks, 4, dim).contiguous()

    def route(scale: float) -> torch.Tensor:
        layer = _semantic_layer(dim, scale, router_input="rank_rms")
        assert layer.adapter._can_fuse_experts(layer.adapter.all_experts)
        with gse_routing_context(prompts):
            layer(shared_hidden)
        return layer.adapter.router_stats["probabilities"]

    hidden_only = route(0.0)
    joint = route(1.0)

    assert torch.allclose(hidden_only, hidden_only[0].expand_as(hidden_only))
    assert not torch.allclose(joint, joint[0].expand_as(joint))
    assert joint.std(dim=0).max().item() > 1e-3

    base = nn.Linear(dim, dim, bias=False)
    layer = GSELinear(
        base,
        make_config(
            routing_mode="topk",
            routing_granularity="sequence",
            router_input="rank_rms",
            router_initialization="normal",
            initialization="svd",
            scaling_mode="gse",
            svd_rho=10.0,
            preserve_svd_output=True,
            semantic_conditioning=True,
            semantic_embedding_dim=dim,
        ),
    )
    inputs = torch.randn(num_tasks, 4, dim)
    with torch.no_grad(), gse_routing_context(prompts):
        assert torch.allclose(layer(inputs), base(inputs), atol=1e-5)
