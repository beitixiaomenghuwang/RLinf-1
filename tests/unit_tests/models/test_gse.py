"""Unit tests for generalized and specialized expert adapters."""

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


def test_svd_initialization_uses_full_factors_and_preserves_output() -> None:
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
        ),
    )
    left, singular_values, right = torch.linalg.svd(
        base_layer.weight.detach().float(), full_matrices=False
    )
    joint_a = joint_lora_a(layer.all_experts).float()
    joint_b = torch.cat(
        [expert.lora_b.weight for expert in layer.all_experts], dim=1
    ).float()
    inputs = torch.randn(2, 12)

    identity = torch.eye(4)
    torch.testing.assert_close(joint_a @ joint_a.mT, identity)
    torch.testing.assert_close(
        joint_a.mT @ joint_a,
        right[:4].mT @ right[:4],
    )
    torch.testing.assert_close(
        joint_b @ joint_a,
        (left[:, :4] * singular_values[:4].unsqueeze(0)) @ right[:4],
    )
    assert torch.equal(layer(inputs), original(inputs))


def test_svd_initialization_trains_both_factors_from_step_zero() -> None:
    layer = GSELinear(
        nn.Linear(12, 7),
        make_config(
            initialization="svd",
            total_rank=4,
            num_experts=2,
            num_generalized_experts=1,
            top_k=1,
            normalize_topk=False,
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
