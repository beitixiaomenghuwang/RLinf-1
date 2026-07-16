"""Initialization helpers for GSE adapters."""

import math
from collections.abc import Sequence

import torch
from torch import nn


def _cpu_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def initialize_expert_factors(
    lora_a_layers: Sequence[nn.Linear],
    lora_b_layers: Sequence[nn.Linear],
    *,
    method: str,
    seed: int | None,
    orthogonal_gain: float,
) -> None:
    """Initialize expert factors without modifying the base model weight."""
    if len(lora_a_layers) != len(lora_b_layers):
        raise ValueError("A and B layer counts must match")
    if not lora_a_layers:
        return

    generator = _cpu_generator(seed)
    in_features = lora_a_layers[0].in_features
    total_rank = sum(layer.out_features for layer in lora_a_layers)

    if method == "orthogonal_zero":
        random_matrix = torch.randn(
            in_features,
            total_rank,
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
        joint_a = basis.mT.mul_(orthogonal_gain)
    elif method == "kaiming_zero":
        bound = 1.0 / math.sqrt(in_features)
        joint_a = torch.empty(total_rank, in_features, dtype=torch.float32)
        joint_a.uniform_(-bound, bound, generator=generator)
    else:
        raise ValueError(f"Unsupported GSE initialization: {method}")

    offset = 0
    for lora_a in lora_a_layers:
        rank = lora_a.out_features
        lora_a.weight.copy_(
            joint_a[offset : offset + rank].to(
                device=lora_a.weight.device,
                dtype=lora_a.weight.dtype,
            )
        )
        offset += rank
    for lora_b in lora_b_layers:
        lora_b.weight.zero_()


@torch.no_grad()
def initialize_router(
    router: nn.Linear,
    *,
    standard_deviation: float,
    seed: int | None,
) -> None:
    """Initialize a router with small random logits and optional zero bias."""
    generator = _cpu_generator(seed)
    weight = torch.empty(router.weight.shape, dtype=torch.float32, device="cpu")
    weight.normal_(mean=0.0, std=standard_deviation, generator=generator)
    router.weight.copy_(weight.to(router.weight))
    if router.bias is not None:
        router.bias.zero_()


def joint_lora_a(layers: Sequence[nn.Module]) -> torch.Tensor:
    """Concatenate all expert A factors in expert order."""
    weights = [layer.lora_a.weight for layer in layers]
    if not weights:
        raise ValueError("At least one expert is required")
    return torch.cat(weights, dim=0)


def orthogonality_error(layers: Sequence[nn.Module]) -> torch.Tensor:
    """Return the maximum absolute error from a scaled orthogonal Gram matrix."""
    joint_a = joint_lora_a(layers).float()
    row_norm = joint_a.norm(dim=1).mean()
    gram = joint_a @ joint_a.mT
    target = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    target = target * row_norm.square()
    return (gram - target).abs().max()
