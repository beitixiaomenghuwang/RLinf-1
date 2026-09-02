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
def _full_svd_factors(
    weight: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sign-canonicalized leading factors from a full float32 SVD.

    As in MoORE, the complete reduced SVD is computed on the visible CUDA
    device when the base model is still on CPU. The GSE rank only determines
    how many exact singular triplets are retained as trainable factors; it does
    not switch to a randomized low-rank approximation.
    """
    compute_device = weight.device
    if compute_device.type == "cpu" and torch.cuda.is_available():
        compute_device = torch.device("cuda", torch.cuda.current_device())
    matrix = weight.detach().to(device=compute_device, dtype=torch.float32)
    left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
    left = left[:, :rank]
    singular_values = singular_values[:rank]
    right = right[:rank]
    pivot = right.abs().argmax(dim=1)
    row_indices = torch.arange(rank, device=right.device)
    signs = torch.sign(right[row_indices, pivot])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (
        left * signs.unsqueeze(0),
        singular_values,
        right * signs.unsqueeze(1),
    )


@torch.no_grad()
def initialize_expert_factors(
    lora_a_layers: Sequence[nn.Linear],
    lora_b_layers: Sequence[nn.Linear],
    *,
    method: str,
    seed: int | None,
    orthogonal_gain: float,
    base_weight: torch.Tensor | None = None,
    scalings: Sequence[float] | None = None,
    svd_rho: float = 1.0,
) -> None:
    """Initialize expert factors from a random basis or a balanced full SVD.

    ``svd`` follows VLA-GSE's factor geometry: each expert receives disjoint
    singular triplets and splits ``S / (scaling * rho)`` evenly across A and B.
    """
    if len(lora_a_layers) != len(lora_b_layers):
        raise ValueError("A and B layer counts must match")
    if not lora_a_layers:
        return

    generator = _cpu_generator(seed)
    in_features = lora_a_layers[0].in_features
    out_features = lora_b_layers[0].out_features
    total_rank = sum(layer.out_features for layer in lora_a_layers)

    joint_b: torch.Tensor | None = None
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
    elif method == "svd":
        if base_weight is None:
            raise ValueError(
                f"{method} initialization requires the wrapped base weight"
            )
        if base_weight.shape != (out_features, in_features):
            raise ValueError(
                "base_weight shape must match the wrapped linear layer, got "
                f"{tuple(base_weight.shape)} != {(out_features, in_features)}"
            )
        if total_rank > min(base_weight.shape):
            raise ValueError(
                "svd initialization requires total rank no larger than the base "
                f"weight rank, got {total_rank} > {min(base_weight.shape)}"
            )
        if scalings is None or len(scalings) != len(lora_a_layers):
            raise ValueError("svd initialization requires one scaling per expert")
        if svd_rho <= 0:
            raise ValueError("svd_rho must be positive")
        left, singular_values, right = _full_svd_factors(base_weight, total_rank)
        joint_a = torch.empty_like(right)
        joint_b = torch.empty_like(left)
        offset = 0
        for lora_a, scaling in zip(lora_a_layers, scalings, strict=True):
            rank = lora_a.out_features
            root = (
                singular_values[offset : offset + rank] / (scaling * svd_rho)
            ).sqrt()
            joint_a[offset : offset + rank] = (
                root.unsqueeze(1) * right[offset : offset + rank]
            )
            joint_b[:, offset : offset + rank] = left[
                :, offset : offset + rank
            ] * root.unsqueeze(0)
            offset += rank
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
    if method == "svd":
        offset = 0
        assert joint_b is not None
        for lora_b in lora_b_layers:
            rank = lora_b.in_features
            lora_b.weight.copy_(
                joint_b[:, offset : offset + rank].to(
                    device=lora_b.weight.device,
                    dtype=lora_b.weight.dtype,
                )
            )
            offset += rank
    else:
        for lora_b in lora_b_layers:
            lora_b.weight.zero_()


@torch.no_grad()
def initialize_router(
    router: nn.Linear,
    *,
    method: str,
    standard_deviation: float,
    seed: int | None,
) -> None:
    """Initialize a router from the configured distribution.

    ``default`` and ``kaiming`` both reproduce ``nn.Linear.reset_parameters``
    (``kaiming_uniform_`` with ``a=sqrt(5)``), which draws from
    ``uniform(+/-1/sqrt(in_features))`` and so rescales itself with the router
    width. Only ``normal`` reads ``standard_deviation``; the official GSE repo
    never touches its router, making ``default`` the faithful setting.
    """
    generator = _cpu_generator(seed)
    weight = torch.empty(router.weight.shape, dtype=torch.float32, device="cpu")
    if method == "normal":
        weight.normal_(mean=0.0, std=standard_deviation, generator=generator)
    elif method in ("default", "kaiming"):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    else:
        raise ValueError(f"Unsupported router initialization: {method}")
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
