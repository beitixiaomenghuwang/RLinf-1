"""SVD and router initialization for Ortho-Hydra."""

import torch
from torch import nn


@torch.no_grad()
def principal_bases(
    weight: torch.Tensor, total_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sign-canonicalized top left and right singular bases."""
    compute_device = weight.device
    if compute_device.type == "cpu" and torch.cuda.is_available():
        compute_device = torch.device("cuda", torch.cuda.current_device())
    matrix = weight.detach().to(device=compute_device, dtype=torch.float32)
    left, _, right = torch.linalg.svd(matrix, full_matrices=False)
    left = left[:, :total_rank]
    right = right[:total_rank]
    pivot = right.abs().argmax(dim=1)
    rows = torch.arange(total_rank, device=right.device)
    signs = torch.sign(right[rows, pivot])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return left * signs.unsqueeze(0), right * signs.unsqueeze(1)


@torch.no_grad()
def initialize_router(router: nn.Linear, std: float, seed: int | None) -> None:
    """Initialize one router without changing global RNG state."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    weight = torch.empty(router.weight.shape, dtype=torch.float32, device="cpu")
    weight.normal_(mean=0.0, std=std, generator=generator)
    router.weight.copy_(weight.to(router.weight))
    if router.bias is not None:
        router.bias.zero_()
