"""Structured Sparse Writing: contiguous blocks, sorted gather, unit weights."""
import math

import torch


@torch.no_grad()
def select_blocks(scores, keep_fraction, block_size=64, *, method="score", seed=0):
    """Select indices from token scores [..., N] for a completed write chunk.

    Block scores are sums. Keep ceil(rho * number_of_blocks) whole blocks.
    Supported controls: random blocks, reverse ranking, and periodic blocks.
    Use block_size=1 for token selection. Ties follow torch.topk semantics.
    """
    n = scores.shape[-1]
    if block_size < 1 or n == 0 or n % block_size:
        raise ValueError("write chunk must contain complete nonempty blocks")
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    if method not in ("score", "random", "reverse", "periodic"):
        raise ValueError(f"unknown selection method: {method}")
    count = n // block_size
    k = math.ceil(count * keep_fraction)
    batch = scores.shape[:-1]
    if k == count:
        return torch.arange(n, device=scores.device).expand(*batch, n)
    if method == "periodic":
        blocks = (torch.arange(k, device=scores.device) * count // k).expand(*batch, k)
    else:
        if method == "random":
            rng = torch.Generator(device=scores.device).manual_seed(seed)
            values = torch.rand((*batch, count), device=scores.device, generator=rng)
        else:
            values = scores.float().reshape(*batch, count, block_size).sum(-1)
        blocks = values.topk(k, largest=method != "reverse", sorted=False).indices
        blocks = blocks.sort(dim=-1).values
    return (blocks.unsqueeze(-1) * block_size + torch.arange(
        block_size, device=scores.device
    )).reshape(*batch, k * block_size)


def gather_tokens(tensor, indices):
    """Gather [..., N, D] with matching [..., K] indices; preserve dtype."""
    return tensor.gather(-2, indices.unsqueeze(-1).expand(*indices.shape, tensor.shape[-1]))
