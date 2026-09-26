"""LaCT gather interface; score reduction and inner optimizer stay external."""
import torch

from ssw import gather_tokens, select_blocks


@torch.no_grad()
def gather_write(scores, keys, values, learning_rates, *, keep_fraction=0.125,
                 block_size=64, method="score", seed=0):
    """Gather associated [..., N, D] tensors using supplied [..., N] scores.

    scores must already be the backend's reduced learning-rate statistic.
    learning_rates is a tuple of token-associated maps; no rescaling is applied.
    Returns keys, values, learning-rate maps, and selected indices.
    """
    indices = select_blocks(scores, keep_fraction, block_size, method=method, seed=seed)
    if indices.shape[-1] == keys.shape[-2]:
        return keys, values, learning_rates, indices
    return (gather_tokens(keys, indices), gather_tokens(values, indices),
            tuple(gather_tokens(rate, indices) for rate in learning_rates), indices)
