"""In-Place adapter at the existing native update contraction."""
import torch

from ssw import gather_tokens, select_blocks


@torch.no_grad()
def token_scores(h, t, projection=None):
    """||h_i||_2 ||t_i P||_2, computed in float32 without outer products."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        v = t.float() if projection is None else t.float() @ projection.float()
        return torch.linalg.vector_norm(h.float(), dim=-1) * torch.linalg.vector_norm(v, dim=-1)


@torch.no_grad()
def update(h, t, projection, learning_rate, *, contract,
           keep_fraction=0.5, block_size=64, method="score", seed=0,
           layer_idx=0, chunk_idx=0, write_allowed=True):
    """Return native dw for [N, D] inputs, or None for an excluded write.

    Pass the backend's original opt_einsum.contract. The caller owns commit,
    clipping, resets, target construction, and eligibility. No N/K rescaling.
    """
    if not write_allowed:
        return None
    if keep_fraction < 1:
        scores = token_scores(h, t, projection) if method in ("score", "reverse") else h.new_zeros(h.shape[0])
        indices = select_blocks(
            scores, keep_fraction, block_size, method=method,
            seed=seed + 1009 * layer_idx + 9176 * chunk_idx,
        )
        h, t = gather_tokens(h, indices), gather_tokens(t, indices)
    if projection is not None:
        return contract("c h, c d, d e -> e h", h, t, projection) * learning_rate
    return contract("c h, c d -> d h", h, t) * learning_rate
