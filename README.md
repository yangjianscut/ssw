# Structured Sparse Writing (SSW): reference implementation

This repository provides the core write-selection method from **What to Write:
Structured Sparse Writing for Test-Time Training**. It includes block selection,
chronological gathering, an In-Place update adapter, and a LaCT gather interface.
The functions integrate SSW at an existing backend's write site while preserving
its read stream and native update operator. Experimental results are documented
in the accompanying supplementary records.

## Files and installation

- `ssw.py`: block-score sums, Top-K, chronological indices, and gather.
- `inplace.py`: float32 rank-one norm scores and the native update contraction.
- `lact.py`: gather keys, values, and learning-rate maps with shared indices.
- `configs.json`: reference settings for the adapters, including the archived
  In-Place protocol; pass these settings explicitly to the functions below.
- `requirements.txt`: dependencies for these adapters.

Use Python 3.10 or newer in the backend environment:

```sh
python -m pip install -r requirements.txt
```

Put this directory on the backend's Python import path. Install the dependencies
in a PyTorch environment compatible with the host backend.

## Method

At an eligible completed write chunk, sum nonnegative token scores within
contiguous blocks. Retain `ceil(keep_fraction * number_of_blocks)` blocks,
sort their indices into sequence order, and gather every associated write
tensor with the same indices. Retained contributors keep their original
weights and learning rates, with a rescaling factor of one.
Reads, features, targets, positional operations, convolution, write schedule,
reset behavior, and incomplete-chunk handling remain in the original backend.
Call selection after feature/target construction and immediately before the
native write operation. Full-retention updates bypass gathering and scoring.

`method="score"` is SSW. `random`, `reverse`, and `periodic` provide small
selection controls; `block_size=1` provides token ranking. Random sampling uses
a local generator. The periodic rule selects evenly spaced blocks (every
other block at 50%). Ties use PyTorch Top-K behavior. The `random` control in
this package uses gathering; LaCT's reported random baseline uses full-size
masking. Preserve that execution distinction when comparing evaluation results.

## In-Place integration

For one completed chunk, `h` and `t` have shapes `[N, D_h]` and `[N, D_t]`;
`projection` is `[D_t, D_v]` or `None`. Replace only the native `dw`
contraction with:

```python
from inplace import update

dw = update(
    current_h, current_t,
    self.ttt_proj.weight if self.ttt_proj is not None else None,
    self.ttt_lr,
    contract=contract,  # original opt_einsum.contract
    layer_idx=self.layer_idx,
    chunk_idx=i,
    keep_fraction=0.5,
    block_size=64,
)
current_w = current_w + dw
```

If the backend excludes a write, pass `write_allowed=False` and commit only
when `dw is not None`. The backend defines query boundaries and write eligibility.
A random control uses `method="random", seed=...`; its per-site seed is
`seed + 1009 * layer_idx + 9176 * chunk_idx`, as in the archived block adapter.
Full retention uses `keep_fraction=1.0` and the original contraction.

The archived In-Place reference configuration uses 1024-token chunks, 64-token
blocks, 50% retention, learning rate 3, and adaptation at layers
0/6/12/18/24/30. These settings describe the archived adapter protocol;
main-table evaluation settings should be taken from their corresponding run
records. Score norms and target projection for scoring use float32; gathered
tensors and the native contraction retain backend dtypes. The archived update
is unclipped. This function extracts the strength/gather path of the archived
`ttt_block_sparse_runtime.py` and `inplace_block_sparse_runtime.py` adapters.

## LaCT integration

Supply the already reduced learning-rate token statistic as `scores [..., N]`.
The chunk tensors use `[..., N, D]` layout and share batch/head dimensions:

```python
from lact import gather_write

# vi is stored as [..., D, N] in the native LaCT inner loop.
ki, v_write, rates, indices = gather_write(
    scores, ki, vi.transpose(-2, -1), (lr0i, lr1i, lr2i),
    keep_fraction=0.125, block_size=64,
)
vi = v_write.transpose(-2, -1)
lr0i, lr1i, lr2i = rates
# Continue the existing inner optimizer with ki, vi, lr0i, lr1i, lr2i.
# Keep the full query/read stream unchanged.
```

For complete 2048-token chunks this retains 4 of 32 blocks (256 tokens).
The caller supplies `scores` after the backend-specific head/inner-step
reduction of the learning-rate statistic. This interface starts with those
reduced scores; Muon, momentum, and normalization remain in the host backend.

## Integration scope

This is a method reference implementation covering score-based selection,
shared-index gathering, and the write interfaces illustrated above. For model
evaluation, integrate these functions with the original backend model code,
tokenizer, checkpoints, benchmark inputs, prompt construction, and scoring
protocol. Use the evaluation-specific settings and source descriptions in the
paper and accompanying records to identify the protocol for each result.
