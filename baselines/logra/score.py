"""LoGRA baseline scorer, driven on influcoder's eval split.

Faithful to influence_eval/compute_logra_scores.py: the pool (candidates) is
encoded first as the "train" corpus (accumulating the FIM, is_test=False), then
the anchors are encoded (is_test=False for the raw variant), and similarity is
cosine. Two variants are produced, exactly as the old pipeline did:

  logra_raw -- cosine on raw LoRA-B per-sample gradients (no FIM)
  logra_fim -- cosine on FIM-preconditioned anchor gradients

`modeling_logra.LoGra` is copied verbatim from the old repo; the only new code
is feeding it influcoder Samples (tokenized identically to the GT featurizer)
instead of pre-tokenized HF datasets on disk.

BIG_GPU
-------
Set the `BIG_GPU=1` env var (or pass `big_gpu=True` to `score_logra`) on a
GPU with plenty of headroom (40GB+) to batch the per-sample gradient encode
instead of the default batch_size=1 loop. This is SAFE, not an approximation:
`modeling_logra.py` passes `num_items_in_batch=1` to the loss, forcing
`reduction="sum"`, which decomposes exactly per sample -- padded positions
carry `-100` labels and contribute zero. See FINDINGS.md's corrected batching
entry (the earlier "batching dilutes the per-sample gradient" claim was
wrong) and `.tuning_logs/vercors18_logra_batched_50x50.py` for the
measurements this was validated against (~1.6-2.8x speedup, agg-Spearman
drift ~0.001-0.013, noise-level).

This method's custom per-sample-gradient backward retains full per-layer
activations for every sample in the batch at once, so memory scales steeply
with batch size (much more so than ordinary batched forward+backward) --
there is no safe universal batch size, so BIG_GPU auto-retries with a halved
batch size (fresh model reload, not a resumed encode) on CUDA OOM rather than
assume one.

compute_fim
-----------
`compute_fim: bool = True` (default preserves existing behavior exactly, so
every current caller is unaffected). Set `compute_fim=False` when only
`logra_raw` is needed (every actual usage in this repo, per FINDINGS.md:
"FIM-preconditioned LoGRA is much worse than raw at every rank/proxy -- use
raw") to skip `_precondition`'s `torch.linalg.pinv` entirely.

This is not a minor optimization at higher rank: `CustomLoraB.fim` has shape
`[n_blocks, k, k]` where `k = rank**2` (the flattened r x r weight), so pinv
cost scales roughly as `k**3 = rank**6`. Measured directly (Qwen3-0.6B,
rank=32, big_gpu batched): computing both variants took ~585ms/sample;
`logra_raw` alone (`compute_fim=False`) takes ~37ms/sample -- a ~16x
difference from one unused matrix inversion, invisible at rank=8 (k=64, this
step is trivial there) but dominant at rank=32 (k=1024, (32/8)**6 = 4096x
more pinv FLOPs than at rank=8).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from baselines.common import ListDataset, tokenized_dataset
from baselines.logra.modeling_logra import LoGra

BIG_GPU = bool(int(os.environ.get("BIG_GPU", "0")))
BIG_GPU_START_BATCH_SIZE = 32

# The old pipeline's config_influence.sh ran LoGRA with this explicit module set
# (all attention + MLP projections), which overrides mlp_only. It matches the set
# the influcoder GT featurizer LoRA-adapts ("all-linear"), so LoGRA and the GT see
# the same gradient geometry -- the fair comparison. mlp_only=True (LoGRA's own
# default) would restrict to MLP layers only and understate it.
LOGRA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]


def _encode_sorted(logra, ds, batch_size, is_test):
    """Length-sort `ds` before calling logra.encode() (keeps padding waste low
    within a batch), then un-sort the returned [N, k] embeddings back to
    `ds`'s original order."""
    lengths = [len(ids) for ids in ds.input_ids]
    order = np.argsort(lengths)
    rows = [(ds.input_ids[i], ds.labels[i]) for i in order]
    sorted_ds = ListDataset(rows)
    embeds = logra.encode(sorted_ds, batch_size=batch_size, is_test=is_test)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return embeds[inverse]


def score_logra(splits, grad_model: str, lora_rank: int = 8, mlp_only: bool = True,
                max_len: int = 1024, batch_size: int = 1,
                target_modules: list | None = None, seed: int | None = None,
                attn_implementation: str = "sdpa", big_gpu: bool | None = None,
                compute_fim: bool = True,
                meter=None) -> dict[str, torch.Tensor]:
    """Returns {"logra_raw": [A,P], "logra_fim": [A,P]} score matrices over the
    eval anchors (A) x eval pool (P) -- or just {"logra_raw": [A,P]} if
    `compute_fim=False` (see module docstring: this skips a pinv step that's
    trivial at rank=8 but can dominate total cost at higher rank).

    `seed` only makes the draw reproducible; it changes no algorithm. LoGra
    initializes its logix_lora_A / _C projections with kaiming_uniform_ and
    never seeds them (upstream behaviour, copied verbatim), so every run is a
    fresh random projection and repeated runs at identical settings differ by a
    non-trivial margin. Any single LoGRA number is one draw from a distribution
    -- report a spread over seeds, not a point estimate.

    `big_gpu` (default: the BIG_GPU env var) switches from the plain
    batch_size=1 loop to length-sorted batching with an auto-retry-on-OOM
    ladder (see module docstring) -- everything else about the computation,
    including its correctness, is unchanged.
    """
    if target_modules is None:
        target_modules = LOGRA_TARGET_MODULES
    if big_gpu is None:
        big_gpu = BIG_GPU

    bs = BIG_GPU_START_BATCH_SIZE if big_gpu else batch_size
    while True:
        try:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            logra = LoGra.from_pretrained(
                model_name=grad_model, rank=lora_rank, mlp_only=mlp_only,
                target_modules=target_modules, attn_implementation=attn_implementation,
            )
            tok = logra.tokenizer
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            if meter is not None:
                meter.model_ready()

            pool_ds = tokenized_dataset(tok, splits["eval_pool"], max_len)      # corpus
            anchor_ds = tokenized_dataset(tok, splits["eval_anchors"], max_len)  # queries

            # Order matters: corpus first (accumulates FIM), then anchors.
            if big_gpu:
                pool_embeds = _encode_sorted(logra, pool_ds, bs, is_test=False)
                train_fim = logra.fim
                raw_anchor_embeds = _encode_sorted(logra, anchor_ds, bs, is_test=False)
            else:
                pool_embeds = logra.encode(pool_ds, batch_size=bs, is_test=False)
                train_fim = logra.fim
                raw_anchor_embeds = logra.encode(anchor_ds, batch_size=bs, is_test=False)
            break
        except torch.cuda.OutOfMemoryError:
            try:
                del logra
            except NameError:
                pass
            torch.cuda.empty_cache()
            if not big_gpu or bs <= 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [BIG_GPU] OOM at batch_size={bs * 2} for {grad_model} -- "
                  f"retrying at batch_size={bs}")

    raw = torch.from_numpy(
        logra.similarity(raw_anchor_embeds, pool_embeds, mode="cosine")
    ).float()

    if not compute_fim:
        return {"logra_raw": raw}

    logra.fim = train_fim
    # pinv dispatches to aten._linalg_svd, which FlopCounterMode has no handler
    # for -- charge it analytically so logra_fim's cost is honest.
    if meter is not None:
        n_blocks, k, _ = train_fim.size()
        from baselines.cost import pinv_flops
        meter.add_flops(pinv_flops(n_blocks, k),
                        f"logra_fim pinv: {n_blocks} blocks x {k}^3")
    precond_anchor_embeds = (
        logra._precondition(torch.from_numpy(raw_anchor_embeds)).float().numpy()
    )
    fim = torch.from_numpy(
        logra.similarity(precond_anchor_embeds, pool_embeds, mode="cosine")
    ).float()

    return {"logra_raw": raw, "logra_fim": fim}
