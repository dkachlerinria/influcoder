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
"""

from __future__ import annotations

import torch

from baselines.common import tokenized_dataset
from baselines.logra.modeling_logra import LoGra


# The old pipeline's config_influence.sh ran LoGRA with this explicit module set
# (all attention + MLP projections), which overrides mlp_only. It matches the set
# the influcoder GT featurizer LoRA-adapts ("all-linear"), so LoGRA and the GT see
# the same gradient geometry -- the fair comparison. mlp_only=True (LoGRA's own
# default) would restrict to MLP layers only and understate it.
LOGRA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]


def score_logra(splits, grad_model: str, lora_rank: int = 8, mlp_only: bool = True,
                max_len: int = 1024, batch_size: int = 1,
                target_modules: list | None = None, seed: int | None = None,
                attn_implementation: str = "sdpa",
                meter=None) -> dict[str, torch.Tensor]:
    """Returns {"logra_raw": [A,P], "logra_fim": [A,P]} score matrices over the
    eval anchors (A) x eval pool (P).

    `seed` only makes the draw reproducible; it changes no algorithm. LoGra
    initializes its logix_lora_A / _C projections with kaiming_uniform_ and
    never seeds them (upstream behaviour, copied verbatim), so every run is a
    fresh random projection and repeated runs at identical settings differ by a
    non-trivial margin. Any single LoGRA number is one draw from a distribution
    -- report a spread over seeds, not a point estimate.
    """
    if target_modules is None:
        target_modules = LOGRA_TARGET_MODULES
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
    pool_embeds = logra.encode(pool_ds, batch_size=batch_size, is_test=False)
    train_fim = logra.fim
    raw_anchor_embeds = logra.encode(anchor_ds, batch_size=batch_size, is_test=False)

    raw = torch.from_numpy(
        logra.similarity(raw_anchor_embeds, pool_embeds, mode="cosine")
    ).float()

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
