"""Bi-encoder distillation: train a sentence encoder so that embedding cosine
reproduces gradient-influence cosine.

Loss = alpha * (1 - Pearson r over the score block)      -- global alignment
     + (1-alpha) * row-wise KL of softmaxed scores        -- listwise ranking
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .gradients import hardware_profile


def load_encoder(name: str, device: str = "cuda", max_seq_len: int = 512):
    from sentence_transformers import SentenceTransformer
    _, attn = hardware_profile(device)
    enc = SentenceTransformer(name, device=device,
                              model_kwargs={"attn_implementation": attn})
    enc.max_seq_length = max_seq_len
    return enc


def embed(enc, texts: list[str], batch_size: int = 32) -> np.ndarray:
    enc.eval()
    with torch.inference_mode():
        return enc.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)


def pearson_kl_loss(scores: torch.Tensor, labels: torch.Tensor,
                    alpha: float = 0.5, temperature: float = 0.05) -> torch.Tensor:
    s, l = scores.flatten(), labels.flatten()
    sc, lc = s - s.mean(), l - l.mean()
    pearson = 1 - (sc * lc).sum() / (sc.norm() * lc.norm() + 1e-8)
    kl = F.kl_div(F.log_softmax(scores / temperature, dim=1),
                  F.softmax(labels / temperature, dim=1), reduction="batchmean")
    return alpha * pearson + (1 - alpha) * kl


def _soft_rank(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Differentiable rank approximation: rank_i ~= sum_j sigmoid((x_i-x_j)/eps).
    Pairwise O(n^2), fine at block sizes here (k_anchors*m_candidates)."""
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    return torch.sigmoid(diff / eps).sum(dim=1)


def soft_spearman_loss(scores: torch.Tensor, labels: torch.Tensor,
                       eps: float = 1e-2) -> torch.Tensor:
    """1 - Pearson correlation of soft ranks -- a differentiable surrogate
    for Spearman rather than Pearson-on-raw-values, since Spearman is what
    the eval metric actually measures."""
    rs, rl = _soft_rank(scores.flatten(), eps), _soft_rank(labels.flatten(), eps)
    rsc, rlc = rs - rs.mean(), rl - rl.mean()
    return 1 - (rsc * rlc).sum() / (rsc.norm() * rlc.norm() + 1e-8)


def infonce_loss(scores: torch.Tensor, labels: torch.Tensor,
                 temperature: float = 0.05) -> torch.Tensor:
    """Cross-entropy against each anchor's single top-target candidate (a
    hard positive), instead of matching the whole soft label distribution
    the KL term uses -- sharpens toward the one best match per anchor."""
    pos_idx = labels.argmax(dim=1)
    return F.cross_entropy(scores / temperature, pos_idx)


LOSS_FNS = {
    "pearson_kl": pearson_kl_loss,
    "soft_spearman": soft_spearman_loss,
    "infonce": infonce_loss,
}


def _sample_candidates(rng, targets_row_block: torch.Tensor, n_pool: int,
                       m_candidates: int, hard_ratio: float) -> torch.Tensor:
    """m_candidates indices into the pool: a hard_ratio fraction are the
    hardest negatives for this anchor block (highest target similarity --
    the ones a lazy encoder would conflate), the rest uniform random. Pure
    random candidates are almost all easy negatives once the pool is more
    than a few dozen items, so the loss stops teaching the encoder anything
    once it clears the easy majority; hard negatives keep gradient signal
    where the ranking is actually still wrong."""
    n_hard = int(m_candidates * hard_ratio)
    if n_hard > 0:
        # union of each anchor's top candidates, so a block of anchors pulls
        # in a diverse hard set rather than one anchor's neighborhood only
        top_k = max(1, n_hard // targets_row_block.shape[0] + 1)
        hard = torch.unique(targets_row_block.topk(min(top_k * 2, n_pool), dim=1).indices)
        if hard.numel() > n_hard:
            hard = hard[torch.randperm(hard.numel())[:n_hard]]
    else:
        hard = torch.empty(0, dtype=torch.long)
    n_random = m_candidates - hard.numel()
    remaining = [i for i in range(n_pool) if i not in set(hard.tolist())]
    # dtype=torch.long is required explicitly: torch.tensor([]) (the
    # hard_ratio=1.0 case, n_random=0) otherwise defaults to float32, and
    # cat-ing that with `hard` silently produces a float index tensor that
    # crashes at the pool_texts[j] lookup instead of here.
    random_idx = torch.tensor(rng.sample(remaining, min(n_random, len(remaining))),
                              dtype=torch.long)
    return torch.cat([hard, random_idx])


def distill(enc, anchor_texts: list[str], pool_texts: list[str],
            targets: torch.Tensor, epochs: int, k_anchors: int = 8,
            m_candidates: int = 16, lr: float = 5e-5, seed: int = 0,
            hard_ratio: float = 0.5, epoch_eval=None,
            select_best_on: str = "per_anchor_mean", restore_best: bool = True,
            grad_accum_steps: int = 1, alpha: float = 0.5,
            temperature: float = 0.05, weight_decay: float = 0.01,
            max_grad_norm: float = 1.0, warmup_frac: float = 0.1,
            lr_schedule: str = "linear", hard_ratio_end: float | None = None) -> dict:
    """Each step embeds a block of k anchors x m candidates (mixed
    random/hard-negative) and regresses the cosine block onto the gradient
    targets. By default (`restore_best=True`), restores the encoder to
    whichever epoch scored best on `epoch_eval` (by `select_best_on`) before
    returning -- eval Spearman is noisy at these sample sizes and reliably
    peaks then degrades, so reporting the last epoch is reporting
    overfitting, not the method. Pass `restore_best=False` to keep whatever
    the encoder actually looks like after the full `epochs` budget instead --
    e.g. EXP1's BIG_GPU_FINAL config, which reports fixed-epoch-8 metrics and
    wants the SAVED checkpoint to match that (previously only the reported
    metric was pinned to the last epoch; the actual `enc`/saved checkpoint
    was still silently best-epoch-restored underneath it regardless of what
    got printed).

    `grad_accum_steps` accumulates over that many blocks before stepping, so
    the effective batch is k_anchors * grad_accum_steps. This matters more
    here than in ordinary training: the Pearson term is a correlation
    estimated over the score block, and a small block gives a noisy estimate
    of it. The old repo trained at 12 anchors x 4 accumulation = 48 per
    update; 1 reproduces the no-accumulation behaviour this repo had.

    `hard_ratio_end`, if set, linearly ramps hard_ratio -> hard_ratio_end over
    the run (curriculum: start easy, finish hard) instead of holding it fixed.
    Returns {"epoch_losses", "epoch_metrics", "best_epoch"}."""
    device = enc.device.type if hasattr(enc.device, "type") else "cuda"
    amp_dtype, _ = hardware_profile(device)
    use_scaler = amp_dtype == torch.float32  # pre-Ampere: autocast fp16 + scaler
    autocast_dtype = torch.float16 if use_scaler else amp_dtype

    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=weight_decay)
    blocks_per_epoch = math.ceil(len(anchor_texts) / k_anchors)
    # The schedule advances per OPTIMIZER step, not per block, or accumulation
    # would silently compress the warmup and decay into 1/accum of the run.
    total_steps = math.ceil(blocks_per_epoch * epochs / grad_accum_steps)
    warmup = max(1, int(warmup_frac * total_steps))
    if lr_schedule == "cosine":
        def lr_lambda(s):
            if s < warmup:
                return (s + 1) / warmup
            prog = (s - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    else:
        def lr_lambda(s):
            return min(1.0, (s + 1) / warmup) * max(
                0.0, 1 - max(0, s - warmup) / max(1, total_steps - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)
    rng = random.Random(seed)

    def tokenize(texts):
        feats = enc.tokenize(texts)
        return {k: v.to(device) for k, v in feats.items() if torch.is_tensor(v)}

    epoch_losses, epoch_metrics = [], []
    best_score, best_epoch, best_state = -float("inf"), -1, None
    enc.train()
    for epoch in range(epochs):
        if hard_ratio_end is not None and epochs > 1:
            cur_hard_ratio = hard_ratio + (hard_ratio_end - hard_ratio) * epoch / (epochs - 1)
        else:
            cur_hard_ratio = hard_ratio
        order = list(range(len(anchor_texts)))
        rng.shuffle(order)
        losses = []
        starts = list(range(0, len(order), k_anchors))
        for b, i in enumerate(starts):
            a_idx = torch.tensor(order[i:i + k_anchors])
            c_idx = _sample_candidates(rng, targets[a_idx], len(pool_texts),
                                       m_candidates, cur_hard_ratio)
            a_feats = tokenize([anchor_texts[j] for j in a_idx])
            c_feats = tokenize([pool_texts[j] for j in c_idx])
            with torch.amp.autocast(device, dtype=autocast_dtype):
                za = F.normalize(enc(a_feats)["sentence_embedding"], dim=1)
                zc = F.normalize(enc(c_feats)["sentence_embedding"], dim=1)
                loss = pearson_kl_loss(za @ zc.T, targets[a_idx][:, c_idx].to(device),
                                       alpha=alpha, temperature=temperature)
            # Scale so the accumulated gradient is the MEAN over the group, not
            # the sum -- otherwise accumulation silently multiplies the LR.
            scaler.scale(loss / grad_accum_steps).backward()
            losses.append(loss.item())
            # Step on a full group, and on the last block of the epoch so a
            # trailing partial group is not dropped.
            if (b + 1) % grad_accum_steps == 0 or (b + 1) == len(starts):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        epoch_losses.append(float(np.mean(losses)))
        msg = f"  epoch {epoch + 1}/{epochs}  loss={epoch_losses[-1]:.4f}"
        if epoch_eval is not None:
            m = epoch_eval()
            epoch_metrics.append(m)
            msg += (f"  eval per-anchor rho={m['per_anchor_mean']:+.4f}"
                    f"  agg rho={m['aggregated']:+.4f}")
            if m[select_best_on] > best_score:
                best_score, best_epoch = m[select_best_on], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
                msg += "  *"
            enc.train()
        print(msg)

    if best_state is not None and restore_best:
        enc.load_state_dict(best_state)
        print(f"  restored best checkpoint: epoch {best_epoch + 1} "
              f"({select_best_on}={best_score:+.4f})")
    elif best_state is not None:
        print(f"  restore_best=False: keeping final-epoch ({epochs}) weights "
              f"(best was epoch {best_epoch + 1}, {select_best_on}={best_score:+.4f})")
    return {"epoch_losses": epoch_losses, "epoch_metrics": epoch_metrics, "best_epoch": best_epoch}
