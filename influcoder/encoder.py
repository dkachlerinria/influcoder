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
    random_idx = torch.tensor(rng.sample(remaining, min(n_random, len(remaining))))
    return torch.cat([hard, random_idx])


def distill(enc, anchor_texts: list[str], pool_texts: list[str],
            targets: torch.Tensor, epochs: int, k_anchors: int = 8,
            m_candidates: int = 16, lr: float = 5e-5, seed: int = 0,
            hard_ratio: float = 0.5, alpha: float = 0.5, epoch_eval=None,
            select_best_on: str = "per_anchor_mean") -> dict:
    """Each step embeds a block of k anchors x m candidates (mixed
    random/hard-negative) and regresses the cosine block onto the gradient
    targets. Restores the encoder to whichever epoch scored best on
    `epoch_eval` (by `select_best_on`) before returning -- eval Spearman is
    noisy at these sample sizes and reliably peaks then degrades, so
    reporting the last epoch is reporting overfitting, not the method.
    Returns {"epoch_losses", "epoch_metrics", "best_epoch"}."""
    device = enc.device.type if hasattr(enc.device, "type") else "cuda"
    amp_dtype, _ = hardware_profile(device)
    use_scaler = amp_dtype == torch.float32  # pre-Ampere: autocast fp16 + scaler
    autocast_dtype = torch.float16 if use_scaler else amp_dtype

    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=0.01)
    total_steps = math.ceil(len(anchor_texts) / k_anchors) * epochs
    warmup = max(1, int(0.1 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup)
        * max(0.0, 1 - max(0, s - warmup) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)
    rng = random.Random(seed)

    def tokenize(texts):
        feats = enc.tokenize(texts)
        return {k: v.to(device) for k, v in feats.items() if torch.is_tensor(v)}

    epoch_losses, epoch_metrics = [], []
    best_score, best_epoch, best_state = -float("inf"), -1, None
    enc.train()
    for epoch in range(epochs):
        order = list(range(len(anchor_texts)))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), k_anchors):
            a_idx = torch.tensor(order[i:i + k_anchors])
            c_idx = _sample_candidates(rng, targets[a_idx], len(pool_texts),
                                       m_candidates, hard_ratio)
            a_feats = tokenize([anchor_texts[j] for j in a_idx])
            c_feats = tokenize([pool_texts[j] for j in c_idx])
            with torch.amp.autocast(device, dtype=autocast_dtype):
                za = F.normalize(enc(a_feats)["sentence_embedding"], dim=1)
                zc = F.normalize(enc(c_feats)["sentence_embedding"], dim=1)
                loss = pearson_kl_loss(za @ zc.T, targets[a_idx][:, c_idx].to(device), alpha=alpha)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(loss.item())
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

    if best_state is not None:
        enc.load_state_dict(best_state)
        print(f"  restored best checkpoint: epoch {best_epoch + 1} "
              f"({select_best_on}={best_score:+.4f})")
    return {"epoch_losses": epoch_losses, "epoch_metrics": epoch_metrics, "best_epoch": best_epoch}
