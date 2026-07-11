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


def distill(enc, anchor_texts: list[str], pool_texts: list[str],
            targets: torch.Tensor, epochs: int, k_anchors: int = 8,
            m_candidates: int = 16, lr: float = 5e-5, seed: int = 0,
            epoch_eval=None) -> list[float]:
    """Each step embeds a block of k anchors x m random candidates and
    regresses the cosine block onto the gradient targets. Returns per-epoch
    mean losses."""
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

    epoch_losses = []
    enc.train()
    for epoch in range(epochs):
        order = list(range(len(anchor_texts)))
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), k_anchors):
            a_idx = torch.tensor(order[i:i + k_anchors])
            c_idx = torch.tensor(rng.sample(range(len(pool_texts)),
                                            min(m_candidates, len(pool_texts))))
            a_feats = tokenize([anchor_texts[j] for j in a_idx])
            c_feats = tokenize([pool_texts[j] for j in c_idx])
            with torch.amp.autocast(device, dtype=autocast_dtype):
                za = F.normalize(enc(a_feats)["sentence_embedding"], dim=1)
                zc = F.normalize(enc(c_feats)["sentence_embedding"], dim=1)
                loss = pearson_kl_loss(za @ zc.T, targets[a_idx][:, c_idx].to(device))
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
            msg += (f"  eval per-anchor rho={m['per_anchor_mean']:+.4f}"
                    f"  agg rho={m['aggregated']:+.4f}")
            enc.train()
        print(msg)
    return epoch_losses
