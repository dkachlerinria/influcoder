"""RDS+ baseline.

Faithful to influence_eval/compute_rdsplus_scores.py: cosine of the target
model's representations, using SGPT 'weighted_mean' pooling over the last hidden
state (token i weighted proportional to its position i+1). No gradients.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from baselines.common import tokenized_dataset


@torch.no_grad()
def weighted_mean_embeds(model, dataloader, device) -> torch.Tensor:
    """Public (not `_`-prefixed) so `baselines.exp1.part3` can time just this
    per-sample representation step directly, the same way it reuses LoGRA's
    `encode_sorted` -- without going through `score_rdsplus`'s full
    anchor-vs-pool matmul, which is out of scope for a process-only timing."""
    all_embeds = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
        hidden = out.hidden_states[-1]
        w = torch.arange(hidden.size(1), device=device).unsqueeze(0) + 1
        w = w / w.sum(dim=1, keepdim=True)
        emb = torch.sum(hidden * w.unsqueeze(-1), dim=1)
        emb = emb / torch.linalg.vector_norm(emb, dim=1, keepdim=True)
        all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0)


def score_rdsplus(splits, model_name: str, max_len: int = 1024,
                  batch_size: int = 1, attn_implementation: str = "eager",
                  meter=None) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    if meter is not None:
        meter.model_ready()

    pool_dl = DataLoader(tokenized_dataset(tok, splits["eval_pool"], max_len),
                         batch_size=batch_size, shuffle=False)
    anchor_dl = DataLoader(tokenized_dataset(tok, splits["eval_anchors"], max_len),
                           batch_size=batch_size, shuffle=False)
    pool = weighted_mean_embeds(model, pool_dl, device)
    anchors = weighted_mean_embeds(model, anchor_dl, device)
    return torch.matmul(anchors, pool.T).float()
