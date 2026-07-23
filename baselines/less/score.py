"""LESS baseline.

Faithful to influence_eval/compute_gradient_scores.py: attach a fresh seeded
LoRA to the base model, take per-sample SGD gradients on the LoRA params,
random-project them with a TRAK projector (fast_jl CudaProjector when available,
BasicProjector otherwise), L2-normalize, cosine between anchors and pool.

Same gradient source and projection dim as influcoder's ground truth; the
difference is LESS uses a raw random projection of a *large* LoRA (rank 128,
the LESS-paper default) with no distillation, so its number isolates "random
projection of gradients" from influcoder's learned encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from baselines.common import tokenized_dataset
from baselines.less.less_embeds import collect_grads, normalize_embeddings_in_chunks
from baselines.less.model_utils import load_base_with_fresh_lora


def score_less(splits, model_name: str, proj_dim: int = 8192, max_len: int = 1024,
               lora_rank: int = 128, lora_alpha: int = 512, lora_dropout: float = 0.1,
               lora_seed: int = 0, project_interval: int = 8,
               meter=None) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_base_with_fresh_lora(
        model_name=model_name, tokenizer=tok, lora_target_modules="all-linear",
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        seed=lora_seed,
    )
    if meter is not None:
        meter.model_ready()

    used_cuda_proj = False

    def grads(samples):
        nonlocal used_cuda_proj
        dl = torch.utils.data.DataLoader(
            tokenized_dataset(tok, samples, max_len), batch_size=1, shuffle=False)
        g, uses_custom_cuda = collect_grads(
            dl, model, proj_dim=proj_dim, adam_optimizer_state=None,
            gradient_type="sgd", project_interval=project_interval)
        used_cuda_proj = uses_custom_cuda
        return normalize_embeddings_in_chunks(g, chunk_size=10000, dim=1,
                                              eps=1e-12, in_place=False)

    pool = grads(splits["eval_pool"])
    anchors = grads(splits["eval_anchors"])

    # fast_jl's CudaProjector is a custom kernel FlopCounterMode cannot see, so
    # charge the projection analytically. TRAK's BasicProjector is a torch.mm and
    # IS counted -- adding it there would double count.
    if meter is not None and used_cuda_proj:
        from baselines.cost import projection_flops
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n = len(splits["eval_pool"]) + len(splits["eval_anchors"])
        meter.add_flops(projection_flops(n_lora, proj_dim, n),
                        f"LESS CudaProjector: 2*{n_lora}*{proj_dim}*{n}")

    return (anchors.float() @ pool.float().T)
