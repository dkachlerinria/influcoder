"""Baseline influence-scoring methods, ported from the tis-ie legacy pipeline
(runs/influence_spearman), for comparison against this repo's own gradient
ground truth (influcoder.gradients.GradientFeaturizer) and trained encoder.

Every function takes anchor/pool `Sample` lists (influcoder.data.Sample) and
returns an [n_anchors, n_pool] numpy score matrix -- the same shape
influcoder.metrics.spearman_metrics expects as `pred`.

Two of tis-ie's five baselines aren't reimplemented here because this repo
already computes the same method as part of its own pipeline:

  * LESS is LoRA-gradient cosine similarity with a JL-style projection --
    exactly what GradientFeaturizer already computes (CountSketch here
    instead of tis-ie's fast_jl/BasicProjector, same idea: project each
    per-sample gradient, cosine the projections). `less_scores` below is a
    thin wrapper so it's callable under the same name as the other
    baselines; it introduces no new algorithm. Note also that unlike
    tis-ie's fast_jl/BasicProjector, CountSketch's cost is independent of
    proj_dim (one scatter_add over num_params regardless of output size),
    so there's no cheap-vs-expensive tradeoff distinguishing "LESS" from
    "ground truth" in this repo the way there is in tis-ie's benchmark --
    they're the same computation at whatever proj_dim you pick.

  * semantic/embedding is untrained bi-encoder cosine similarity -- exactly
    run.py's pre-distillation "baseline" eval. `embedding_scores` wraps
    influcoder.encoder for the same reason.

RDS+, LoGRA (raw + FIM), and TF-IDF are genuinely new methods, ported from:
  tis-ie/influence_eval/compute_rdsplus_scores.py
  tis-ie/logra/less/utils/modeling_logra.py  (via influcoder/logra.py)
  tis-ie/influence_eval/compute_tfidf_scores.py
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .data import Sample
from .encoder import embed, load_encoder
from .gradients import GradientFeaturizer, hardware_profile


# ============================================================================
# LESS -- thin wrapper, no new algorithm (see module docstring)
# ============================================================================

def less_scores(anchors: List[Sample], pool: List[Sample], model_name: str,
                lora_rank: int = 8, lora_seed: int = 0, proj_dim: int = 2048,
                proj_seed: int = 43, max_len: int = 1024) -> np.ndarray:
    """LoRA-gradient cosine similarity, CountSketch-projected. Identical
    machinery to influcoder.gradients.GradientFeaturizer, which is what
    run.py already uses for ground truth -- so proj_dim/proj_seed default to
    values DIFFERENT from GradientFeaturizer's own defaults (8192/42). Using
    the same values would make this method byte-identical to GT by
    construction (same model, same LoRA seed, same sketch), which tests
    nothing; a smaller, differently-seeded sketch is a genuine (if, in this
    repo, not cost-motivated -- see module docstring) cheaper-approximation
    comparison, in the same spirit as gradients.projection_fidelity."""
    feat = GradientFeaturizer(model_name, lora_rank=lora_rank, lora_seed=lora_seed,
                              proj_dim=proj_dim, proj_seed=proj_seed, max_len=max_len)
    g_a, _ = feat.features(anchors, "LESS anchors")
    g_p, _ = feat.features(pool, "LESS pool")
    feat.close()
    return (g_a @ g_p.T).numpy()


# ============================================================================
# Semantic / embedding -- thin wrapper, no new algorithm (see module docstring)
# ============================================================================

def embedding_scores(anchors: List[Sample], pool: List[Sample],
                     encoder_model: str = "jhu-clsp/ettin-encoder-68m") -> np.ndarray:
    """Untrained sentence-encoder cosine similarity over Sample.text --
    exactly run.py's pre-distillation baseline eval, wrapped as a named
    baseline for direct comparison against the other four."""
    enc = load_encoder(encoder_model)
    a = embed(enc, [s.text for s in anchors])
    p = embed(enc, [s.text for s in pool])
    return a @ p.T


# ============================================================================
# TF-IDF -- ported from tis-ie/influence_eval/compute_tfidf_scores.py
# ============================================================================

def tfidf_scores(anchors: List[Sample], pool: List[Sample]) -> np.ndarray:
    """TF-IDF cosine similarity over Sample.text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    anchor_texts = [s.text for s in anchors]
    pool_texts = [s.text for s in pool]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=100000)
    pool_tfidf = vectorizer.fit_transform(pool_texts)
    anchor_tfidf = vectorizer.transform(anchor_texts)
    return linear_kernel(anchor_tfidf, pool_tfidf)


# ============================================================================
# RDS+ -- ported from tis-ie/influence_eval/compute_rdsplus_scores.py
# ============================================================================

@torch.no_grad()
def _rdsplus_embed(model, tokenizer, samples: List[Sample], device: str,
                   batch_size: int, max_len: int) -> torch.Tensor:
    """SGPT-style weighted-mean pooling of the last hidden state (Muennighoff
    2022, https://arxiv.org/abs/2202.08904). Ported from
    compute_rdsplus_scores.py:get_rdsplus_embeddings. tis-ie always runs this
    at batch_size=1 so padding never comes up there; the attention_mask
    multiply below is added so batch_size>1 pools correctly here too --
    it's a no-op at batch_size=1, identical to the source in that case."""
    all_embeds = []
    for i in tqdm(range(0, len(samples), batch_size), desc="RDS+"):
        batch = samples[i:i + batch_size]
        ids_list = [
            torch.tensor(tokenizer(s.context + s.target, add_special_tokens=False)
                        .input_ids[:max_len])
            for s in batch
        ]
        input_ids = pad_sequence(ids_list, batch_first=True,
                                 padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        weighting_mask = torch.arange(hidden_states.size(1), device=device).unsqueeze(0) + 1
        weighting_mask = weighting_mask / weighting_mask.sum(dim=1, keepdim=True)
        mask = attention_mask.unsqueeze(-1)

        embeddings = torch.sum(hidden_states * weighting_mask.unsqueeze(-1) * mask, dim=1)
        embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)
        all_embeds.append(embeddings.cpu())

    return torch.cat(all_embeds, dim=0)


def rdsplus_scores(anchors: List[Sample], pool: List[Sample], model_name: str,
                   batch_size: int = 1, max_len: int = 1024,
                   device: str = "cuda") -> np.ndarray:
    """Forward-pass-only cosine similarity of weighted-mean-pooled hidden
    states -- no backward pass, no LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype, attn = hardware_profile(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attn
    ).to(device)
    model.eval()

    a = _rdsplus_embed(model, tokenizer, anchors, device, batch_size, max_len)
    p = _rdsplus_embed(model, tokenizer, pool, device, batch_size, max_len)
    del model
    torch.cuda.empty_cache()
    return (a @ p.T).numpy()


# ============================================================================
# LoGRA -- ported from tis-ie/logra/less/utils/modeling_logra.py (see logra.py)
# ============================================================================

def logra_scores(anchors: List[Sample], pool: List[Sample], model_name: str,
                 rank: int = 8, mlp_only: bool = True, target_modules=None,
                 batch_size: int = 1, max_len: int = 1024,
                 device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Returns (raw_scores, fim_scores). Matches compute_logra_scores.py's
    dual-variant flow: pool encoded once (accumulating FIM), anchors encoded
    once (raw gradients, is_test=False) -- FIM preconditioning is then
    applied to those same anchor embeddings after the fact, so the anchor
    forward/backward pass is never repeated."""
    from .logra import LoGra

    logra = LoGra(model_name, rank=rank, mlp_only=mlp_only,
                  target_modules=target_modules, max_len=max_len, device=device)

    pool_embeds = logra.encode(pool, batch_size=batch_size, is_test=False)
    pool_fim = logra.fim

    raw_anchor_embeds = logra.encode(anchors, batch_size=batch_size, is_test=False)
    raw_scores = logra.similarity(raw_anchor_embeds, pool_embeds, mode="cosine")

    logra.fim = pool_fim
    precond_anchor_embeds = logra._precondition(torch.from_numpy(raw_anchor_embeds)).float().numpy()
    fim_scores = logra.similarity(precond_anchor_embeds, pool_embeds, mode="cosine")

    logra.close()
    return raw_scores, fim_scores
