"""Semantic (sentence-transformer embedding) baseline.

Faithful to influence_eval/compute_embedding_scores.py: encode pool and anchors
with an off-the-shelf SentenceTransformer (the *untrained* encoder), cosine
between them. This is the same encoder influcoder distills into, so this row is
influcoder's "untrained encoder" baseline and its per-sample inference cost is
the floor influcoder is measured against.

max_seq_len defaults to 512 to match run.py's `load_encoder`, and attention is
pinned to eager so the FLOP counter can see the attention matmuls.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def score_semantic(splits, encoder_model: str, max_len: int = 512,
                   batch_size: int = 32, meter=None) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_model,
                                model_kwargs={"attn_implementation": "eager"})
    model.max_seq_length = max_len
    if torch.cuda.is_available():
        model.to("cuda")
    if meter is not None:
        meter.model_ready()

    def embed(samples):
        arr = model.encode([s.text for s in samples], batch_size=batch_size,
                           show_progress_bar=False, convert_to_numpy=True,
                           normalize_embeddings=False)
        return F.normalize(torch.from_numpy(arr).float(), p=2, dim=1)

    pool = embed(splits["eval_pool"])
    anchors = embed(splits["eval_anchors"])
    return (anchors @ pool.T).float()
