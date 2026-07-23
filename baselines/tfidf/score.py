"""TF-IDF baseline.

Faithful to influence_eval/compute_tfidf_scores.py: fit a TF-IDF vectorizer on
the pool texts, transform the anchors, cosine (linear_kernel on L2-normalized
TF-IDF vectors). Pure lexical overlap, no model.
"""

from __future__ import annotations

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def score_tfidf(splits, meter=None) -> torch.Tensor:
    if meter is not None:
        meter.model_ready()  # no model to load
    pool_texts = [s.text for s in splits["eval_pool"]]
    anchor_texts = [s.text for s in splits["eval_anchors"]]
    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=100000)
    pool_tfidf = vec.fit_transform(pool_texts)
    anchor_tfidf = vec.transform(anchor_texts)
    return torch.from_numpy(linear_kernel(anchor_tfidf, pool_tfidf)).float()
