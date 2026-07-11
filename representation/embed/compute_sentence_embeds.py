"""Sentence-transformer encoding helpers.

This module does NOT load or tokenize data.  Callers should read pre-built text
files (produced by `influence_eval.prepare_data`) and pass the text list to
`encode_texts`.
"""

import json
import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def l2_normalize_batch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2-normalize a 2D tensor.  [N, D] → [N, D]."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    return F.normalize(x, p=2, dim=1, eps=eps)


def encode_texts(
    model,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode a list of strings with a SentenceTransformer-compatible model.

    Returns a [N, D] torch.Tensor (L2-normalized by default).
    """
    if len(texts) == 0:
        raise ValueError("encode_texts called with empty list")
    logger.info("Encoding %d texts with model on %s ...", len(texts), getattr(model, "device", "?"))
    arr: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    out = torch.from_numpy(arr)
    if normalize:
        out = l2_normalize_batch(out)
    return out


def load_texts(path: str) -> List[str]:
    """Read a JSON list[str] file produced by prepare_data.py."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Text file not found at {path!r}.  "
            f"Run `bash runs/influence_spearman/prepare_data.sh` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    if not isinstance(texts, list):
        raise ValueError(f"{path!r} did not contain a list[str]")
    return texts


def encode_texts_from_file(
    model,
    texts_path: str,
    batch_size: int = 32,
    normalize: bool = True,
) -> torch.Tensor:
    """Load texts from a JSON file and encode them.  Convenience wrapper."""
    texts = load_texts(texts_path)
    return encode_texts(model, texts, batch_size=batch_size, normalize=normalize)
