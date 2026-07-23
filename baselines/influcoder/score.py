"""InfluCoder itself, scored through the same harness as the baselines.

Loads the distilled encoder saved by `run.py` (runs_out/<preset>/encoder) and
scores the eval split with it. Method-wise this is identical to the `semantic`
baseline -- same architecture, same single forward pass, no gradients -- the
only difference is the weights, which is exactly the claim: influcoder buys its
accuracy at distillation time, not at selection time. Running both through this
harness makes their per-sample inference cost directly comparable.

Attention is pinned to eager (as in `semantic`) so the FLOP counter can see the
attention matmuls; run.py's `load_encoder` picks SDPA on Ampere+, which would
make the two rows incomparable under measurement.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def score_influcoder(splits, encoder_dir: str, max_len: int = 512,
                     batch_size: int = 32, meter=None) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    path = Path(encoder_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"trained encoder not found at {path}. Run `python run.py --preset "
            f"<preset>` first so it is written to runs_out/<preset>/encoder."
        )

    model = SentenceTransformer(str(path),
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
