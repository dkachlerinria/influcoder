"""Datasets: BBH anchors and Dolly candidates.

Every example is reduced to a Sample with three views:
  context -- conditioning text; no loss is taken on it
  target  -- completion whose loss gradient defines the sample's influence feature
  text    -- compact string the bi-encoder sees

For BBH, the gradient context keeps the full CoT prompt (that is what defines
the influence geometry at eval time), but the encoder text drops it: the CoT
prefix is constant across a task, carries no discriminative signal, and would
eat the encoder's whole window.
"""

from __future__ import annotations

import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path

ANCHOR_PREFIX = (
    "Instruct: Given a sample, find the passages closest to that sample.\nQuery: "
)


@dataclass
class Sample:
    context: str
    target: str
    text: str


def load_bbh(root: Path | str, seed: int) -> list[Sample]:
    """All BBH examples, seeded shuffle. The first two lines of each
    cot-prompts file are a header, not part of the prompt."""
    root = Path(root)
    samples = []
    for task_file in sorted(glob.glob(str(root / "bbh" / "*.json"))):
        name = Path(task_file).stem
        cot_file = root / "cot-prompts" / f"{name}.txt"
        prefix = ""
        if cot_file.exists():
            prefix = "".join(cot_file.read_text().splitlines(keepends=True)[2:]).strip()
        with open(task_file) as f:
            for ex in json.load(f)["examples"]:
                samples.append(Sample(
                    context=f"{prefix}\n\nQ: {ex['input']}\nA:",
                    target=" " + ex["target"],
                    text=ANCHOR_PREFIX + f"Q: {ex['input']}\nA: {ex['target']}",
                ))
    random.Random(seed).shuffle(samples)
    return samples


def load_dolly(path: Path | str, seed: int) -> list[Sample]:
    """Dolly rows (first user turn -> first assistant turn), seeded shuffle."""
    samples = []
    with open(path) as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            user = next((m["content"].strip() for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"].strip() for m in msgs if m["role"] == "assistant"), "")
            if user and asst:
                samples.append(Sample(
                    context=f"<|user|>\n{user}\n<|assistant|>\n",
                    target=asst,
                    text=f"{user}\n{asst}",
                ))
    random.Random(seed).shuffle(samples)
    return samples


def load_fineweb(seed: int, max_docs: int = 600, min_chars: int = 800,
                 max_chars: int = 2000, split_frac: float = 0.6) -> list[Sample]:
    """FineWeb web-text documents, reduced to (context, target) via a
    continuation split rather than an instruction/response split -- there is
    no user/assistant structure here, so "target" is just the tail of the
    document the "context" head precedes. Used as an out-of-distribution
    pool: neither BBH's Q/A format nor Dolly's chat format, so it tests
    whether the trained encoder generalizes past both training text shapes.

    Streams a single fixed parquet shard directly (`data_files=...`) rather
    than letting `datasets` resolve the whole `sample-10BT` config, which
    tries to enumerate ~100 x 2GB files before yielding a single row.

    Returns a fixed-size, seeded-shuffle list like load_bbh/load_dolly --
    callers slice front ranges for eval/train (e.g. via disjoint_splits) and
    get a disjoint set regardless of how many they ultimately take, because
    the scan itself doesn't depend on the caller's requested size.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", data_files="sample/10BT/000_00000.parquet",
                      split="train", streaming=True)
    samples = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < min_chars:
            continue
        text = text[:max_chars]
        cut = text.rfind(" ", 0, int(len(text) * split_frac))
        if cut <= 0:
            continue
        samples.append(Sample(context=text[:cut], target=text[cut:], text=text))
        if len(samples) >= max_docs:
            break
    random.Random(seed).shuffle(samples)
    return samples


def disjoint_splits(anchors: list[Sample], pool: list[Sample],
                    n_eval_a: int, n_train_a: int,
                    n_eval_p: int, n_train_p: int) -> dict[str, list[Sample]]:
    """Front slices for eval, the next slices for training -- never overlapping."""
    if len(anchors) < n_eval_a + n_train_a:
        raise ValueError(f"need {n_eval_a + n_train_a} anchors, have {len(anchors)}")
    if len(pool) < n_eval_p + n_train_p:
        raise ValueError(f"need {n_eval_p + n_train_p} pool samples, have {len(pool)}")
    return {
        "eval_anchors": anchors[:n_eval_a],
        "train_anchors": anchors[n_eval_a:n_eval_a + n_train_a],
        "eval_pool": pool[:n_eval_p],
        "train_pool": pool[n_eval_p:n_eval_p + n_train_p],
    }
