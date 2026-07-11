"""Centralized data preparation for the influence-Spearman pipeline.

Produces all data files every downstream method reads.  No method should
load or tokenize data itself — everything comes from `${OUTPUT_DIR}/data/`.

For each split we write two artifacts:
  1. A HuggingFace dataset (saved with `save_to_disk`) containing
     {input_ids, attention_mask, labels} — used by gradient-based methods
     (GT, LESS, logra, IProX, influcoder stocking).
  2. A `<split>_texts.json` list[str] — used by sentence-transformer methods
     (embedding, influcoder scoring).

Both formats come from the SAME shuffled+sliced rows, guaranteeing every
method sees byte-identical underlying samples.

Determinism: both dolly and BBH are explicitly shuffled with seed=42 BEFORE
splitting so source ordering never affects results.
"""

import argparse
import json
import logging
import os
import sys
from typing import List

from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Make sibling packages importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.data import construct_test_sample, encode_with_messages_format
from influence_eval.bbh_data import load_bbh_samples

logger = logging.getLogger(__name__)

_ENCODER_PREFIX = (
    "Instruct: Given a sample, find the passages closest to that sample.\nQuery:"
)


def _dolly_text_from_messages(messages: list, eos: str) -> str:
    """Chat-template format used by sentence-transformer pool encoders.
    Matches `pool_dicts_to_texts` in influcoder.train_influence_encoder.
    """
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "").strip()
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}{eos}\n")
    return "".join(parts).strip()


def _build_pool_split(dolly_subset, tokenizer, max_seq_length: int) -> tuple:
    """Tokenize a dolly slice + produce sentence-transformer texts + input dicts.

    Returns (HFDataset[torch, only input_ids/attention_mask/labels], texts, input_dicts).
    """
    eos = tokenizer.eos_token or ""
    # Capture raw dicts and texts BEFORE tokenization since map drops the messages col.
    input_dicts = [{"messages": list(ex["messages"])} for ex in dolly_subset]
    texts = [_dolly_text_from_messages(d["messages"], eos) for d in input_dicts]

    ds = dolly_subset.map(
        lambda x: encode_with_messages_format(
            example=x,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            include_response=True,
        ),
        num_proc=1,
        load_from_cache_file=False,
    )
    # Drop all original cols; keep only the three needed for gradient methods.
    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format(type="torch", columns=keep)

    return ds, texts, input_dicts


def _build_anchor_split(bbh_subset: list, tokenizer, max_seq_length: int) -> tuple:
    """Tokenize a BBH slice + produce sentence-transformer texts + input dicts.

    `bbh_subset` is a list of {"prompt", "response", "task"} from load_bbh_samples.
    Returns (HFDataset[torch], texts, input_dicts).
    """
    input_dicts = [{"prompts": s["prompt"], "labels": s["response"]} for s in bbh_subset]
    texts = [f"{_ENCODER_PREFIX} {s['prompt']} {s['response']}".strip() for s in bbh_subset]

    ds = HFDataset.from_list(input_dicts)
    ds = ds.map(
        lambda x: construct_test_sample(
            tokenizer=tokenizer, sample=x, max_length=max_seq_length
        ),
        num_proc=1,
        load_from_cache_file=False,
    )
    keep = ["input_ids", "attention_mask", "labels"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format(type="torch", columns=keep)

    return ds, texts, input_dicts


def _save_split(ds: HFDataset, texts: List[str], input_dicts: list, data_dir: str, name: str) -> None:
    """Save HF dataset to `data_dir/name/`, texts to `<name>_texts.json`, raw dicts to `<name>_inputs.json`."""
    ds_path = os.path.join(data_dir, name)
    txt_path = os.path.join(data_dir, f"{name}_texts.json")
    inp_path = os.path.join(data_dir, f"{name}_inputs.json")
    import shutil
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)
    ds.save_to_disk(ds_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    with open(inp_path, "w", encoding="utf-8") as f:
        json.dump(input_dicts, f, ensure_ascii=False)
    logger.info("✓ %s — %d samples → %s + %s + %s", name, len(ds), ds_path, txt_path, inp_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dolly_path", required=True, help="Local JSONL path with {messages: [...]} per line.")
    p.add_argument("--gradient_model", required=True, help="HF model name; used for the tokenizer.")
    p.add_argument("--output_dir", required=True, help="Files go to {output_dir}/data/.")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--shuffle_seed", type=int, default=42)

    # Sizes
    p.add_argument("--end_index", type=int, required=True, help="Spearman eval pool size (dolly).")
    p.add_argument("--num_anchors", type=int, required=True, help="Spearman eval anchor count (BBH).")
    p.add_argument("--iprox_n_train_p", type=int, required=True)
    p.add_argument("--iprox_n_train_a", type=int, required=True)
    p.add_argument("--influcoder_n_train_p", type=int, required=True)
    p.add_argument("--influcoder_n_train_a", type=int, required=True)
    p.add_argument("--influcoder_n_eval_p", type=int, required=True)
    p.add_argument("--influcoder_n_eval_a", type=int, required=True)
    args = p.parse_args()

    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    logger.info("🔤 Loading tokenizer: %s", args.gradient_model)
    tokenizer = AutoTokenizer.from_pretrained(args.gradient_model)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load + shuffle dolly ───────────────────────────────────────────────
    logger.info("📂 Loading dolly: %s", args.dolly_path)
    dolly = load_dataset("json", data_files=[args.dolly_path])["train"]
    logger.info("   raw size: %d", len(dolly))
    dolly = dolly.shuffle(seed=args.shuffle_seed)
    logger.info("   shuffled with seed=%d", args.shuffle_seed)

    # ── Load BBH (already seed=42 shuffled internally) ─────────────────────
    bbh_full = load_bbh_samples(n_samples=None, start_index=0)
    logger.info("📂 Loaded BBH: %d samples (seed=42 shuffled internally)", len(bbh_full))

    # ── Compute disjoint ranges ────────────────────────────────────────────
    # Dolly axis
    dp = {}
    dp["eval_pool"]              = (0,                                                          args.end_index)
    dp["iprox_train_pool"]       = (dp["eval_pool"][1],                                          dp["eval_pool"][1] + args.iprox_n_train_p)
    dp["influcoder_train_pool"]  = (dp["iprox_train_pool"][1],                                   dp["iprox_train_pool"][1] + args.influcoder_n_train_p)
    dp["influcoder_eval_pool"]   = (dp["influcoder_train_pool"][1],                              dp["influcoder_train_pool"][1] + args.influcoder_n_eval_p)

    # BBH axis
    ba = {}
    ba["eval_anchors"]              = (0,                                                       args.num_anchors)
    ba["iprox_train_anchors"]       = (ba["eval_anchors"][1],                                    ba["eval_anchors"][1] + args.iprox_n_train_a)
    ba["influcoder_train_anchors"]  = (ba["iprox_train_anchors"][1],                             ba["iprox_train_anchors"][1] + args.influcoder_n_train_a)
    ba["influcoder_eval_anchors"]   = (ba["influcoder_train_anchors"][1],                        ba["influcoder_train_anchors"][1] + args.influcoder_n_eval_a)

    # Sanity check sizes
    if dp["influcoder_eval_pool"][1] > len(dolly):
        raise ValueError(
            f"Requested dolly slices total {dp['influcoder_eval_pool'][1]} samples but dolly has {len(dolly)}"
        )
    if ba["influcoder_eval_anchors"][1] > len(bbh_full):
        raise ValueError(
            f"Requested BBH slices total {ba['influcoder_eval_anchors'][1]} samples but BBH has {len(bbh_full)}"
        )

    # ── Build + save each split ────────────────────────────────────────────
    logger.info("⚙️  Building splits...")
    splits_info = {}

    for name, (start, end) in dp.items():
        subset = dolly.select(range(start, end))
        ds, texts, inputs = _build_pool_split(subset, tokenizer, args.max_seq_length)
        _save_split(ds, texts, inputs, data_dir, name)
        splits_info[name] = {"source": "dolly", "range": [start, end], "size": end - start}

    for name, (start, end) in ba.items():
        subset = bbh_full[start:end]
        ds, texts, inputs = _build_anchor_split(subset, tokenizer, args.max_seq_length)
        _save_split(ds, texts, inputs, data_dir, name)
        splits_info[name] = {"source": "bbh", "range": [start, end], "size": end - start}

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "source": {
            "dolly_path": args.dolly_path,
            "dolly_total_samples": len(dolly),
            "bbh_total_samples": len(bbh_full),
        },
        "shuffle": {
            "dolly_seed": args.shuffle_seed,
            "bbh_seed": 42,  # hard-coded in load_bbh_samples
        },
        "tokenizer": args.gradient_model,
        "max_seq_length": args.max_seq_length,
        "splits": splits_info,
    }
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("📝 Manifest → %s", manifest_path)
    logger.info("✅ Done. All data in %s", data_dir)


if __name__ == "__main__":
    main()
