#!/usr/bin/env python3
"""Run DATE-LM's Rep-Sim baseline on the Counterfact/Pythia-1b application task,
timed on this GPU for a direct comparison against InfluCoder/Grad_Sim/LESS's
own locally-measured wall-clock (the paper doesn't report timing, only accuracy).

`baselines/repsim.py` (this repo's own vendored Rep-Sim implementation) is
built around jsonl train/val files, not `get_dataset()`'s in-memory Counterfact
split -- this script re-hosts its actual algorithm (last-token hidden state as
the representation, forward-only, no backward pass -- see
`collect_forward_reps`/`compute_forward_similarity` there for the reference
implementation this mirrors) against `get_dataset("Counterfact", "Pythia-1b")`
and the same `checkpoints_load_func` every other method in this doc uses, so
the checkpoint-loading path is identical across all timed methods.

batch_size=1, matching the convention already established for the Grad_Sim/
LESS timing runs on this same task (see EXP2.md's dattri.py timing section).

`methods/model_utils.py` (imported for `checkpoints_load_func`) unconditionally imports
`litgpt`/`lightning` at module level for a sibling checkpoint loader this script never calls --
if those aren't installed, either `pip install litgpt lightning`, or set `PYTHONPATH` to a
directory containing harmless import-only stub packages (`litgpt/__init__.py` empty,
`litgpt/lora.py` with stub `GPT`/`Config` classes and `NotImplementedError`-raising
`lora_filter`/`merge_lora_weights`, `litgpt/utils.py` with a stub `extend_checkpoint_dir`,
`lightning/__init__.py` empty) -- confirmed safe since `checkpoints_load_func`'s HF/peft-based
path never touches any of those symbols.

Usage:
    python methods/run_repsim_counterfact.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datamodules.load_data import get_dataset, prepare_chat_format  # noqa: E402
from datamodules.data_utils import MessageDatasetRepSim  # noqa: E402
from methods.model_utils import checkpoints_load_func  # noqa: E402

TASK = "Counterfact"
SUBSET = "Pythia-1b"
BASE_MODEL = "EleutherAI/pythia-1b"
CHECKPOINT = "DataAttributionEval/Pythia-1b-counterfactual"
MAX_LEN = 1024
BATCH_SIZE = 1
SAVE_PATH = Path("results/factual-attribution-repsim/Rep_Sim.pt")


def get_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [torch.tensor(x["input_ids"]) for x in batch]
        attention_masks = [torch.tensor(x["attention_mask"]) for x in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_masks}
    return collate_fn


def collect_reps(loader, model, desc):
    device = next(model.parameters()).device
    reps = []
    for batch in tqdm(loader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            ids = torch.arange(len(input_ids), device=device)
            pos = attention_mask.sum(dim=1) - 1
            batch_reps = out.hidden_states[-1][ids, pos]
        reps.append(batch_reps.float().cpu())
    return normalize(torch.cat(reps), dim=1)


def main():
    t_start = time.perf_counter()
    print(f"Rep-Sim / DATE-LM | task={TASK} subset={SUBSET} | base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    tokenizer, model = checkpoints_load_func(None, CHECKPOINT, BASE_MODEL)
    model.eval()

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    print(f"local train={n_train} ref={n_ref}\n")

    train_chat = prepare_chat_format(local_train)
    ref_chat = prepare_chat_format(local_ref)

    collate = get_collate_fn(tokenizer)
    train_ds = MessageDatasetRepSim(train_chat, tokenizer=tokenizer, max_length=MAX_LEN)
    ref_ds = MessageDatasetRepSim(ref_chat, tokenizer=tokenizer, max_length=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    ref_loader = DataLoader(ref_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    t_reps_start = time.perf_counter()
    train_reps = collect_reps(train_loader, model, desc="train forward reps")
    ref_reps = collect_reps(ref_loader, model, desc="ref forward reps")
    t_reps_s = time.perf_counter() - t_reps_start
    print(f"\nforward-pass wall time (train+ref): {t_reps_s:.1f}s")

    sim = train_reps @ ref_reps.T  # [n_train, n_ref]
    # Match dattri.py's Counterfact save convention: [n_ref, n_train]-shaped list-of-lists
    final_scores = sim.T.tolist()

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(final_scores, f)

    t_total_s = time.perf_counter() - t_start
    print(f"\nwrote {SAVE_PATH}")
    print(f"TOTAL wall time (model load + forward reps + save): {t_total_s:.1f}s")


if __name__ == "__main__":
    main()
