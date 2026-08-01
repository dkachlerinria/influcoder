#!/usr/bin/env python3
"""Independent leak audit for the leak-free InfluCoder DATE-LM scripts.

EXP2.md originally cited an "independent audit" script (`audit_leakage.py`)
that was never actually committed -- the "0 exact-prompt overlaps, 0 subject
overlaps, 0 exact (prompt,response) pair overlaps" claim had no reproducible
backing. This is that script, written for real.

For each noleak_*.py run, this reconstructs -- via the exact same code paths
those scripts use (imported, not reimplemented) -- precisely which external
rows the run drew for its anchor/candidate/held-out-eval sets, up through the
same valid-loss-span tokenizer filtering, then independently checks those
specific rows against freshly-loaded local train+ref data on three axes:
exact prompt text, subject (Counterfact only, via a prompt->subject map built
directly from the raw external dataset rather than reusing the training
script's internal filtering), and exact (prompt, response) pair.

No GPU or model loading required -- `sample_valid_indices` only needs a
tokenizer, not the teacher model itself, so this runs in a couple of minutes
on CPU (dominated by re-downloading/streaming the external pools).

Usage:
    python methods/audit_leakage.py
"""
from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from datamodules.load_data import get_dataset, prepare_chat_format  # noqa: E402
from methods.influcoder import _bootstrap  # noqa: F401,E402
from methods.influcoder.teacher_grads import sample_valid_indices  # noqa: E402

COUNTERFACT_MODULES = [
    "methods.influcoder_attribute_noleak_counterfact_extquery",
    "methods.influcoder_attribute_noleak_counterfact_extquery_moredata",
]
TOXICITY_MODULES = [
    "methods.influcoder_attribute_noleak_toxicity",
    "methods.influcoder_attribute_noleak_toxicity_moredata",
    "methods.influcoder_attribute_noleak_toxicity_wildguard",
    "methods.influcoder_attribute_noleak_toxicity_wildguard_moredata",
]
# Handled separately below (audit_toxicity_mixed): draws from TWO independent
# toxic pools (build_wildguard_toxic_pool for anchors+eval, a module-specific
# candidate-pool builder -- build_toxicchat_toxic_pool or
# build_beavertails_toxic_pool -- for candidates) instead of the single
# build_external_toxic_pool every other toxicity script uses, so these don't
# fit audit_toxicity()'s generic shape.
TOXICITY_MIXED_MODULES = [
    "methods.influcoder_attribute_noleak_toxicity_mixed",
    "methods.influcoder_attribute_noleak_toxicity_mixed_beavertails",
]


def audit_counterfact(module_path: str, tokenizer):
    m = importlib.import_module(module_path)

    local_train, local_ref = get_dataset(m.TASK, m.SUBSET)
    local_rows = list(local_train) + list(local_ref)
    local_prompts = set(r["prompt"].strip() for r in local_rows)
    local_pairs = set((r["prompt"].strip(), r["response"].strip()) for r in local_rows)

    # Independent prompt->subject map, built directly from the raw external
    # dataset -- NOT reusing build_clean_external_pool's internal filtering.
    ext = load_dataset(m.EXTERNAL_REPO)["train"]
    prompt_to_subjects: dict[str, set[str]] = {}
    for r in ext:
        p = r["prompt"].strip()
        prompt_to_subjects.setdefault(p, set()).add(r["subject"].strip().lower())
    local_subjects = set()
    for r in local_rows:
        local_subjects |= prompt_to_subjects.get(r["prompt"].strip(), set())

    random.seed(m.SEED)
    pool = m.build_clean_external_pool(local_train, local_ref)
    perm = list(range(len(pool)))
    random.Random(m.SEED).shuffle(perm)

    pool_chat = prepare_chat_format(pool)
    ext_anchor_idx = sample_valid_indices(pool_chat, tokenizer, m.MAX_LEN, perm, m.N_EXT_ANCHORS)
    teacher_idx = sample_valid_indices(pool_chat, tokenizer, m.MAX_LEN, perm, m.N_TEACHER_TRAIN,
                                       exclude=set(ext_anchor_idx))
    eval_idx = sample_valid_indices(pool_chat, tokenizer, m.MAX_LEN, perm, m.N_EVAL_TRAIN,
                                    exclude=set(ext_anchor_idx) | set(teacher_idx))
    used_idx = set(ext_anchor_idx) | set(teacher_idx) | set(eval_idx)
    used_rows = [pool[i] for i in used_idx]

    prompt_overlap = [r for r in used_rows if r["prompt"] in local_prompts]
    subject_overlap = [r for r in used_rows if prompt_to_subjects.get(r["prompt"], set()) & local_subjects]
    pair_overlap = [r for r in used_rows if (r["prompt"], r["response"]) in local_pairs]

    print(f"\n== {module_path} ==")
    print(f"  external rows actually drawn (post valid-loss-span filter): {len(used_rows)} "
          f"(anchors={len(ext_anchor_idx)} candidates={len(teacher_idx)} held-out-eval={len(eval_idx)})")
    print(f"  exact-prompt overlaps with local train+ref: {len(prompt_overlap)}")
    print(f"  subject overlaps with local train+ref:      {len(subject_overlap)}")
    print(f"  exact (prompt,response) pair overlaps:      {len(pair_overlap)}")
    return len(prompt_overlap), len(subject_overlap), len(pair_overlap)


def audit_toxicity(module_path: str, tokenizer):
    m = importlib.import_module(module_path)

    local_train, local_ref = get_dataset(m.TASK, m.SUBSET)
    local_rows = list(local_train) + list(local_ref)
    local_benign_prompts = set(r["prompt"].strip() for r in local_train if r["type"] == "Benign")
    local_all_prompts = set(r["prompt"].strip() for r in local_rows)
    local_pairs = set((r["prompt"].strip(), r["response"].strip()) for r in local_rows)

    random.seed(m.SEED)
    n_toxic_needed = m.N_EXT_ANCHORS + m.N_EXT_TOXIC_POS + m.N_EVAL_TOXIC
    n_benign_needed = m.N_EXT_BENIGN + m.N_EVAL_BENIGN
    toxic_pool = m.build_external_toxic_pool(local_all_prompts, n_toxic_needed)
    benign_pool = m.build_external_benign_pool(local_benign_prompts, n_benign_needed)

    random.Random(m.SEED).shuffle(toxic_pool)
    random.Random(m.SEED + 1).shuffle(benign_pool)

    ext_anchors = toxic_pool[:m.N_EXT_ANCHORS]
    toxic_candidates = toxic_pool[m.N_EXT_ANCHORS:m.N_EXT_ANCHORS + m.N_EXT_TOXIC_POS]
    toxic_eval = toxic_pool[m.N_EXT_ANCHORS + m.N_EXT_TOXIC_POS:
                            m.N_EXT_ANCHORS + m.N_EXT_TOXIC_POS + m.N_EVAL_TOXIC]
    benign_candidates = benign_pool[:m.N_EXT_BENIGN]
    benign_eval = benign_pool[m.N_EXT_BENIGN:m.N_EXT_BENIGN + m.N_EVAL_BENIGN]

    candidates = benign_candidates + toxic_candidates
    eval_pool = benign_eval + toxic_eval

    anchor_chat = prepare_chat_format(ext_anchors)
    cand_chat = prepare_chat_format(candidates)
    eval_chat = prepare_chat_format(eval_pool)

    anchor_idx = sample_valid_indices(anchor_chat, tokenizer, m.MAX_LEN, list(range(len(anchor_chat))), len(anchor_chat))
    cand_idx = sample_valid_indices(cand_chat, tokenizer, m.MAX_LEN, list(range(len(cand_chat))), len(cand_chat))
    eval_idx = sample_valid_indices(eval_chat, tokenizer, m.MAX_LEN, list(range(len(eval_chat))), len(eval_chat))

    used_rows = ([ext_anchors[i] for i in anchor_idx]
                 + [candidates[i] for i in cand_idx]
                 + [eval_pool[i] for i in eval_idx])

    prompt_overlap = [r for r in used_rows if r["prompt"].strip() in local_all_prompts]
    pair_overlap = [r for r in used_rows
                    if (r["prompt"].strip(), r["response"].strip()) in local_pairs]

    print(f"\n== {module_path} ==")
    print(f"  external rows actually drawn (post valid-loss-span filter): {len(used_rows)} "
          f"(anchors={len(anchor_idx)} candidates={len(cand_idx)} held-out-eval={len(eval_idx)})")
    print(f"  exact-prompt overlaps with local train+ref: {len(prompt_overlap)}")
    print(f"  exact (prompt,response) pair overlaps:      {len(pair_overlap)}")
    print("  (no subject field for this task -- prompt+pair are the applicable checks)")
    return len(prompt_overlap), len(pair_overlap)


def audit_toxicity_mixed(module_path: str, tokenizer):
    """Cross-source variant: toxic anchors+held-out-eval come from one
    WildGuardMix pool, toxic candidates from an independent second pool
    (ToxicChat or BeaverTails, depending on the module). Mirrors the module's
    own role-assignment logic exactly (same seeds), then independently checks
    the resulting used rows against local data. The candidate-pool builder
    differs by module (`build_toxicchat_toxic_pool(prompts, n)` vs.
    `build_beavertails_toxic_pool(prompts, pairs, n)`) -- called via
    introspection so this one function covers both."""
    m = importlib.import_module(module_path)

    local_train, local_ref = get_dataset(m.TASK, m.SUBSET)
    local_rows = list(local_train) + list(local_ref)
    local_all_prompts = set(r["prompt"].strip() for r in local_rows)
    local_pairs = set((r["prompt"].strip(), r["response"].strip()) for r in local_rows)
    local_benign_prompts = set(r["prompt"].strip() for r in local_train if r["type"] == "Benign")

    random.seed(m.SEED)
    n_wildguard_needed = m.N_EXT_ANCHORS + m.N_EVAL_TOXIC
    wildguard_pool = m.build_wildguard_toxic_pool(local_all_prompts, n_wildguard_needed)
    random.Random(m.SEED).shuffle(wildguard_pool)
    ext_anchors = wildguard_pool[:m.N_EXT_ANCHORS]
    toxic_eval = wildguard_pool[m.N_EXT_ANCHORS:m.N_EXT_ANCHORS + m.N_EVAL_TOXIC]

    if hasattr(m, "build_toxicchat_toxic_pool"):
        candidate_pool = m.build_toxicchat_toxic_pool(local_all_prompts, m.N_EXT_TOXIC_POS)
    elif hasattr(m, "build_beavertails_toxic_pool"):
        candidate_pool = m.build_beavertails_toxic_pool(local_all_prompts, local_pairs, m.N_EXT_TOXIC_POS)
    else:
        raise NotImplementedError(f"{module_path}: no known candidate-pool builder found")
    random.Random(m.SEED + 1).shuffle(candidate_pool)
    toxic_candidates = candidate_pool[:m.N_EXT_TOXIC_POS]

    benign_pool = m.build_external_benign_pool(local_benign_prompts, m.N_EXT_BENIGN + m.N_EVAL_BENIGN)
    random.Random(m.SEED + 2).shuffle(benign_pool)
    benign_candidates = benign_pool[:m.N_EXT_BENIGN]
    benign_eval = benign_pool[m.N_EXT_BENIGN:m.N_EXT_BENIGN + m.N_EVAL_BENIGN]

    candidates = benign_candidates + toxic_candidates
    eval_pool = benign_eval + toxic_eval

    anchor_chat = prepare_chat_format(ext_anchors)
    cand_chat = prepare_chat_format(candidates)
    eval_chat = prepare_chat_format(eval_pool)

    anchor_idx = sample_valid_indices(anchor_chat, tokenizer, m.MAX_LEN, list(range(len(anchor_chat))), len(anchor_chat))
    cand_idx = sample_valid_indices(cand_chat, tokenizer, m.MAX_LEN, list(range(len(cand_chat))), len(cand_chat))
    eval_idx = sample_valid_indices(eval_chat, tokenizer, m.MAX_LEN, list(range(len(eval_chat))), len(eval_chat))

    used_rows = ([ext_anchors[i] for i in anchor_idx]
                 + [candidates[i] for i in cand_idx]
                 + [eval_pool[i] for i in eval_idx])

    prompt_overlap = [r for r in used_rows if r["prompt"].strip() in local_all_prompts]
    pair_overlap = [r for r in used_rows
                    if (r["prompt"].strip(), r["response"].strip()) in local_pairs]

    print(f"\n== {module_path} ==")
    print(f"  external rows actually drawn (post valid-loss-span filter): {len(used_rows)} "
          f"(anchors={len(anchor_idx)} candidates={len(cand_idx)} held-out-eval={len(eval_idx)})")
    print(f"  exact-prompt overlaps with local train+ref: {len(prompt_overlap)}")
    print(f"  exact (prompt,response) pair overlaps:      {len(pair_overlap)}")
    print("  (no subject field for this task -- prompt+pair are the applicable checks)")
    return len(prompt_overlap), len(pair_overlap)


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")

    results = {}
    for mod in COUNTERFACT_MODULES:
        results[mod] = audit_counterfact(mod, tokenizer)
    for mod in TOXICITY_MODULES:
        results[mod] = audit_toxicity(mod, tokenizer)
    for mod in TOXICITY_MIXED_MODULES:
        results[mod] = audit_toxicity_mixed(mod, tokenizer)

    print("\n== summary ==")
    all_clean = True
    for mod, counts in results.items():
        clean = all(c == 0 for c in counts)
        all_clean &= clean
        print(f"  {'CLEAN' if clean else 'LEAK FOUND'}: {mod} {counts}")
    if not all_clean:
        sys.exit(1)


if __name__ == "__main__":
    main()
