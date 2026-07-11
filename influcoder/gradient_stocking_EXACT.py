"""Gradient stocking that matches compute_gradient_scores.py EXACTLY.

The only difference from compute_gradient_scores.py is the projection: this
script uses a sparse CountSketch projection (sign-hash, scatter_add) instead
of TRAK's dense Rademacher CudaProjector.  Per-data-type formatting is
identical to the final-pipeline GT:
  - BBH anchors → construct_test_sample (raw prompt+label tokenization)
  - Tulu pool   → encode_with_messages_format (chat template)

Because both this script and compute_gradient_scores.py use the same
load_anchor_dataset/load_train_dataset, the same fresh-LoRA model, and the
same obtain_gradients() loop, the stocked CountSketch labels and the
final-pipeline GT labels are projections of the same gradient vector.
They differ only in projection dimension.

Output per split (file-based, replaces SQLite):
  {output_dir}/{output_name}_grads.pt    — [N, proj_dim] float32, L2-normalized
  {output_dir}/{output_name}_inputs.json — list of input dicts in the same order
  {output_dir}/{output_name}_meta.json   — projection/dataset metadata

Inputs JSON shape:
  anchors: {"prompts": str, "labels": str}                    → construct_test_sample
  pool:    {"messages": [{"role": ..., "content": ...}, ...]} → encode_with_messages_format
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

# Make sibling packages importable when run as a script
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time
from influence_eval.flops_measure import flop_counter, add_phase_flops, add_phase_timing
from influence_eval.model_utils import load_base_with_fresh_lora
from representation.less.compute_less_embeds import (
    get_number_of_trainable_params,
    normalize_embeddings_in_chunks,
    obtain_gradients,
    prepare_batch,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CountSketch projection (replaces TRAK CudaProjector)
# =============================================================================

class CountSketchProjector:
    """Sparse sign-hash (CountSketch) projection.

    For each parameter i, draws (hash_idx_i, sign_i) once from a seeded RNG.
    Project: out[:, hash_idx_i] += sign_i * grad[:, i] for each i.

    Done in chunks of `chunk_size` parameters to keep GPU memory bounded
    regardless of total parameter count.  Output dtype is float32 to match
    compute_gradient_scores.py's normalize step.
    """

    def __init__(
        self,
        num_params: int,
        proj_dim: int,
        seed: int,
        device: torch.device,
        chunk_size: int = 1_000_000,
    ):
        self.num_params = int(num_params)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.device = device
        self.chunk_size = int(chunk_size)

    @torch.no_grad()
    def project(self, grads: torch.Tensor) -> torch.Tensor:
        """grads: [N, num_params] or [num_params]. Returns [N, proj_dim] float32 (on self.device)."""
        single = (grads.dim() == 1)
        if single:
            grads = grads.unsqueeze(0)
        n = grads.shape[0]
        assert grads.shape[1] == self.num_params, (
            f"Expected {self.num_params} params, got {grads.shape[1]}"
        )

        out = torch.zeros((n, self.proj_dim), device=self.device, dtype=torch.float32)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        for start in range(0, self.num_params, self.chunk_size):
            end = min(start + self.chunk_size, self.num_params)
            k = end - start
            hash_idx = torch.randint(
                0, self.proj_dim, (k,), generator=gen, device=self.device
            )
            signs = (
                torch.randint(0, 2, (k,), generator=gen, device=self.device, dtype=torch.float32)
                .mul_(2)
                .sub_(1)
            )
            signed = grads[:, start:end].float() * signs.unsqueeze(0)
            out.scatter_add_(1, hash_idx.unsqueeze(0).expand(n, -1), signed)

        return out.squeeze(0) if single else out


# =============================================================================
# Gradient collection — mirrors collect_grads() in compute_less_embeds.py
# but uses CountSketchProjector instead of TRAK CudaProjector.
# =============================================================================

def collect_grads_countsketch(
    dataloader,
    model,
    proj_dim: int,
    proj_seed: int,
    project_interval: int = 8,
) -> torch.Tensor:
    """SGD gradients (no Adam state); same loop pattern as compute_less_embeds.collect_grads."""
    device = next(model.parameters()).device

    num_params = get_number_of_trainable_params(model)
    logger.info("Projecting %d parameters per sample (CountSketch d=%d, seed=%d)",
                num_params, proj_dim, proj_seed)
    projector = CountSketchProjector(
        num_params=num_params, proj_dim=proj_dim, seed=proj_seed, device=device
    )

    full_grads: List[torch.Tensor] = []
    projected_grads: List[torch.Tensor] = []
    model.train()
    count = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc="Collecting grads"):
        count += 1
        prepare_batch(batch, device=device)
        vectorized_grads = obtain_gradients(model, batch)
        full_grads.append(vectorized_grads.detach())
        del vectorized_grads
        model.zero_grad()

        if count % project_interval == 0:
            stacked = torch.stack(full_grads).to(torch.float16)
            projected_grads.append(projector.project(stacked).cpu())
            full_grads = []
            torch.cuda.empty_cache()

    if len(full_grads) > 0:
        stacked = torch.stack(full_grads).to(torch.float16)
        projected_grads.append(projector.project(stacked).cpu())
        full_grads = []

    return torch.cat(projected_grads, dim=0)


# =============================================================================
# Per-split dispatch
# =============================================================================

_ANCHOR_SPLITS = {"train_anchors", "eval_anchors"}
_POOL_SPLITS = {"pool", "eval_pool"}


def stock_split(
    split: str,
    model,
    proj_dim: int,
    proj_seed: int,
    project_interval: int,
    output_dir: str,
    output_name: str,
    tokenized_input_path: str,
    inputs_json_path: str,
) -> str:
    """Run one split's stocking from pre-tokenized data on disk.

    `tokenized_input_path` — HF dataset (input_ids/attention_mask/labels), from prepare_data.py.
    `inputs_json_path`     — raw input_dicts JSON for downstream encoder training.
    Returns the saved grads path.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(tokenized_input_path):
        raise FileNotFoundError(
            f"❌ Tokenized input dataset not found at {tokenized_input_path!r}.  "
            f"Run `bash runs/influence_spearman/prepare_data.sh` first."
        )
    if not os.path.exists(inputs_json_path):
        raise FileNotFoundError(
            f"❌ Inputs JSON not found at {inputs_json_path!r}.  "
            f"Run `bash runs/influence_spearman/prepare_data.sh` first."
        )

    ds = load_from_disk(tokenized_input_path)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    with open(inputs_json_path, "r", encoding="utf-8") as f:
        input_dicts = json.load(f)
    if len(input_dicts) != len(ds):
        raise ValueError(
            f"Length mismatch: {len(input_dicts)} inputs vs {len(ds)} tokenized samples in {output_name}"
        )

    formatter = "construct_test_sample" if split in _ANCHOR_SPLITS else "encode_with_messages_format"
    dataset = "bbh" if split in _ANCHOR_SPLITS else "dolly"

    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    t0 = time.perf_counter()
    with flop_counter() as counter:
        grads = collect_grads_countsketch(
            dataloader=dataloader,
            model=model,
            proj_dim=proj_dim,
            proj_seed=proj_seed,
            project_interval=project_interval,
        )
        grads = normalize_embeddings_in_chunks(
            grads, chunk_size=10000, dim=1, eps=1e-12, in_place=False
        )
    split_time_s = time.perf_counter() - t0
    fwdbwd_flops = int(counter.get_total_flops())

    # CountSketch projection uses scatter_add_ (custom indexing op), not visible
    # to FlopCounterMode.  Add analytically: per sample, multiply each of P_lora
    # params by ±1 and scatter-add into one of proj_dim bins → 2 × P_lora ops
    # per sample (one multiply, one accumulate).
    num_params = get_number_of_trainable_params(model)
    n_samples  = int(grads.shape[0])
    proj_flops = 2 * num_params * n_samples
    split_flops = fwdbwd_flops + proj_flops
    logger.info(
        "Split FLOPs: %.3e (fwd+bwd) + %.3e (CountSketch) = %.3e total",
        fwdbwd_flops, proj_flops, split_flops,
    )

    # Save
    grads_path = os.path.join(output_dir, f"{output_name}_grads.pt")
    torch.save(grads, grads_path)

    inputs_path = os.path.join(output_dir, f"{output_name}_inputs.json")
    with open(inputs_path, "w", encoding="utf-8") as f:
        json.dump(input_dicts, f, ensure_ascii=False)

    time_per_sample_s = split_time_s / max(grads.shape[0], 1)
    meta = {
        "proj_dim": int(proj_dim),
        "proj_seed": int(proj_seed),
        "n_samples": int(grads.shape[0]),
        "split": split,
        "dataset": dataset,
        "tokenized_input_path": tokenized_input_path,
        "formatter": formatter,
        "time_s": float(split_time_s),
        "time_per_sample_s": float(time_per_sample_s),
    }
    meta_path = os.path.join(output_dir, f"{output_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Accumulate FLOPs and wall-clock time across splits
    flops_json = os.path.join(output_dir, "_flops.json")
    timing_json = os.path.join(output_dir, "_timing.json")
    total_flops = add_phase_flops(flops_json, split_flops)
    total_time_s = add_phase_timing(timing_json, split_time_s)
    logger.info(
        "FLOPs this split: %.3e | cumulative: %.3e | time: %.2fs (%.2fms/sample)",
        split_flops, total_flops, split_time_s, 1000 * time_per_sample_s,
    )

    logger.info("Saved %s (grads %s, inputs %s, meta %s)", split, grads.shape, len(input_dicts), meta)
    return grads_path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=sorted(_ANCHOR_SPLITS | _POOL_SPLITS))
    p.add_argument("--tokenized_input_path", required=True,
                   help="Path to HF dataset saved by prepare_data.py (e.g. data/influcoder_train_anchors/).")
    p.add_argument("--inputs_json_path", required=True,
                   help="Path to <name>_inputs.json saved by prepare_data.py.")
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--proj_seed", type=int, default=42)
    p.add_argument("--project_interval", type=int, default=8)
    # LoRA config — match compute_gradient_scores.py defaults
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=512)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_seed", type=int, default=0)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--output_name", required=True, type=str,
                   help="Prefix for output files (e.g. 'train_anchors' → train_anchors_grads.pt).")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_with_fresh_lora(
        model_name=args.model_name,
        tokenizer=tokenizer,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.lora_seed,
    )

    stock_split(
        split=args.split,
        model=model,
        proj_dim=args.proj_dim,
        proj_seed=args.proj_seed,
        project_interval=args.project_interval,
        output_dir=args.output_dir,
        output_name=args.output_name,
        tokenized_input_path=args.tokenized_input_path,
        inputs_json_path=args.inputs_json_path,
    )


if __name__ == "__main__":
    main()
