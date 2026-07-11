"""Compute (num_anchors, num_train) score matrix from gradient-based influence.

Used for BOTH:
  - the ground-truth matrix (high proj_dim, e.g. 65536)
  - the LESS-method matrix (proj_dim=8192)

The only differences between the two cases are `--proj_dim` and `--out_name`.
Everything else (model, LoRA seed, dataset slices, gradient type) is held
constant so the two matrices are directly comparable.

Plain SGD gradients throughout (no Adam state because there is no warmup
checkpoint). LoRA-only: gradients are taken only on the LoRA adapter
parameters, which is what gets trained during SFT.

Data: reads ONLY from pre-tokenized HF datasets on disk (produced by
prepare_data.py).  No internal data loading or tokenization.
"""

import argparse
import logging
import os
from typing import Tuple

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from influence_eval.flops_measure import flop_counter
from influence_eval.model_utils import count_params, load_base_with_fresh_lora
from representation.helper import batch_cosine_similarity
from representation.less.compute_less_embeds import (
    collect_grads,
    normalize_embeddings_in_chunks,
)

logger = logging.getLogger(__name__)


def _require_path(path: str, kind: str) -> None:
    if not (path and os.path.exists(path)):
        raise FileNotFoundError(
            f"❌ {kind} not found at {path!r}.  "
            f"Run `bash runs/influence_spearman/prepare_data.sh` first."
        )


def _load_tokenized(path: str):
    ds = load_from_disk(path)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def _collect_and_normalize(
    dataloader,
    model,
    proj_dim: int,
    project_interval: int,
):
    """Returns (normalized_grads, uses_custom_cuda).

    uses_custom_cuda indicates whether CudaProjector (fast_jl, NOT counted by
    FlopCounterMode) or BasicProjector (torch.mm, IS counted) was used.
    The caller uses this to decide whether to add projection FLOPs manually.
    """
    grads, uses_custom_cuda = collect_grads(
        dataloader=dataloader,
        model=model,
        proj_dim=proj_dim,
        adam_optimizer_state=None,
        gradient_type="sgd",
        project_interval=project_interval,
    )
    return normalize_embeddings_in_chunks(
        grads, chunk_size=10000, dim=1, eps=1e-12, in_place=False
    ), uses_custom_cuda


def compute_scores(
    model,
    save_dir: str,
    out_name: str,
    tokenized_train_path: str,
    tokenized_anchor_path: str,
    proj_dim: int,
    project_interval: int,
    save_grads: bool,
) -> Tuple[str, dict]:
    os.makedirs(save_dir, exist_ok=True)

    _require_path(tokenized_train_path, "tokenized train dataset")
    _require_path(tokenized_anchor_path, "tokenized anchor dataset")

    train_ds = _load_tokenized(tokenized_train_path)
    anchor_ds = _load_tokenized(tokenized_anchor_path)
    logger.info("Loaded train_ds  (%d samples) ← %s", len(train_ds),  tokenized_train_path)
    logger.info("Loaded anchor_ds (%d samples) ← %s", len(anchor_ds), tokenized_anchor_path)

    train_dl  = torch.utils.data.DataLoader(train_ds,  batch_size=1, shuffle=False)
    anchor_dl = torch.utils.data.DataLoader(anchor_ds, batch_size=1, shuffle=False)

    logger.info("Computing %d train gradients (proj_dim=%d)", len(train_ds), proj_dim)
    logger.info("Computing %d anchor gradients (proj_dim=%d)", len(anchor_ds), proj_dim)

    import time
    t0 = time.perf_counter()
    with flop_counter() as counter:
        train_grads,  cuda_proj_train  = _collect_and_normalize(train_dl,  model, proj_dim, project_interval)
        anchor_grads, cuda_proj_anchor = _collect_and_normalize(anchor_dl, model, proj_dim, project_interval)
        scores = batch_cosine_similarity(
            dev_reps=anchor_grads,
            train_reps=train_grads,
            chunk_size=256,
            normalize=False,
        ).float()
    # Both calls use the same device/fast_jl availability, so the flag is the same.
    uses_custom_cuda = cuda_proj_train
    inference_time_s = time.perf_counter() - t0
    n_samples = len(anchor_ds) + len(train_ds)
    time_per_sample_s = inference_time_s / max(n_samples, 1)

    # TRAK CudaProjector (fast_jl custom CUDA kernel) is NOT counted by FlopCounterMode.
    # BasicProjector (torch.mm) IS counted — adding manually would double-count.
    # Only add analytically when CudaProjector was actually used.
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fwdbwd_flops = int(counter.get_total_flops())
    proj_flops   = 2 * lora_params * proj_dim * n_samples if uses_custom_cuda else 0
    measured_flops = fwdbwd_flops + proj_flops
    logger.info(
        "Projector: %s | FLOPs: %.3e (fwd+bwd) + %.3e (projection) = %.3e total | "
        "Wall-clock: %.2fs (%.2fms/sample)",
        "CudaProjector" if uses_custom_cuda else "BasicProjector",
        fwdbwd_flops, proj_flops, measured_flops,
        inference_time_s, 1000 * time_per_sample_s,
    )

    if save_grads:
        torch.save(train_grads,  os.path.join(save_dir, f"{out_name}_train_grads.pt"))
        torch.save(anchor_grads, os.path.join(save_dir, f"{out_name}_anchor_grads.pt"))

    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    meta = {
        "num_anchors":      int(scores.shape[0]),
        "num_train":        int(scores.shape[1]),
        "proj_dim":         int(proj_dim),
        "measured_flops":   measured_flops,
        "inference_time_s": float(inference_time_s),
        "time_per_sample_s": float(time_per_sample_s),
    }
    return out_path, meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--save_dir",   required=True, type=str)
    p.add_argument("--out_name",   required=True, type=str,
                   help="Prefix for output artifacts (e.g. 'ground_truth', 'less').")
    p.add_argument("--tokenized_train_path",  required=True,
                   help="Pre-tokenized HF dataset (input_ids/attention_mask/labels) from prepare_data.py.")
    p.add_argument("--tokenized_anchor_path", required=True,
                   help="Pre-tokenized HF anchor dataset from prepare_data.py.")
    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--lora_rank",    type=int,   default=128)
    p.add_argument("--lora_alpha",   type=int,   default=512)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_seed",    type=int,   default=0)
    p.add_argument("--project_interval", type=int, default=8)
    p.add_argument("--save_grads", action="store_true")
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

    out_path, meta = compute_scores(
        model=model,
        save_dir=args.save_dir,
        out_name=args.out_name,
        tokenized_train_path=args.tokenized_train_path,
        tokenized_anchor_path=args.tokenized_anchor_path,
        proj_dim=args.proj_dim,
        project_interval=args.project_interval,
        save_grads=args.save_grads,
    )

    # Save param counts alongside scores so FLOPS can read them later
    params = count_params(model)
    params_path = os.path.join(args.save_dir, f"{args.out_name}_params.pt")
    torch.save({**params, **meta, "model_name": args.model_name}, params_path)
    logger.info("Saved param/meta info: %s", params_path)


if __name__ == "__main__":
    main()
