"""Compute (num_anchors, num_train) score matrix using a trained influence encoder.

Loads the encoder from --encoder_dir (output of train_influence_encoder.py),
then embeds anchor + train text lists prepared by `influence_eval.prepare_data`.
No internal data loading or tokenization.
"""

import argparse
import logging
import os
import time

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM

from influence_eval.flops_measure import flop_counter, load_phase_flops, load_phase_timing
from influence_eval.model_utils import count_params
from representation.embed.compute_sentence_embeds import encode_texts, load_texts
from representation.helper import batch_cosine_similarity

logger = logging.getLogger(__name__)


def _count_gradient_model_params(model_name: str) -> dict:
    """Load gradient model on CPU just to count parameters, then discard."""
    logger.info("Loading gradient model on CPU to count params: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    total = sum(p.numel() for p in model.parameters())
    linear = sum(
        m.weight.numel()
        for m in model.modules()
        if isinstance(m, torch.nn.Linear)
    )
    del model
    return {"grad_model_total": total, "grad_model_linear": linear}


def compute_influcoder_scores(
    encoder_dir: str,
    gradient_model: str,
    save_dir: str,
    train_texts_path: str,
    anchor_texts_path: str,
    n_stock_anchors: int,
    n_stock_pool: int,
    proj_dim: int,
    stocking_flops_path: str,
    training_flops_path: str,
    stocking_timing_path: str = None,
    training_timing_path: str = None,
    batch_size: int = 32,
    out_name: str = "influcoder",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Loading trained influence encoder from: %s", encoder_dir)
    model = SentenceTransformer(
        encoder_dir,
        model_kwargs={"attn_implementation": "eager"},
    )
    if torch.cuda.is_available():
        model.to("cuda")

    anchor_texts = load_texts(anchor_texts_path)
    train_texts  = load_texts(train_texts_path)
    logger.info("anchor_texts: %d ← %s", len(anchor_texts), anchor_texts_path)
    logger.info("train_texts : %d ← %s", len(train_texts),  train_texts_path)

    t0 = time.perf_counter()
    with flop_counter() as counter:
        train_embeds  = encode_texts(model, train_texts,  batch_size=batch_size, normalize=True)
        anchor_embeds = encode_texts(model, anchor_texts, batch_size=batch_size, normalize=False)
        scores = batch_cosine_similarity(
            dev_reps=anchor_embeds,
            train_reps=train_embeds,
            chunk_size=1024,
            normalize=False,
        ).float()
    inference_time_s = time.perf_counter() - t0
    inference_flops  = int(counter.get_total_flops())
    n_samples        = int(anchor_embeds.shape[0] + train_embeds.shape[0])
    inference_time_per_sample_s = inference_time_s / max(n_samples, 1)

    stocking_flops  = load_phase_flops(stocking_flops_path)
    training_flops  = load_phase_flops(training_flops_path)
    total_flops     = stocking_flops + training_flops + inference_flops
    stocking_time_s = load_phase_timing(stocking_timing_path) if stocking_timing_path else 0.0
    training_time_s = load_phase_timing(training_timing_path) if training_timing_path else 0.0
    total_time_s    = stocking_time_s + training_time_s + inference_time_s

    logger.info(
        "FLOPs — stocking=%.3e, training=%.3e, inference=%.3e, total=%.3e",
        stocking_flops, training_flops, inference_flops, total_flops,
    )
    logger.info(
        "Wall-clock — stocking=%.2fs, training=%.2fs, inference=%.2fs (%.2fms/sample), total=%.2fs",
        stocking_time_s, training_time_s, inference_time_s,
        1000 * inference_time_per_sample_s, total_time_s,
    )
    if stocking_flops == 0:
        logger.warning("No stocking FLOPs at %s — was stocking re-run after instrumentation?",
                       stocking_flops_path)
    if training_flops == 0:
        logger.warning("No training FLOPs at %s — was encoder training re-run after instrumentation?",
                       training_flops_path)

    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    encoder_params = count_params(model)
    grad_params    = _count_gradient_model_params(gradient_model)

    meta = {
        **encoder_params,
        **grad_params,
        "emb_dim":           int(train_embeds.shape[1]),
        "num_anchors":       int(scores.shape[0]),
        "num_train":         int(scores.shape[1]),
        "model_name":        encoder_dir,
        "n_stock_anchors":   int(n_stock_anchors),
        "n_stock_pool":      int(n_stock_pool),
        "proj_dim":          int(proj_dim),
        "stocking_flops":    int(stocking_flops),
        "training_flops":    int(training_flops),
        "inference_flops":   int(inference_flops),
        "measured_flops":    int(total_flops),
        "stocking_time_s":   float(stocking_time_s),
        "training_time_s":   float(training_time_s),
        "inference_time_s":  float(inference_time_s),
        "time_per_sample_s": float(inference_time_per_sample_s),
        "measured_time_s":   float(total_time_s),
    }
    torch.save(meta, os.path.join(save_dir, f"{out_name}_params.pt"))
    logger.info("Saved params for FLOPS accounting")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_dir",       required=True, type=str)
    p.add_argument("--gradient_model",    required=True, type=str,
                   help="HF model used for gradient stocking (for FLOPS param count).")
    p.add_argument("--save_dir",          required=True, type=str)
    p.add_argument("--train_texts_path",  required=True, type=str,
                   help="JSON list[str] from prepare_data.py.")
    p.add_argument("--anchor_texts_path", required=True, type=str,
                   help="JSON list[str] from prepare_data.py.")
    p.add_argument("--n_stock_anchors",   type=int, required=True, help="N_TRAIN_A + N_EVAL_A (FLOPS only).")
    p.add_argument("--n_stock_pool",      type=int, required=True, help="N_TRAIN_P + N_EVAL_P (FLOPS only).")
    p.add_argument("--proj_dim",          type=int, required=True, help="INFLUCODER_PROJ_DIM (FLOPS only).")
    p.add_argument("--stocking_flops_path",  required=True, type=str)
    p.add_argument("--training_flops_path",  required=True, type=str)
    p.add_argument("--stocking_timing_path", type=str, default=None)
    p.add_argument("--training_timing_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_name",   type=str, default="influcoder")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    compute_influcoder_scores(
        encoder_dir=args.encoder_dir,
        gradient_model=args.gradient_model,
        save_dir=args.save_dir,
        train_texts_path=args.train_texts_path,
        anchor_texts_path=args.anchor_texts_path,
        n_stock_anchors=args.n_stock_anchors,
        n_stock_pool=args.n_stock_pool,
        proj_dim=args.proj_dim,
        stocking_flops_path=args.stocking_flops_path,
        training_flops_path=args.training_flops_path,
        stocking_timing_path=args.stocking_timing_path,
        training_timing_path=args.training_timing_path,
        batch_size=args.batch_size,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
