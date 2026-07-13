""""Actual" ground truth, tis-ie style: plain-SGD LoRA gradients, projected
with TRAK's Rademacher random projection (the `fast_jl` CUDA kernel if
available, dense `BasicProjector` fallback otherwise), L2-normalized, cosine
similarity. This is the exact method tis-ie's
influence_eval/compute_gradient_scores.py + representation/less/compute_less_embeds.py
use to define ground truth in the legacy benchmark.

This is a genuinely different projection from influcoder.gradients'
CountSketch: TRAK projects each gradient onto a dense/structured Rademacher
matrix (computed in blocks); CountSketch is a single sparse
hash-and-scatter_add. Comparing the two is the point of this module -- does
this repo's own CountSketch-based ground truth (and everything trained or
evaluated against it) agree with tis-ie's actual definition of ground truth?

Ported close to source:
  - get_trak_projector, obtain_gradients, collect_grads's core loop  ←
    representation/less/compute_less_embeds.py
  - load_base_with_fresh_lora  ← influence_eval/model_utils.py

Deliberate, documented deviations:
  - attn_implementation is picked via influcoder.gradients.hardware_profile()
    (sdpa on Ampere+) instead of tis-ie's hardcoded "eager" -- that pin was
    only for FLOP-counter compatibility, which isn't used here.
  - Sample tokenization reuses GradientFeaturizer._encode's context/target
    convention (loss on completion tokens only) rather than tis-ie's own
    chat-template builders, matching every other baseline ported so far.

NOT changed, and worth flagging loudly: tis-ie's stock LoRA config has
lora_dropout=0.1 and calls model.train() (not .eval()) during gradient
collection. Faithfully reproduced here, so "legacy" ground truth is
stochastic (dropout-noised) from run to run -- unlike GradientFeaturizer,
which forces eval() specifically for determinism. Pass lora_dropout=0.0 for
a deterministic variant.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import Sample
from .gradients import hardware_profile

# tis-ie's runs/influence_spearman/config_influence.sh LORA_TARGET_MODULES.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]


def _load_base_with_fresh_lora(model_name: str, tokenizer, target_modules: List[str],
                               rank: int, alpha: int, dropout: float, seed: int,
                               device: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype, attn = hardware_profile(device)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=attn
    ).to(device)

    if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
        base_model.resize_token_embeddings(len(tokenizer))

    torch.manual_seed(seed)
    model = get_peft_model(base_model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, task_type="CAUSAL_LM", bias="none",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  legacy GT: {model_name} ({dtype}, {attn}), LoRA r={rank} "
          f"target={target_modules} -> {trainable:,}/{total:,} params")
    return model


def _get_trak_projector(num_params: int, proj_dim: int, device: torch.device, dtype,
                        seed: int = 0, block_size: int = 128,
                        projector_batch_size: int = 16):
    from trak.projectors import BasicProjector, CudaProjector, ProjectionType
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        cls = CudaProjector
        print("  legacy GT: using CudaProjector (fast_jl)")
    except Exception as e:
        cls = BasicProjector
        print(f"  legacy GT: using BasicProjector (CudaProjector failed: {e})")
    return cls(grad_dim=num_params, proj_dim=proj_dim, seed=seed,
               proj_type=ProjectionType.rademacher, device=device, dtype=dtype,
               block_size=block_size, max_batch_size=projector_batch_size)


def _obtain_gradient(model, input_ids, attention_mask, labels) -> torch.Tensor:
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    g = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    model.zero_grad(set_to_none=True)
    return g


def _tokenize(tok, sample: Sample, max_len: int, device: str):
    """Same context/target split as GradientFeaturizer._encode: the target is
    tokenized first and the context tail-truncated to fit, so the loss span
    is never truncated away."""
    tgt = tok(sample.target + (tok.eos_token or ""), add_special_tokens=False).input_ids
    tgt = tgt[: max_len // 2]
    ctx = tok(sample.context, add_special_tokens=False).input_ids
    ctx = ctx[-(max_len - len(tgt)):]
    input_ids = torch.tensor([ctx + tgt], device=device)
    labels = torch.tensor([[-100] * len(ctx) + tgt], device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels


def _collect_projected_grads(model, tokenizer, samples: List[Sample], proj, device: str,
                             max_len: int, project_interval: int, desc: str) -> torch.Tensor:
    full_grads, projected = [], []
    for i, s in enumerate(tqdm(samples, desc=desc)):
        input_ids, attention_mask, labels = _tokenize(tokenizer, s, max_len, device)
        full_grads.append(_obtain_gradient(model, input_ids, attention_mask, labels))
        if (i + 1) % project_interval == 0:
            stacked = torch.stack(full_grads).to(torch.float16)
            projected.append(proj.project(stacked, model_id=0).cpu())
            full_grads = []
            torch.cuda.empty_cache()
    if full_grads:
        stacked = torch.stack(full_grads).to(torch.float16)
        projected.append(proj.project(stacked, model_id=0).cpu())
    return torch.cat(projected, dim=0)


def legacy_gt_scores(anchors: List[Sample], pool: List[Sample], model_name: str,
                     lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1,
                     lora_target_modules: Optional[List[str]] = None, lora_seed: int = 0,
                     proj_dim: int = 65536, project_interval: int = 1,
                     max_len: int = 1024, device: str = "cuda") -> np.ndarray:
    """Defaults match tis-ie's stock runs/influence_spearman/config_influence.sh
    (LORA_RANK=16, LORA_ALPHA=32, LORA_DROPOUT=0.1, LORA_SEED=0,
    GT_PROJ_DIM=65536, PROJECT_INTERVAL=1)."""
    from transformers import AutoTokenizer

    target_modules = lora_target_modules or DEFAULT_TARGET_MODULES
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_base_with_fresh_lora(model_name, tokenizer, target_modules,
                                       lora_rank, lora_alpha, lora_dropout, lora_seed, device)
    model.train()  # matches collect_grads -- dropout active, gradients are stochastic

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  legacy GT: projecting {num_params:,} parameters per sample")
    model_device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    torch.random.manual_seed(0)  # matches collect_grads's own projector seeding
    proj = _get_trak_projector(num_params, proj_dim, model_device, dtype)

    anchor_grads = _collect_projected_grads(model, tokenizer, anchors, proj, device,
                                            max_len, project_interval, "legacy GT anchors")
    pool_grads = _collect_projected_grads(model, tokenizer, pool, proj, device,
                                          max_len, project_interval, "legacy GT pool")

    anchor_grads = F.normalize(anchor_grads.float(), p=2, dim=1, eps=1e-12)
    pool_grads = F.normalize(pool_grads.float(), p=2, dim=1, eps=1e-12)

    del model
    torch.cuda.empty_cache()
    return (anchor_grads @ pool_grads.T).numpy()
