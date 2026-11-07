"""Per-sample projected-gradient features of DATE-LM's own trained LoRA
checkpoints -- the InfluCoder distillation target on this benchmark.

DATE-LM's other attribution methods (Grad_Dot, Grad_Sim, LESS, DataInf,
EKFAC) all score training/reference examples by the per-sample loss
gradient of the checkpoint's LoRA parameters. InfluCoder's role is the same
as in EXP1: distill that (expensive, one-backward-pass-per-sample) signal
into a cheap sentence encoder, so scoring the full train set only costs a
forward pass through a tiny encoder instead of a backward pass through the
1B-parameter checkpoint.

`CountSketch`/`hardware_profile` are imported unchanged from the parent
`influcoder` repo's own package -- they're pure model-agnostic utilities (a
random sign-hash projection and a GPU-capability dtype/attn picker), so
reusing them here doesn't require any changes to that package. See
`_bootstrap.py` for how it's located on sys.path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import _bootstrap  # noqa: F401  (sys.path side effect, must run first)
from influcoder.gradients import CountSketch, hardware_profile  # noqa: E402


def load_teacher_model(base_model_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load DATE-LM's base model + trained LoRA checkpoint for gradient
    computation. Deliberately separate from `methods.model_utils.
    checkpoints_load_func` (which loads fp32/eager, fine for the other
    methods' batch_size=1 forward+backward but slower than necessary): this
    picks bf16+sdpa on Ampere+ cards the same way EXP1's GradientFeaturizer
    does, since per-sample-gradient computation here is InfluCoder's whole
    up-front cost and dominates its total wall-clock.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype, attn = hardware_profile(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, attn_implementation=attn
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.config.pad_token_id = base_model.config.pad_token_id
    for name, p in model.named_parameters():
        p.requires_grad_("lora" in name.lower())
    model.to(device)
    model.eval()  # dropout off -> deterministic per-sample gradients (same reasoning as GradientFeaturizer)
    print(f"  teacher: {base_model_path} + {checkpoint_path} ({dtype}, {attn})")
    return tokenizer, model


def tokenize_chat_examples(chat_examples: list[dict], tokenizer, max_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(input_ids, labels) pairs, each shaped (1, seq_len), via DATE-LM's own
    `encode_with_messages_format` (masks non-assistant tokens out of the
    loss) -- reused unchanged so the loss InfluCoder distills from is
    computed exactly the way DATE-LM's other methods compute it."""
    from datamodules.data_utils import encode_with_messages_format

    out = []
    for ex in chat_examples:
        enc = encode_with_messages_format(ex, tokenizer, max_len)
        out.append((enc["input_ids"].unsqueeze(0), enc["labels"].unsqueeze(0)))
    return out


def has_valid_loss_span(example: dict, tokenizer, max_len: int) -> bool:
    """False iff truncation at max_len cuts off the example's entire
    assistant span, leaving an all -100 labels tensor -- a real occurrence at
    max_len=1024 on this benchmark's longer (e.g. UltraChat-derived)
    prompt/response pairs, which yields a zero (not NaN) gradient row that
    would otherwise crash `compute_gradient_features`'s finite/nonzero
    assertions. Callers should sample teacher-gradient subsets only from
    examples this returns True for."""
    from datamodules.data_utils import encode_with_messages_format

    labels = encode_with_messages_format(example, tokenizer, max_len)["labels"]
    return bool((labels != -100).any())


def sample_valid_indices(chat_examples: list[dict], tokenizer, max_len: int,
                         order: list[int], n_needed: int, exclude: set[int] = frozenset()) -> list[int]:
    """First `n_needed` indices from `order` (a pre-shuffled candidate order)
    whose example has a valid loss span, skipping anything in `exclude` (so
    disjoint train/eval subsets can be drawn from the same shuffled order
    without double-tokenizing to check disjointness)."""
    picked = []
    for i in order:
        if i in exclude:
            continue
        if has_valid_loss_span(chat_examples[i], tokenizer, max_len):
            picked.append(i)
            if len(picked) == n_needed:
                break
    return picked


def compute_gradient_features(model, tokenized_examples: list[tuple[torch.Tensor, torch.Tensor]],
                              proj_dim: int, proj_seed: int = 42, device: str = "cuda",
                              desc: str = "grads") -> torch.Tensor:
    """[N, proj_dim] L2-normalized CountSketch features of each example's
    loss gradient w.r.t. the model's (already-trained) LoRA parameters.
    Mirrors `influcoder.gradients.GradientFeaturizer.features`, adapted to
    an already-loaded model and pre-tokenized (input_ids, labels) pairs
    instead of building its own fresh LoRA + `Sample`-based tokenization."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sketch = CountSketch(n_params, proj_dim, proj_seed, device)
    out = torch.empty((len(tokenized_examples), proj_dim), device=device)
    for i, (input_ids, labels) in enumerate(tqdm(tokenized_examples, desc=desc)):
        input_ids, labels = input_ids.to(device), labels.to(device)
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.grad is not None]).float()
        out[i] = sketch(g)
        model.zero_grad(set_to_none=True)
    assert torch.isfinite(out).all(), f"non-finite gradients in {desc}"
    assert (out.norm(dim=1) > 0).all(), f"zero gradient row in {desc} (masked-out loss span?)"
    return F.normalize(out, dim=1).cpu()


def free_teacher_model(model):
    del model
    torch.cuda.empty_cache()
