"""Per-sample gradient features of a LoRA-adapted LM.

A GradientFeaturizer attaches a fresh, seeded LoRA adapter to a causal LM and
maps each Sample to the CountSketch of its loss gradient on the adapter
parameters, L2-normalized. Cosine between two features approximates cosine
between the full per-sample gradients (JL property of the sketch), which is
the influence score this whole method distills.

Performance notes (vs. the legacy pipeline this replaces):
  * dtype/attention are picked from the GPU's compute capability -- bf16+SDPA
    on Ampere or newer, fp32+eager on older cards (which emulate bf16 slowly
    and lack flash attention).
  * the sketch's hash/sign tensors are drawn once at construction, not
    regenerated per projection chunk.
  * each gradient is projected on-GPU immediately after backward; features
    accumulate in one preallocated GPU tensor and cross to CPU once.
  * gradients for a given sample are computed exactly once per run.

The remaining bottleneck is one backward pass per sample (per-sample grads
cannot be batched with a plain summed loss). torch.func vmap'd per-sample
gradients or sharding samples across GPUs are the next steps if this ever
dominates again; at current scales on an Ampere+ card it does not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from .data import Sample


def hardware_profile(device: str = "cuda") -> tuple[torch.dtype, str]:
    """(dtype, attn_implementation) appropriate for the GPU."""
    if not torch.cuda.is_available():
        return torch.float32, "eager"
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:  # Ampere+: native bf16, SDPA/flash kernels
        return torch.bfloat16, "sdpa"
    return torch.float32, "eager"


class CountSketch:
    """Seeded sign-hash projection R^P -> R^dim; one scatter_add per call."""

    def __init__(self, num_params: int, dim: int, seed: int, device: str):
        g = torch.Generator().manual_seed(seed)
        self.idx = torch.randint(0, dim, (num_params,), generator=g).to(device)
        self.sign = (torch.randint(0, 2, (num_params,), generator=g).float() * 2 - 1).to(device)
        self.dim = dim

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.dim, device=grad.device, dtype=torch.float32)
        out.scatter_add_(0, self.idx, grad.float() * self.sign)
        return out


class GradientFeaturizer:
    def __init__(self, model_name: str, lora_rank: int = 8, lora_seed: int = 0,
                 proj_dim: int = 8192, proj_seed: int = 42,
                 max_len: int = 1024, device: str = "cuda"):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_len = max_len
        dtype, attn = hardware_profile(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation=attn
        ).to(device)
        torch.manual_seed(lora_seed)
        self.model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0,
            target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
        ))
        self.model.eval()  # dropout off -> deterministic per-sample gradients

        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.sketch = CountSketch(self.n_params, proj_dim, proj_seed, device)
        print(f"  featurizer: {model_name} ({dtype}, {attn}), "
              f"LoRA r={lora_rank} -> {self.n_params:,} params, sketch dim={proj_dim}")

    def _encode(self, sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize (context, target) with loss on target tokens only. The
        target is tokenized first and the context tail-truncated to fit, so
        the loss span can never be truncated away (long BBH CoT prompts lose
        their head, keeping the actual question)."""
        tok = self.tokenizer
        tgt = tok(sample.target + (tok.eos_token or ""), add_special_tokens=False).input_ids
        tgt = tgt[: self.max_len // 2]
        ctx = tok(sample.context, add_special_tokens=False).input_ids
        ctx = ctx[-(self.max_len - len(tgt)):]
        ids = torch.tensor([ctx + tgt], device=self.device)
        labels = torch.tensor([[-100] * len(ctx) + tgt], device=self.device)
        return ids, labels

    def features(self, samples: list[Sample], desc: str = "grads",
                 keep_full: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """[N, proj_dim] L2-normalized features (CPU). Optionally also the
        first keep_full FULL gradient vectors for the fidelity check."""
        out = torch.empty((len(samples), self.sketch.dim), device=self.device)
        fulls: list[torch.Tensor] = []
        for i, s in enumerate(tqdm(samples, desc=desc)):
            ids, labels = self._encode(s)
            loss = self.model(input_ids=ids, labels=labels).loss
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in self.model.parameters()
                           if p.grad is not None]).float()
            if i < keep_full:
                fulls.append(g.cpu().clone())
            out[i] = self.sketch(g)
            self.model.zero_grad(set_to_none=True)
        assert torch.isfinite(out).all(), f"non-finite gradients in {desc}"
        assert (out.norm(dim=1) > 0).all(), f"zero gradient row in {desc} (truncation bug?)"
        return F.normalize(out, dim=1).cpu(), fulls

    def close(self):
        del self.model
        torch.cuda.empty_cache()


def projection_fidelity(fulls: list[torch.Tensor], sketches: torch.Tensor) -> dict:
    """Exact pairwise gradient cosines vs. sketched cosines -- the numeric
    check that the featurizer core (gradients + projection) is right."""
    import numpy as np
    Xf = F.normalize(torch.stack(fulls), dim=1)
    n = len(fulls)
    iu = np.triu_indices(n, 1)
    exact = (Xf @ Xf.T)[iu]
    approx = (sketches[:n] @ sketches[:n].T)[iu]
    err = (exact - approx).abs()
    return {
        "pairs": int(len(err)),
        "mean_abs_err": float(err.mean()),
        "max_abs_err": float(err.max()),
        "spearman": float(spearmanr(exact.numpy(), approx.numpy()).statistic),
    }
