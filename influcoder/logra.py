"""LoGRA: gradient-based influence via per-sample LoRA-B gradients with
Fisher Information Matrix (FIM) preconditioning.

The custom-autograd core (LoraBFunction, CustomLoraB, LoraLinear,
replace_linear_with_lora, LoGraModel) is copied close to verbatim from
tis-ie's logra/less/utils/modeling_logra.py -- that machinery (per-sample
gradient capture + FIM accumulation via a custom backward) is the actual
LoGRA algorithm and isn't specific to that repo's data pipeline.

Only the outer `LoGra` high-level class is rewritten: tis-ie's version reads
pre-tokenized HF datasets and Tulu chat-formatted dicts; this version tokenizes
influcoder.data.Sample directly, using the same context/target split and
truncation rule as influcoder.gradients.GradientFeaturizer._encode (loss on
target tokens only, target tokenized first so it's never truncated away).

FLOPs/timing instrumentation from the tis-ie version (influence_eval.flops_measure)
is not ported -- this repo doesn't track that anywhere else either.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .data import Sample
from .gradients import hardware_profile

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# LoRA custom backward for per-sample gradients + FIM (verbatim from tis-ie)
# ============================================================================

class LoraBFunction(autograd.Function):
    """Custom backward function to compute per-sample gradients and FIM."""

    @staticmethod
    def forward(ctx, input, weight, module, grad_store):
        ctx.save_for_backward(input, weight)
        ctx.module = module
        ctx.grad_store = grad_store
        output = input.matmul(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        if grad_output.dim() > 2:
            B = grad_output.size(0)
            r = grad_output.size(-1)
            grad_output_flat = grad_output.view(B, -1, r)
            input_flat = input.view(B, -1, r)
            per_token_grad = torch.einsum("bni,bnj->bnij", grad_output_flat, input_flat)
            per_sample_grad = per_token_grad.sum(dim=1)
        else:
            per_sample_grad = torch.einsum("bi,bj->bij", grad_output, input)

        ctx.grad_store[id(ctx.module)] = per_sample_grad.detach()

        B = per_sample_grad.size(0)
        g_flat = per_sample_grad.view(B, -1)
        batch_fim = torch.einsum("bi,bj->ij", g_flat, g_flat)
        ctx.module.fim.add_(batch_fim)

        grad_weight = per_sample_grad.sum(dim=0)
        grad_input = torch.matmul(grad_output.to(weight.dtype), weight)
        return grad_input, grad_weight, None, None


class CustomLoraB(nn.Module):
    """Custom LoRA B layer that tracks per-sample gradients and FIM."""

    def __init__(self, rank: int, grad_store: dict):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rank, rank))
        self.module_id = id(self)
        self.grad_store = grad_store
        self.register_buffer('fim', torch.zeros(self.weight.numel(), self.weight.numel()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LoraBFunction.apply(input, self.weight, self, self.grad_store)


class LoraLinear(nn.Module):
    """LoRA-augmented linear layer: output = linear(x) + C(B(A(x)))."""

    def __init__(self, rank: int, linear: nn.Linear, grad_store: dict,
                 shared_module: nn.Module = None, only_lora: bool = False):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features
        self.rank = min(rank, in_features, out_features)
        self.only_lora = only_lora

        if not self.only_lora:
            self.logix_lora_A = nn.Linear(in_features, self.rank, bias=False)
            self.logix_lora_C = nn.Linear(self.rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.logix_lora_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.logix_lora_C.weight, a=math.sqrt(5))

        self.logix_lora_B = shared_module if shared_module is not None \
            else CustomLoraB(self.rank, grad_store)
        self._linear = linear

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.only_lora:
            return self.logix_lora_B(input_tensor)
        output = self._linear(input_tensor)
        output = output + self.logix_lora_C(self.logix_lora_B(self.logix_lora_A(input_tensor)))
        return output


def _get_submodules(model: nn.Module, key: str):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def replace_linear_with_lora(
    model: nn.Module,
    rank: int,
    grad_store: dict,
    only_lora: bool = False,
    mlp_only: bool = False,
    target_modules: Optional[List[str]] = None,
):
    """Replace linear layers in model with LoRA-augmented versions."""
    linear_module_names = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    for name in linear_module_names:
        if target_modules is not None:
            if not any(name.endswith(t) for t in target_modules):
                continue
        elif mlp_only and "mlp" not in name.lower():
            continue
        current_module = model.get_submodule(name)
        parent, target, target_name = _get_submodules(model, name)
        lora_module = LoraLinear(rank, current_module, grad_store, only_lora=only_lora)
        lora_module.to(current_module.weight.dtype)
        setattr(parent, target_name, lora_module)


class LoGraModel(nn.Module):
    """LoGra influence estimator using per-sample gradients."""

    def __init__(self, model: nn.Module, rank: int = 8, only_lora: bool = False, mlp_only: bool = True,
                 target_modules: Optional[List[str]] = None):
        super().__init__()
        self.rank = rank
        self.only_lora = only_lora
        self.grad_store = {}

        replace_linear_with_lora(model, rank, self.grad_store, only_lora=only_lora, mlp_only=mlp_only,
                                  target_modules=target_modules)
        self.model = model

        for name, param in self.model.named_parameters():
            if "logix_lora_B" not in name:
                param.requires_grad = False

        self.lora_module_ids = []
        for module in self.model.modules():
            if isinstance(module, LoraLinear):
                self.lora_module_ids.append(module.logix_lora_B.module_id)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self) -> torch.Tensor:
        """Collect per-sample gradients after backward pass."""
        per_sample_grad_list = []
        for module_id in self.lora_module_ids:
            grad_tensor = self.grad_store.get(module_id, None)
            if grad_tensor is None:
                continue
            per_sample_grad_list.append(grad_tensor.view(grad_tensor.size(0), -1))
        if not per_sample_grad_list:
            return None
        concat_grad = torch.cat(per_sample_grad_list, dim=1)
        self.grad_store.clear()
        return concat_grad

    def get_fim(self, reset: bool = False) -> torch.Tensor:
        """Fisher Information Matrix from all LoRA-B modules."""
        fim_mat = []
        for module_id in self.lora_module_ids:
            for module in self.model.modules():
                if isinstance(module, LoraLinear) and module.logix_lora_B.module_id == module_id:
                    b_module = module.logix_lora_B
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(b_module.fim, op=torch.distributed.ReduceOp.SUM)
                    fim_mat.append(b_module.fim.detach().cpu())
                    if reset:
                        b_module.fim.zero_()
                    break
        return torch.stack(fim_mat, dim=0)


# ============================================================================
# High-level interface, adapted to influcoder Samples
# ============================================================================

class LoGra:
    """Encodes influcoder Samples into LoGRA gradient embeddings and scores
    them by cosine similarity, with optional FIM preconditioning.

    Matches tis-ie's compute_logra_scores.py flow: encode() the pool first
    (is_test=False, accumulates+stores FIM), then encode() the anchors
    (also is_test=False -- raw gradients, no forward/backward is repeated),
    then either similarity() the raw anchor embeddings directly (logra_raw)
    or _precondition() them against the saved pool FIM first (logra_fim).
    encode(is_test=True) is kept for API parity with the source but isn't
    used by baselines.logra_scores(); it would just redo a second
    forward/backward pass to get what _precondition() gives for free.
    """

    def __init__(self, model_name: str, rank: int = 8, only_lora: bool = False,
                 mlp_only: bool = True, target_modules: Optional[List[str]] = None,
                 max_len: int = 1024, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_len = max_len
        dtype, attn = hardware_profile(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lm_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation=attn
        )
        self.model = LoGraModel(lm_model, rank=rank, only_lora=only_lora, mlp_only=mlp_only,
                                target_modules=target_modules).to(device)
        self.model.eval()
        self.fim = None  # set by encode(is_test=False)

    def _tokenize_batch(self, batch: List[Sample]):
        """Same context/target split as GradientFeaturizer._encode: the
        target is tokenized first and the context tail-truncated to fit, so
        the loss span can never be truncated away."""
        tok = self.tokenizer
        ids_list, label_list = [], []
        for s in batch:
            tgt = tok(s.target + (tok.eos_token or ""), add_special_tokens=False).input_ids
            tgt = tgt[: self.max_len // 2]
            ctx = tok(s.context, add_special_tokens=False).input_ids
            ctx = ctx[-(self.max_len - len(tgt)):]
            ids_list.append(torch.tensor(ctx + tgt))
            label_list.append(torch.tensor([-100] * len(ctx) + tgt))
        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=tok.pad_token_id).to(self.device)
        labels = pad_sequence(label_list, batch_first=True, padding_value=-100).to(self.device)
        return input_ids, labels

    def encode(self, samples: List[Sample], batch_size: int = 1, is_test: bool = False,
               show_progress_bar: bool = True) -> np.ndarray:
        """is_test=False (pool): computes+stores FIM, returns raw gradients.
        is_test=True (anchors): applies the stored FIM preconditioning."""
        all_embeds = []
        desc = "LoGRA anchors" if is_test else "LoGRA pool"
        for i in tqdm(range(0, len(samples), batch_size), desc=desc, disable=not show_progress_bar):
            batch = samples[i:i + batch_size]
            input_ids, labels = self._tokenize_batch(batch)
            loss = self.model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            per_sample_grads = self.model.step()
            if per_sample_grads is not None:
                all_embeds.append(per_sample_grads.cpu())
            self.model.zero_grad()

        if not all_embeds:
            raise ValueError(f"No embeddings collected for {len(samples)} samples")
        embeds = torch.cat(all_embeds, dim=0)

        if not is_test:
            self.fim = self.model.get_fim(reset=True)
            return embeds.float().numpy()
        if self.fim is None:
            raise ValueError("Must encode pool data first to compute FIM")
        return self._precondition(embeds).float().numpy()

    def _precondition(self, embeds: torch.Tensor, batch_size: int = 512,
                      damping: Optional[float] = None) -> torch.Tensor:
        n, k, _ = self.fim.size()
        result = []
        fim = self.fim.float()
        eye = torch.eye(k, device=fim.device, dtype=fim.dtype).unsqueeze(0)

        if damping is None:
            traces = fim.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
            damping_module = 0.1 * traces / k
            damping_module = damping_module.view(n, 1, 1)
        else:
            damping_module = damping

        fim_damped = fim + damping_module * eye
        fim_inv = torch.linalg.pinv(fim_damped)  # handles near-singular FIMs

        for i in range(0, embeds.size(0), batch_size):
            batch = embeds[i:i + batch_size].view(-1, n, k).float()
            out = torch.einsum('ank,nkj->anj', batch, fim_inv)
            out = out.reshape(-1, n * k)
            result.append(out)

        return torch.cat(result, dim=0)

    def similarity(self, query_embeddings: np.ndarray, corpus_embeddings: np.ndarray,
                   mode: str = 'cosine', batch_size: int = 2048) -> np.ndarray:
        all_scores = np.zeros(shape=(len(query_embeddings), len(corpus_embeddings)))
        device = next(self.model.parameters()).device

        query_tensor = torch.from_numpy(query_embeddings).to(device).float()
        if mode == 'cosine':
            query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)

        for i in range(0, len(corpus_embeddings), batch_size):
            batch = corpus_embeddings[i:i + batch_size]
            batch = torch.tensor(batch, device=device).float()
            if mode == 'cosine':
                batch = torch.nn.functional.normalize(batch, p=2, dim=1)
            score = torch.mm(query_tensor, batch.T).cpu().float().numpy()
            all_scores[:, i:i + batch_size] = score

        return all_scores

    def close(self):
        del self.model
        torch.cuda.empty_cache()
