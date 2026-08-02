#!/usr/bin/env python3
"""Estimate LESS/Grad_Sim/DataInf wall-clock on the Toxicity/Bias task WITHOUT a full run.

EXP2.md section 9 has a completed Grad Sim/Toxicity-Bias number (1764s, A40) but
LESS/Toxicity-Bias was killed at 56% (~50min in, GPU reclaimed) with no number,
and DataInf/Toxicity-Bias was never run at all. Both existing numbers are on an
A40 anyway, not the RTX 6000 Ada this session is using. Rather than running any
method to completion, this times `attributor.cache()` + `attributor.attribute()`
together (see dattri/algorithm/base.py and influence_function.py) on two small
train subsets and fits a line, the same way `baselines/time_logra_batch.py`
derives ms/sample from a handful of timed points instead of a full sweep.

For Grad_Sim/LESS, `cache()` is O(1) (just stores the dataloader reference) so
timing it costs nothing extra. For DataInf it is NOT free: `IFAttributorDataInf
.cache()` runs a real backward pass over `ceil(n * fim_estimate_data_ratio)`
train examples to estimate the FIM, scaling with subset size exactly like
`attribute()` does -- so both must be timed together for the extrapolation to
be correct.

Mirrors dattri.py's actual method construction (get_lora_layers/get_lora_modules,
same AttributionTask/attributor classes) so the estimate reflects the real code
path, not an approximation of it. Mirrors run_repsim_counterfact.py's
model-load-vs-compute split (t_model_load reported separately from the timed
compute) and its litgpt/lightning stub requirement (see PYTHONPATH note below).

Usage (from EXP2-datelm/, with the stub packages on PYTHONPATH):
    PYTHONPATH=_stubs:. ../.venv_py311/bin/python methods/time_toxicity_subset.py \
        --method LESS --n_a 30 --n_b 150
    PYTHONPATH=_stubs:. ../.venv_py311/bin/python methods/time_toxicity_subset.py \
        --method Grad_Sim --n_a 30 --n_b 150
    PYTHONPATH=_stubs:. ../.venv_py311/bin/python methods/time_toxicity_subset.py \
        --method DataInf --n_a 30 --n_b 150
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datamodules.load_data import get_dataset, prepare_chat_format  # noqa: E402
from methods.model_utils import checkpoints_load_func  # noqa: E402
from methods.dattri import get_lora_layers, get_lora_modules, get_loader  # noqa: E402
from dattri.algorithm.base import BaseCosineSimilarityAttributor
from dattri.algorithm.influence_function import IFAttributorLESS, IFAttributorDataInf  # noqa: E402
from dattri.task import AttributionTask  # noqa: E402

TASK = "Toxicity/Bias"
SUBSET = "XSTest-response-Het"
CONFIG_PATH = {
    "LESS": project_root / "configs/toxicity-bias-less.yaml",
    "Grad_Sim": project_root / "configs/toxicity-bias-gradsim.yaml",
    "DataInf": project_root / "configs/toxicity-bias-datainf.yaml",
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_attributor(method, cfg, task, lora_layers, lora_cnt):
    if method == "Grad_Sim":
        return BaseCosineSimilarityAttributor(task=task, device=cfg.device)
    if method == "LESS":
        return IFAttributorLESS(
            task=task,
            layer_name=lora_layers,
            proj_dim=cfg.proj_dim,
            grad_in_dim=lora_cnt,
            device=cfg.device,
        )
    if method == "DataInf":
        return IFAttributorDataInf(
            task=task,
            layer_name=lora_layers,
            device=cfg.device,
            regularization=cfg.regularization,
            fim_estimate_data_ratio=cfg.fim_estimate_data_ratio,
        )
    raise NotImplementedError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["LESS", "Grad_Sim", "DataInf"])
    ap.add_argument("--n_a", type=int, default=30, help="smaller train subset size")
    ap.add_argument("--n_b", type=int, default=150, help="larger train subset size")
    ap.add_argument("--warmup_n", type=int, default=5,
                     help="throwaway subset size to absorb first-call warmup cost before timing")
    args = ap.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH[args.method])
    print(f"{args.method} / {TASK} ({SUBSET}) | base_model={cfg.base_model_path} "
          f"checkpoint={cfg.checkpoint} device={cfg.device}\n")

    t0 = time.perf_counter()
    tokenizer, model = checkpoints_load_func(None, cfg.checkpoint, cfg.base_model_path)
    sync()
    t_model_load = time.perf_counter() - t0
    print(f"model load: {t_model_load:.1f}s")

    train, ref = get_dataset(TASK, SUBSET)
    n_train_full, n_ref = len(train), len(ref)
    print(f"full dataset: n_train={n_train_full} n_ref={n_ref}\n")

    ref_chat = prepare_chat_format(ref)
    ref_loader = get_loader(ref_chat, tokenizer)

    lora_layers, lora_cnt = get_lora_layers(model)
    lora_modules = get_lora_modules(model)

    def f(params, data_target_pair):
        input_ids, labels = data_target_pair
        outputs = torch.func.functional_call(model, params, input_ids, kwargs={
            "labels": labels.cuda()
        })
        return outputs.loss

    task = AttributionTask(
        loss_func=f,
        model=model,
        base_model_path=cfg.base_model_path,
        checkpoints=cfg.checkpoint,
        checkpoints_load_func=checkpoints_load_func,
    )

    t0 = time.perf_counter()
    attributor = build_attributor(args.method, cfg, task, lora_layers, lora_cnt)
    sync()
    t_attributor_init = time.perf_counter() - t0
    print(f"attributor init ({args.method}{' -- includes optimizer-state download for LESS' if args.method == 'LESS' else ''}): "
          f"{t_attributor_init:.1f}s\n")

    # Warmup: the first attribute() call pays a one-time cost (functorch/vmap
    # transform tracing, cudnn/kernel autotune -- visible as the "no batching
    # rule for aten::_scaled_dot_product_flash_attention_backward" warning)
    # that has nothing to do with steady-state per-example cost. Absorb it
    # here on a throwaway subset so the two timed points below are both warm.
    # NOTE: for DataInf, cache() is NOT free -- IFAttributorDataInf.cache() runs
    # a real backward pass over ceil(len(loader) * fim_estimate_data_ratio)
    # train examples to estimate the FIM, so it scales with subset size just
    # like attribute() does (see dattri/algorithm/influence_function.py's
    # IFAttributorDataInf.cache). For Grad_Sim/LESS, cache() is O(1) (just
    # stores the dataloader reference) so timing it alongside attribute() is
    # harmless. Time both together so the fit is correct for all three methods.
    warmup_n = min(args.warmup_n, args.n_a)
    warmup_subset = train.select(range(warmup_n))
    warmup_loader = get_loader(prepare_chat_format(warmup_subset), tokenizer)
    sync()
    t0 = time.perf_counter()
    attributor.cache(warmup_loader)
    attributor.attribute(warmup_loader, ref_loader)
    sync()
    print(f"warmup (n_train={warmup_n}, discarded): {time.perf_counter()-t0:.2f}s\n")

    points = []
    for n in sorted({args.n_a, args.n_b}):
        subset = train.select(range(n))
        subset_chat = prepare_chat_format(subset)
        subset_loader = get_loader(subset_chat, tokenizer)

        sync()
        t0 = time.perf_counter()
        attributor.cache(subset_loader)
        attributor.attribute(subset_loader, ref_loader)
        sync()
        dt = time.perf_counter() - t0
        print(f"n_train={n:4d}: cache()+attribute() = {dt:.2f}s ({dt/n*1000:.1f} ms/train-example)")
        points.append((n, dt))

    (n_a, t_a), (n_b, t_b) = points
    if n_b == n_a:
        raise SystemExit("--n_a and --n_b must differ to fit a slope")
    per_example_s = (t_b - t_a) / (n_b - n_a)
    fixed_per_call_s = t_a - n_a * per_example_s  # ~= cost of test/ref gradient pass, held constant

    full_attribute_s = fixed_per_call_s + n_train_full * per_example_s
    total_wall_s = t_model_load + t_attributor_init + full_attribute_s

    print(f"\nfit: {per_example_s*1000:.2f} ms/train-example, "
          f"fixed per-call (ref-pass + overhead) = {fixed_per_call_s:.2f}s")
    print(f"EXTRAPOLATED full cache()+attribute() @ n_train={n_train_full}: "
          f"{full_attribute_s:.1f}s (~{full_attribute_s/60:.1f} min)")
    print(f"EXTRAPOLATED total wall-clock (model load + attributor init + full cache()+attribute()): "
          f"{total_wall_s:.1f}s (~{total_wall_s/60:.1f} min)")
    if torch.cuda.is_available():
        print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
