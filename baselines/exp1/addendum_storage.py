#!/usr/bin/env python3
"""EXP1 addendum: how much STORAGE does each method's per-example
representation cost?

Every method in EXP1's ranking-estimation setup ultimately reduces a training
example to one fixed-length vector, then scores by cosine. The vectors differ
enormously in width, so "how big is the index you have to keep around" is a
real axis of comparison that EXP1's quality/time plots don't show.

  method       per-example vector                                  depends on
  -----------  --------------------------------------------------  ----------
  InfluCoder   student-encoder sentence embedding                  encoder hidden size
  RDS+         weighted-mean last hidden state of the target model  target hidden size
  LESS         TRAK random projection of the LoRA gradient          proj_dim (fixed)
  LoGra        per-module flattened [r, r] LoRA-B gradient blocks   n_modules * r^2

The vector width is INDEPENDENT of example length for all four (every method
pools/projects to a fixed dim), so bytes-per-example is a constant and the
1K-example figure is an exact multiple of the 10-example one -- the small run
is a proof of concept, not a sample estimate.

Per the EXP1 setup (`config.py` / `config_biggpu.py`):
  GT/target model   Qwen/Qwen3-4B        (RDS+ scores with this; LESS/LoGra
                                          also run 1.7B/0.6B proxies)
  LESS proj_dim     8192
  LoGra rank        32   (biggpu; 8 in the historical profile)
  LoGra modules     q,k,v,o,gate,up,down  -> 7 per layer
  encoders          ettin-encoder-68m / -150m

InfluCoder uses the UNTRAINED encoder here: distillation changes the weights,
never the output dimension, so the stored artifact is byte-for-byte the same
size. No gradient collection or training is needed for this measurement.

Modes
-----
`--mode analytic` (default) derives each width from the model config, counting
LoGra's target modules on a meta-device model so nothing is downloaded but the
config. `--mode real` actually runs each method on `--n` real dolci-instruct
examples and measures the resulting tensor -- use it on a GPU node to confirm
the analytic widths.

Usage:
    python -m baselines.exp1.addendum_storage --n 10
    python -m baselines.exp1.addendum_storage --n 10 --mode real
    python -m baselines.exp1.addendum_storage --n 10 --mode real --target Qwen/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

LOGRA_TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]


def human(n_bytes: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if (abs(n_bytes) < 1024 if unit == "B" else abs(n_bytes) < 1000) or unit == "TiB":
            return f"{n_bytes:,.1f} {unit}" if unit != "B" else f"{n_bytes:,.0f} B"
        n_bytes /= 1024


def count_logra_modules(model_name: str) -> int:
    """Number of LoraLinear blocks LoGra attaches, == number of flattened
    [r, r] gradient blocks concatenated per sample (modeling_logra.step()).
    Built on the meta device: shapes only, no weights downloaded."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return sum(1 for name, mod in model.named_modules()
               if isinstance(mod, torch.nn.Linear)
               and name.split(".")[-1] in LOGRA_TARGET_SUFFIXES)


def hidden_size(model_name: str) -> int:
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_name).hidden_size


def encoder_dim(model_name: str) -> int:
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_name).hidden_size


# --------------------------------------------------------------------------- #
# Analytic widths -- each traced to the line of code that fixes it
# --------------------------------------------------------------------------- #
def analytic_rows(target: str, encoders: dict, proj_dim: int, logra_rank: int) -> list[dict]:
    """Native dtypes below are the ones MEASURED on 2026-08-26 (A5000, n=1000),
    not the ones the source reads like at a glance -- three of them differ from
    a naive reading, so each carries why. The headline comparison normalizes
    all of them to STORE_DTYPE anyway; these matter only for the `native`
    column."""
    rows = []
    for label, enc_name in encoders.items():
        rows.append(dict(
            method=f"InfluCoder ({label})", dim=encoder_dim(enc_name),
            native_dtype="float32", native_itemsize=4,
            note=f"{enc_name} hidden size; influcoder.encoder.embed() returns "
                 f"convert_to_numpy=True -> float32"))

    rows.append(dict(
        method="RDS+", dim=hidden_size(target),
        native_dtype="float32", native_itemsize=4,
        note=f"{target} hidden size. Native dtype is fp32, NOT the model's bf16: "
             f"rdsplus/score.py builds its position weights with torch.arange() "
             f"(fp32), so `hidden * w` type-promotes the bf16 hidden states"))

    rows.append(dict(
        method="LESS", dim=proj_dim,
        native_dtype="bfloat16", native_itemsize=2,
        note="cfg.LESS_PROJ_DIM -- LESS's 'embedding' is its random projection of "
             "the LoRA gradient down to this width. Native dtype is bf16, NOT the "
             "fp16 _project() casts its INPUT to: the TRAK projector is constructed "
             "with dtype=next(model.parameters()).dtype, so it emits at the model "
             "dtype. Independent of model size AND of LoRA rank"))

    n_mod = count_logra_modules(target)
    for r in sorted({logra_rank, 8}, reverse=True):
        rows.append(dict(
            method=f"LoGra (r={r})", dim=n_mod * r ** 2,
            native_dtype="float32", native_itemsize=4,
            note=f"{n_mod} target modules x r^2={r ** 2}; modeling_logra.step() "
                 f"concatenates flattened [B,r,r] blocks, accumulated in fp32"
                 + ("  [biggpu / final run]" if r == logra_rank else "  [historical config.py profile]")))
    return rows


# --------------------------------------------------------------------------- #
# Real measurement
# --------------------------------------------------------------------------- #
def load_samples(n: int, seed: int = 42):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from influcoder.data import load_pool
    pool = load_pool("dolci_instruct", seed=seed)
    return pool[:n]


# Set once in main(). The common dtype every method's representation is cast
# to before the headline measurement, so the ONLY thing that differs between
# methods is the width of the vector -- see _stats/measure_tensor.
STORE_DTYPE = torch.float32


def _free_gpu():
    """Each real_* loads a full target model; without this they stack up and a
    later method OOMs on a smaller card -- and that OOM would be swallowed into
    measured.error while the table still printed a complete-looking row."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stats(t) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(t, f.name)
        on_disk = Path(f.name).stat().st_size
    return dict(shape=tuple(t.shape), dtype=str(t.dtype),
                nbytes=t.numel() * t.element_size(), on_disk=on_disk)


def measure_tensor(x) -> dict:
    """Two measurements of the same representation:

    `native`     -- exactly what the method's own code path produces. These
                    dtypes are NOT uniform across methods and several are not
                    what the source reads like at a glance (RDS+ type-promotes
                    to fp32 via its fp32 position weights; LESS's TRAK
                    projector is built at the model dtype so it emits bf16
                    despite casting its input to fp16; LoGra accumulates
                    per-sample gradients in fp32). Kept because it is the
                    truth about each implementation.

    `normalized` -- the same tensor cast to STORE_DTYPE. This is the headline
                    number: with dtype held constant, the only thing that can
                    make one method's index bigger than another's is the width
                    of the vector it stores per example (for LESS, that width
                    IS its projection dim -- projecting down to a smaller space
                    is exactly the "embedding size" being compared).
    """
    import numpy as np
    t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    cast = t.to(STORE_DTYPE)
    norm = _stats(cast)
    # Casting DOWN (e.g. --store_dtype float16 over LoGra/LESS fp32 gradient
    # magnitudes) can overflow to inf. Byte counts stay correct either way --
    # they are all this addendum uses -- but the cast tensor would be
    # numerically useless, so say so rather than let it pass silently.
    if not torch.isfinite(cast).all():
        norm["nonfinite_after_cast"] = True
        norm["warning"] = ("values overflowed casting to STORE_DTYPE; byte counts "
                           "are still valid, the cast tensor is not")
    return dict(native=_stats(t), normalized=norm)


def real_influcoder(samples, enc_name: str, max_len: int) -> dict:
    """UNTRAINED encoder on purpose -- distillation never changes the output
    width, so this is exactly the size a trained checkpoint would produce."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from influcoder.encoder import embed, load_encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_encoder(enc_name, device=device, max_seq_len=max_len)
    emb = embed(enc, [s.text for s in samples])
    del enc
    _free_gpu()
    return measure_tensor(emb)


def real_rdsplus(samples, target: str, max_len: int) -> dict:
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from baselines.common import tokenized_dataset
    from baselines.rdsplus.score import weighted_mean_embeds

    tok = AutoTokenizer.from_pretrained(target)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.bfloat16)
    model.to(device).eval()
    dl = DataLoader(tokenized_dataset(tok, samples, max_len), batch_size=1, shuffle=False)
    embeds = weighted_mean_embeds(model, dl, device)
    del model
    _free_gpu()
    return measure_tensor(embeds)


def real_less(samples, target: str, max_len: int, proj_dim: int,
              lora_rank: int, lora_alpha: int, lora_dropout: float,
              lora_seed: int, project_interval: int, block_size: int,
              attn: str) -> dict:
    """Mirrors score_less()'s inner grads() -- the same collect_grads call, so
    what gets measured is exactly the artifact LESS would store."""
    import torch.utils.data
    from transformers import AutoTokenizer
    from baselines.common import tokenized_dataset
    from baselines.less.less_embeds import collect_grads, normalize_embeddings_in_chunks
    from baselines.less.model_utils import load_base_with_fresh_lora

    tok = AutoTokenizer.from_pretrained(target, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_base_with_fresh_lora(
        model_name=target, tokenizer=tok, lora_target_modules="all-linear",
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        seed=lora_seed, gradient_checkpointing=False, attn_implementation=attn)
    dl = torch.utils.data.DataLoader(
        tokenized_dataset(tok, samples, max_len), batch_size=1, shuffle=False)
    g, _ = collect_grads(dl, model, proj_dim=proj_dim, adam_optimizer_state=None,
                         gradient_type="sgd", project_interval=project_interval,
                         block_size=block_size)
    out = normalize_embeddings_in_chunks(g, chunk_size=10000, dim=1, eps=1e-12,
                                         in_place=False)
    del model, g
    _free_gpu()
    return measure_tensor(out)


def real_logra(samples, target: str, max_len: int, rank: int, attn: str) -> dict:
    """Mirrors score_logra()'s encode path (is_test=False, batch_size=1) -- the
    concatenated per-module flattened [r, r] gradient blocks it would store."""
    from baselines.common import tokenized_dataset
    from baselines.logra.modeling_logra import LoGra
    from baselines.logra.score import LOGRA_TARGET_MODULES as SCORE_TARGETS

    logra = LoGra.from_pretrained(
        model_name=target, rank=rank, mlp_only=False,
        target_modules=SCORE_TARGETS, attn_implementation=attn)
    tok = logra.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = tokenized_dataset(tok, samples, max_len)
    embeds = torch.as_tensor(
        logra.encode(ds, batch_size=1, is_test=False, show_progress_bar=True))
    del logra
    _free_gpu()
    return measure_tensor(embeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="proof-of-concept example count")
    ap.add_argument("--extrapolate_to", type=int, default=1000)
    ap.add_argument("--mode", choices=["analytic", "real"], default="analytic")
    ap.add_argument("--target", default="Qwen/Qwen3-4B",
                    help="RDS+/LESS/LoGra target model (EXP1 uses the 4B GT model)")
    ap.add_argument("--proj_dim", type=int, default=8192, help="cfg.LESS_PROJ_DIM")
    ap.add_argument("--logra_rank", type=int, default=32, help="cfg.LOGRA_RANK (biggpu)")
    ap.add_argument("--max_len", type=int, default=1024, help="cfg.MAX_LEN")
    ap.add_argument("--store_dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="common dtype every method is cast to for the headline "
                         "comparison, so only vector width differs between methods")
    ap.add_argument("--less_rank", type=int, default=8, help="cfg.LESS_RANK (biggpu)")
    ap.add_argument("--less_alpha", type=int, default=512, help="cfg.LESS_LORA_ALPHA")
    ap.add_argument("--less_project_interval", type=int, default=8)
    ap.add_argument("--less_block_size", type=int, default=16, help="cfg.LESS_BLOCK_SIZE")
    ap.add_argument("--attn", default="sdpa", help="cfg.ATTN")
    ap.add_argument("--seed", type=int, default=0, help="cfg.SEED")
    ap.add_argument("--force_cpu_target", action="store_true",
                    help="run the target-model methods on CPU anyway (slow)")
    ap.add_argument("--out", default="baselines/out/addendum_storage.json")
    args = ap.parse_args()

    global STORE_DTYPE
    STORE_DTYPE = getattr(torch, args.store_dtype)

    encoders = {"68m": "jhu-clsp/ettin-encoder-68m",
                "150m": "jhu-clsp/ettin-encoder-150m"}

    rows = analytic_rows(args.target, encoders, args.proj_dim, args.logra_rank)

    if args.mode == "real":
        samples = load_samples(args.n)
        print(f"loaded {len(samples)} dolci-instruct examples "
              f"(median {sorted(len(s.text) for s in samples)[len(samples)//2]} chars)\n")
        for row in rows:
            try:
                if row["method"].startswith("InfluCoder"):
                    label = row["method"].split("(")[1].rstrip(")")
                    row["measured"] = real_influcoder(samples, encoders[label], args.max_len)
                elif row["method"] == "RDS+":
                    if torch.cuda.is_available() or args.force_cpu_target:
                        row["measured"] = real_rdsplus(samples, args.target, args.max_len)
                    else:
                        row["measured"] = {"skipped": (
                            f"no GPU on this host: {args.target} forward passes at "
                            f"batch_size=1 are not tractable on CPU at n={args.n}. "
                            "Analytic width still applies (hidden size x fp32); "
                            "re-run on a GPU node, or pass --force_cpu_target.")}
                elif row["method"] == "LESS":
                    if torch.cuda.is_available() or args.force_cpu_target:
                        row["measured"] = real_less(
                            samples, args.target, args.max_len, args.proj_dim,
                            args.less_rank, args.less_alpha, 0.0, args.seed,
                            args.less_project_interval, args.less_block_size, args.attn)
                    else:
                        row["measured"] = {"skipped": "no GPU on this host"}
                elif row["method"].startswith("LoGra"):
                    r = int(row["method"].split("r=")[1].rstrip(")"))
                    if torch.cuda.is_available() or args.force_cpu_target:
                        row["measured"] = real_logra(samples, args.target, args.max_len,
                                                     r, args.attn)
                    else:
                        row["measured"] = {"skipped": "no GPU on this host"}
                else:
                    row["measured"] = {"skipped": "no handler"}
            except Exception as e:  # noqa: BLE001 -- report, don't abort the table
                row["measured"] = {"error": f"{type(e).__name__}: {e}"}

    n, N = args.n, args.extrapolate_to
    store_bytes = torch.empty(0, dtype=STORE_DTYPE).element_size()

    def row_status(row) -> str:
        """A row whose method errored or was skipped still has analytic numbers,
        so without this marker the table would look complete when it is not."""
        m = row.get("measured")
        if m is None:
            return "" if args.mode == "analytic" else "  (not run)"
        if "error" in m:
            return "  !! FAILED"
        if "skipped" in m:
            return "  -- skipped"
        got = m.get("native", {}).get("shape", (None,))[0]
        return "  measured" if got == n else f"  !! ROWS={got}, expected {n}"

    # The ratio baseline is looked up by name rather than taken as rows[0], so
    # reordering the encoder dict can never make the column header lie.
    BASE_METHOD = "InfluCoder (68m)"
    base_row = next((r for r in rows if r["method"] == BASE_METHOD), rows[0])
    base = base_row["dim"] * store_bytes

    print(f"\nHeadline: every method cast to a COMMON dtype ({args.store_dtype}, "
          f"{store_bytes} B/dim), so the only difference between rows is the width "
          f"of the per-example vector.\n")
    # Only show the projected column when it is actually a projection.
    show_proj = N != n
    hdr = (f"{'Method':<22} {'dim':>9} {'B/example':>11} {'n=' + str(n):>12}")
    if show_proj:
        hdr += f" {'n=' + str(N) + ' (proj)':>17}"
    hdr += f"  {'vs ' + BASE_METHOD:>22}   {'native dtype':>13}  status"
    print(hdr)
    print("-" * (len(hdr) + 8))

    for row in rows:
        per = row["dim"] * store_bytes
        row["bytes_per_example_normalized"] = per
        row["bytes_per_example_native"] = row["dim"] * row["native_itemsize"]
        row["store_dtype"] = args.store_dtype
        line = (f"{row['method']:<22} {row['dim']:>9,} {per:>11,} "
                f"{human(per * n):>12}")
        if show_proj:
            line += f" {human(per * N):>17}"
        line += (f"  {per / base:>21.1f}x   {row['native_dtype']:>13}"
                 f"{row_status(row)}")
        print(line)

    print()
    for row in rows:
        print(f"  {row['method']:<22} {row['note']}")
        if "measured" in row:
            print(f"  {'':<22} MEASURED: {row['measured']}")

    failed = [r["method"] for r in rows
              if isinstance(r.get("measured"), dict)
              and ("error" in r["measured"] or "skipped" in r["measured"])]
    bad_rows = [r["method"] for r in rows
                if isinstance(r.get("measured"), dict)
                and "native" in r["measured"]
                and r["measured"]["native"]["shape"][0] != n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        setup=dict(target=args.target, proj_dim=args.proj_dim, logra_rank=args.logra_rank,
                   max_len=args.max_len, n=n, extrapolate_to=N, mode=args.mode),
        rows=rows), indent=2))
    print(f"\nwrote {out}")

    if failed or bad_rows:
        if failed:
            print(f"\nFAILED/SKIPPED: {failed}")
        if bad_rows:
            print(f"WRONG ROW COUNT (expected {n}): {bad_rows}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
