#!/usr/bin/env python3
"""EXP1: consolidate Part 1/2/3's raw outputs into one canonical
EXP1_results.json -- the single file every plot script reads its numbers
from, so nobody (human or agent) has to trust a plotting script's in-memory
re-derivation of a result; the JSON on disk IS the result.

Pure aggregation: copies each part's already-written JSON verbatim into one
file under "part1"/"part2"/"part3" keys, plus a `sources` map recording
exactly which file each section came from. Adds no new computation and
invents no numbers -- if a number is wrong, it was already wrong in the part
script's own output, not introduced here. Run this AFTER Part 1/2/3 have all
written their outputs; refuses to run (loudly) if any source is missing
rather than silently omitting a section.

    python -m baselines.exp1.collect_results
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import os as _os
if _os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg

OUT_DIR = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED)
OUT_PATH = OUT_DIR / "EXP1_results.json"

SOURCES = {
    "part1": OUT_DIR / "exp1_part1.json",
    "part2": OUT_DIR / "exp1_part2.json",
    "part3_less_logra": OUT_DIR / "exp1_part3_less_logra.json",
    "part3_influcoder": OUT_DIR / "exp1_part3_influcoder.json",
}


def main():
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise SystemExit("EXP1_results.json needs all of Part 1/2/3's outputs "
                         f"to exist first -- missing: {missing}")

    result = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "preset": cfg.PRESET,
        "profile": cfg.PROFILE,
        "sources": {name: str(path) for name, path in SOURCES.items()},
        "part1": json.loads(SOURCES["part1"].read_text()),
        "part2": json.loads(SOURCES["part2"].read_text()),
        "part3": {
            "less_logra": json.loads(SOURCES["part3_less_logra"].read_text()),
            "influcoder": json.loads(SOURCES["part3_influcoder"].read_text()),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
