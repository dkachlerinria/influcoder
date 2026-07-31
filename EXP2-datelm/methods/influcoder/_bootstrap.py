"""Puts the parent `influcoder` repo's package on sys.path.

Import this before importing anything from `influcoder.*`. This benchmark
integration (EXP2-datelm) lives nested one level inside the `influcoder` repo
(.../influcoder/EXP2-datelm/methods/influcoder/_bootstrap.py), with the
`influcoder` package itself at the repo root (.../influcoder/influcoder/) --
`parents[3]` from this file lands on that repo root. Override with the
INFLUCODER_REPO_PATH env var if this checkout ever moves elsewhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_INFLUCODER_REPO = Path(__file__).resolve().parents[3]
INFLUCODER_REPO_PATH = Path(os.environ.get("INFLUCODER_REPO_PATH", _DEFAULT_INFLUCODER_REPO))
if not (INFLUCODER_REPO_PATH / "influcoder").is_dir():
    raise ImportError(
        f"influcoder package not found at {INFLUCODER_REPO_PATH} -- set "
        f"INFLUCODER_REPO_PATH to the influcoder repo checkout if this "
        f"benchmark integration isn't nested inside it anymore."
    )
if str(INFLUCODER_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(INFLUCODER_REPO_PATH))
