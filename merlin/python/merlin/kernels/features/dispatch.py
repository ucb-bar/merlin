"""L7 runtime features: dispatch metrics.

Some optimizations are runtime issues, not schedule issues. A kernel that issues many tiny
accelerator commands (config/mvin/mvout) per unit of compute is a candidate for dispatch
batching / command buffers. We count the explicit dispatch-like calls and the fraction that
are non-compute (setup/move) overhead.
"""
from __future__ import annotations

import re

from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel

_GEMMINI_DISPATCH = re.compile(
    r"\b(mvin[23]?|mvout|preload|compute_preloaded|config_(?:ex|ld|st)|fence)\b"
)


def extract_dispatch(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    fam = target_family(nk.target)
    if fam not in {"gemmini"}:
        return {"dispatch_metrics": {"n_dispatches": 0, "small_dispatch_fraction": 0.0}}
    calls = _GEMMINI_DISPATCH.findall(nk.raw_text)
    n = len(calls)
    n_config = sum(1 for c in calls if c.startswith("config"))
    n_compute = sum(1 for c in calls if c == "compute_preloaded")
    frac = round((n - n_compute) / n, 3) if n else 0.0
    return {"dispatch_metrics": {
        "n_dispatches": n,
        "n_config": n_config,
        "n_compute": n_compute,
        "small_dispatch_fraction": frac,
    }}
