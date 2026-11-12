"""L7 runtime features: dispatch metrics.

Some optimizations are runtime issues, not schedule issues. A kernel that issues many tiny
accelerator commands (config/mvin/mvout) per unit of compute is a candidate for dispatch
batching / command buffers. We count the explicit dispatch-like calls and the fraction that
are non-compute (setup/move) overhead.
"""

from __future__ import annotations

from merlin.analysis.kernels.features._tokens import match_opcodes
from merlin.analysis.kernels.framework_contracts import load_feature_contract
from merlin.analysis.kernels.markers import target_family
from merlin.analysis.kernels.types import NormalizedKernel


def extract_dispatch(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    # Which opcodes count as accelerator dispatches is data (the per-family feature contract), not a
    # per-family name branch. A family with no dispatch contract has no dispatch metric.
    spec = load_feature_contract(target_family(nk.target)).get("dispatch")
    if not spec:
        return {}  # no declared opcode vocabulary: a zero count would masquerade as a measurement
    config_prefix = spec.get("config_prefix", "")
    compute_token = spec.get("compute_token", "")
    calls = match_opcodes(nk.raw_text, spec.get("opcodes", ()))
    n = len(calls)
    n_config = sum(1 for c in calls if config_prefix and c.startswith(config_prefix))
    n_compute = sum(1 for c in calls if c == compute_token)
    frac = round((n - n_compute) / n, 3) if n else 0.0
    return {
        "dispatch_metrics": {
            "n_dispatches": n,
            "n_config": n_config,
            "n_compute": n_compute,
            "small_dispatch_fraction": frac,
        }
    }
