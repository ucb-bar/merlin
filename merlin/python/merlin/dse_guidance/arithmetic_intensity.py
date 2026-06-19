"""Hardware-INDEPENDENT roofline: arithmetic intensity + ridge-point regime (P24).

The honest roofline given "we don't know the HW": the roofline x-axis — arithmetic intensity
(MACs per byte) — is a property of the WORKLOAD, not a chip, and we recover it exactly from the
deployment magnitudes (`real_config`). We present only the hardware-free part:

  * **AI = MACs / weight-bytes-moved**, computed two ways:
      - **non-resident** (weights reloaded every step): AI = 1 / dtype_bytes  — every MAC pays for its
        own weight byte; the floor of intensity.
      - **resident** (weights loaded once, reused across the K-step head + the once prefix):
        AI = (prefix_MACs + repeated_MACs·K) / ((prefix+repeated) params · dtype_bytes).
  * **Residency gain** = resident_AI / non-resident_AI = (prefix + repeated·K)/(prefix + repeated) —
    how much residency raises intensity (workload-specific via the prefix/repeated split and K).
  * **Ridge-point regime partition** (parameterized, NOT a chip): a machine with compute:bandwidth
    balance B (MACs/byte) makes this workload compute-bound iff AI > B. We report AI and let any B be
    compared — partitioning the space of possible machines, committing to none.

No peak FLOPs / bandwidth / latency is assumed anywhere — those would need a specific chip. All values
are `recovered_from_model_config` (Tier A/B). MACs≈params per token (one MAC per weight per token); the
prefix is counted once, the repeated head ×K. dtype default bf16 (2 B); int8 doubles every AI.
"""

from __future__ import annotations

from merlin.dse_guidance import real_config as RC

_PREFIX_ROLES = {"prefix_once", "backbone_once"}
_REPEATED_ROLES = {"repeated_head", "decode_lm"}

_AI_COLS = ["workload", "K", "dtype", "prefix_params", "repeated_params",
            "macs_per_replan", "weight_bytes_resident", "weight_bytes_nonresident",
            "ai_resident_mac_per_byte", "ai_nonresident_mac_per_byte", "residency_gain",
            "ridge_balance_compute_bound_below", "regime_note", "evidence"]


def _split_params(g) -> tuple[int, int]:
    """(prefix params loaded once, repeated-head params run per step)."""
    prefix = sum(s.layer_params() * s.n_layers for s in g.stacks if s.role in _PREFIX_ROLES)
    prefix += g.embed_params()                       # embedding/lm_head load once
    repeated = sum(s.layer_params() * s.n_layers for s in g.stacks if s.role in _REPEATED_ROLES)
    return prefix, repeated


def ai_rows(dtype: str = "bf16") -> list[dict]:
    db = RC._DTYPE_BYTES[dtype]
    rows = []
    for w, g in sorted(RC.REAL_GEOMETRY.items()):
        prefix, repeated = _split_params(g)
        if repeated == 0:                            # no repeated head -> skip (not a loop workload)
            continue
        K = g.K
        macs = prefix + repeated * K                 # MACs per replan (1 MAC per weight per token)
        wb_res = (prefix + repeated) * db            # each weight loaded ONCE
        wb_non = (prefix + repeated * K) * db        # repeated weights reloaded every step
        ai_res = macs / wb_res
        ai_non = macs / wb_non                        # == 1/db by construction (floor)
        gain = ai_res / ai_non
        rows.append({
            "workload": w, "K": K, "dtype": dtype,
            "prefix_params": prefix, "repeated_params": repeated,
            "macs_per_replan": macs,
            "weight_bytes_resident": wb_res, "weight_bytes_nonresident": wb_non,
            "ai_resident_mac_per_byte": round(ai_res, 4),
            "ai_nonresident_mac_per_byte": round(ai_non, 4),
            "residency_gain": round(gain, 3),
            # a machine with compute:bandwidth balance below this (MACs/byte) is COMPUTE-bound on the
            # resident workload; above it, memory-bound. No chip assumed — compare any balance B.
            "ridge_balance_compute_bound_below": round(ai_res, 4),
            "regime_note": (f"resident: compute-bound for machine balance < {ai_res:.2f} MAC/byte; "
                            f"residency raises AI {gain:.2f}x over reload-every-step"),
            "evidence": "recovered_from_model_config (hardware-independent; no peak/bandwidth assumed)",
        })
    return rows


def ai_csv(dtype: str = "bf16") -> str:
    from merlin.dse_guidance.case_study import _csv
    return _csv(ai_rows(dtype), _AI_COLS)
