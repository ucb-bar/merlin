"""Phase / rate model — the cadence at which each region runs, and the workload's rate constants.

A flat capture is single-rate (one forward). A VLA is multi-rate: a backbone runs once per replan,
the action head runs K times per replan, an autoregressive decoder runs token-by-token, and the
control loop consumes H actions at a fixed frequency. This module classifies each region's cadence
with deterministic rules and exposes the rate constants (K, H, control_rate_hz, replan deadline)
with an explicit **source** for each — recovered, assumed_reference, derived, or unavailable.

It claims no timing for any phase it cannot derive: per-iteration periods inside the K-loop are
``unavailable`` (the wall-clock split across the K steps is not recoverable from a flat capture),
never an equal-split guess. No speedup, no cycle count.
"""
from __future__ import annotations

from merlin.dse_guidance import topology as TOP
from merlin.dse_guidance.design_envelope import E_CONFIG, E_DERIVED, E_NA

# Cadence vocabulary (the allowed set; the verifier checks membership).
CAD_ONCE_INSTRUCTION = "once_per_instruction"
CAD_ONCE_REPLAN = "once_per_replan"
CAD_K_TIMES = "K_times_per_replan"
CAD_TOKEN_LOOP = "token_loop"
CAD_CONTROL_TICK = "control_tick"
CAD_ONCE_FORWARD = "once_per_forward"
CAD_UNKNOWN = "unknown"

CADENCES = (CAD_ONCE_INSTRUCTION, CAD_ONCE_REPLAN, CAD_K_TIMES, CAD_TOKEN_LOOP,
            CAD_CONTROL_TICK, CAD_ONCE_FORWARD, CAD_UNKNOWN)


def classify_cadence(role: str | None, workload_class: str, invocations: int | None,
                     K: int) -> str:
    """Deterministic cadence for a region from its role + the workload class.

    A repeated head is a ``token_loop`` for an autoregressive decoder and a ``K_times_per_replan``
    loop otherwise (flow-matching / denoise). A once-per-replan backbone / prefix builder is
    ``once_per_replan``. With no role and no loop, fall back to ``once_per_forward`` / ``unknown``.
    """
    if role in ("backbone_once", "prefix_builder"):
        return CAD_ONCE_REPLAN
    if role == "repeated_head":
        if workload_class == TOP.CLASS_AUTOREGRESSIVE:
            return CAD_TOKEN_LOOP
        return CAD_K_TIMES if (invocations or K) > 1 else CAD_ONCE_FORWARD
    if role == "control_loop":
        return CAD_CONTROL_TICK
    return CAD_UNKNOWN


def rate_model(topo) -> dict:
    """The workload's rate constants with a source label per field (nothing invented)."""
    K = int(topo.K)
    H = int(topo.H)
    rate = float(topo.control_rate_hz)
    deadline_s = (topo.replan_deadline_ms / 1000.0) if topo.replan_deadline_ms else None
    return {
        "K": {"value": K, "source": E_CONFIG},
        "H": {"value": H, "source": E_CONFIG},
        "control_rate_hz": {"value": rate, "source": E_CONFIG},
        "replan_deadline_s": {"value": (round(deadline_s, 6) if deadline_s else None),
                              "source": (E_DERIVED if deadline_s else E_NA)},
        "deadline_equation": topo.deadline_equation(),
    }


def phase_period_s(cadence: str, topo) -> tuple[float | None, str]:
    """Period (seconds) for a cadence, with evidence. Per-K-step periods are unavailable."""
    deadline_s = (topo.replan_deadline_ms / 1000.0) if topo.replan_deadline_ms else None
    if cadence == CAD_ONCE_REPLAN:
        return (round(deadline_s, 6) if deadline_s else None,
                E_DERIVED if deadline_s else E_NA)
    if cadence == CAD_CONTROL_TICK:
        return (round(1.0 / topo.control_rate_hz, 6) if topo.control_rate_hz else None,
                E_DERIVED if topo.control_rate_hz else E_NA)
    # K-loop / token-loop per-iteration wall time is NOT recoverable from a flat capture.
    return (None, E_NA)
