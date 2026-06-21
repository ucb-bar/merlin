"""CIRCT sim-skip gate — wrap a sim oracle adapter so a CIRCT-reject skips the expensive sim.

Realizes the abc4 finding (CIRCT screen ~7 ms vs verilator ~222 s/capsule; 0 false-clean on 119 pts): for
each capsule, decode the agent's emitted RoCC trace and run the SAME CIRCT structural screen the advisory
block uses; if it REJECTS, skip the wrapped spike/verilator/VCS run and fail the tier with the CIRCT
reason (catch the structural bug in ms, not minutes). If CIRCT is clean, run the sim normally (the sim
still certifies numerics — CIRCT is structural-only).

Non-invasive: this wraps an adapter `(cb, llvm_text, workdir, timeout) -> result`; the frozen
capsule_runner/capsule_grade are untouched. Records skips + CIRCT wall into a sidecar list.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable

from . import rocc_decode as RD
from . import rtl_checks as RC

_FACTS = (Path(__file__).resolve().parents[4]
          / "merlin/targets/gemmini/contracts/rtl_facts/facts.json")


class CIRCTReject(Exception):
    """Raised in place of running the sim when the CIRCT structural screen rejects the trace.
    capsule_runner treats an adapter exception as a tier failure — which is the intended outcome
    (CIRCT-reject ⟹ would-fail), reached without paying the sim."""


def gated_adapter(inner: Callable, *, log: list | None = None, facts: dict | None = None) -> Callable:
    """Wrap a sim adapter with a CIRCT pre-screen. `log` (if given) collects per-call records."""
    facts = facts or json.loads(_FACTS.read_text())
    rc_facts = None
    try:
        from .rtl_check_compiler import _facts_to_rc as _f2rc
        rc_facts = _f2rc(facts)
    except Exception:
        rc_facts = facts

    def _wrapped(cb, llvm_text, workdir, timeout):
        t0 = time.time()
        verdict = "ok"
        try:
            trace = RD.decode_text(llvm_text, source="circt_gate")
            rep = RC.screen(trace, None, rc_facts)
            verdict = rep.verdict  # 'ok' | 'warn' | 'reject'
        except Exception:
            verdict = "ok"  # screen must never block a legitimate run on its own error (fail-open here)
        circt_s = time.time() - t0
        rec = {"verdict": verdict, "circt_s": round(circt_s, 4)}
        if verdict == "reject":
            rec["sim_skipped"] = True
            if log is not None:
                log.append(rec)
            raise CIRCTReject(f"CIRCT structural screen rejected the trace (sim skipped, "
                              f"{circt_s*1000:.0f} ms); fix the structural findings before the sim.")
        rec["sim_skipped"] = False
        res = inner(cb, llvm_text, workdir, timeout)   # CIRCT clean -> run the real sim (certifies numerics)
        rec["sim_s"] = round(time.time() - t0 - circt_s, 2)
        if log is not None:
            log.append(rec)
        return res

    return _wrapped
