"""CIRCT structural screen — wrap a sim oracle adapter to record a fast structural verdict, ADVISORY only.

For each capsule, decode the agent's emitted RoCC trace and run the same CIRCT structural screen the
advisory block uses, recording its verdict + wall alongside the sim. It is **advisory: it never skips the
sim and never fails the tier.** The sim (RTL) is the authoritative correctness oracle; a structural
`reject` can be a false-positive on a conformant-but-different kernel (no leading FENCE, an UNKNOWN our
decoder cannot yet classify, a legal-but-unusual instruction order), so gating on it would wrongly fail a
correct backend — and, because only the rtlchecks arm is wrapped, would bias the cross-arm A/B comparison.
Earlier this gate SKIPPED the sim on `reject`; that made a cheap structural check pre-empt the authoritative
oracle (the same class fixed on the trace + oracle-tier sides). A validated sim-skip (skipping only on
checks proven to predict RTL rejection) can be reintroduced once those checks are confirmed against L3.

Non-invasive: this wraps an adapter `(cb, llvm_text, workdir, timeout) -> result`; the frozen
capsule_runner/capsule_grade are untouched. Records the screen verdict + CIRCT/sim wall into a sidecar list.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Callable

from .rocc import decode as RD
from . import rtl_checks as RC
from .rtl.facts import load_facts


class CIRCTReject(Exception):
    """Retained for import compatibility; NO LONGER RAISED. The screen is advisory (see module
    docstring) — it never skips the sim, so it never raises this. Kept so existing imports resolve."""


def gated_adapter(inner: Callable, *, log: list | None = None, target: str,
                  facts: dict | None = None) -> Callable:
    """Wrap a sim adapter with a CIRCT pre-screen. `log` (if given) collects per-call records.

    `target` is REQUIRED: the structural screen evaluates the emitted trace against the RUN's resolved
    target facts (`load_facts(target)`), never an assumed default. Callers pass the target they are
    grading (e.g. the target-experiment's resolved target)."""
    facts = facts or load_facts(target)
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
            trace = RD.decode_text(llvm_text, source="circt_gate", target=target)
            rep = RC.screen(trace, None, rc_facts, target=target)
            verdict = rep.verdict  # 'ok' | 'warn' | 'reject'
        except Exception:
            verdict = "ok"  # screen must never block a legitimate run on its own error (fail-open here)
        circt_s = time.time() - t0
        # ADVISORY: record the screen verdict but ALWAYS run the sim. The sim (RTL) is the authoritative
        # correctness oracle; a structural 'reject' can be a false-positive on a conformant-but-different
        # kernel, so skipping the sim here would both mis-fail a correct backend AND bias this arm vs the
        # ungated arms. Never raise, never skip.
        rec = {"verdict": verdict, "circt_s": round(circt_s, 4), "sim_skipped": False}
        res = inner(cb, llvm_text, workdir, timeout)   # the sim certifies numerics + structure on real RTL
        rec["sim_s"] = round(time.time() - t0 - circt_s, 2)
        if log is not None:
            log.append(rec)
        return res

    return _wrapped
