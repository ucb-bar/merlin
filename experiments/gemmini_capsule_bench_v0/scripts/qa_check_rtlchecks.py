"""qa_check wrapper for the RTL-checks track — adds an ADVISORY `rtl_checks` block to the verdict.

Drop-in replacement for :mod:`qa_check` used ONLY by ``run_rtlchecks_qa_loop.py`` (the baseline loop is
untouched). It calls the real ``qa_check.run`` to produce the exact same redacted verdict, then appends an
``rtl_checks`` block: deterministic, RTL-grounded structural checks (FileCheck over the emitted
gemmini-dialect MLIR + decoded trace, with bounds from the CIRCT-extracted facts in
``rtl_facts/facts.json``). The block is answer-free — every ``expected`` is derived from RTL facts +
declared shape + ISA rules, never from a golden — and does NOT change pass/fail.

All other attributes are delegated to the real qa_check via module ``__getattr__``.
"""
from __future__ import annotations

import json
from pathlib import Path

import qa_check as _base  # the real, untouched QA gate

import sys
_PKG = Path(__file__).resolve().parents[3] / "merlin" / "python"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from merlin.targetgen import rtl_check_runner as RUN  # noqa: E402


def __getattr__(name):                     # delegate everything else (e.g. _redact_detail, main helpers)
    return getattr(_base, name)


def _first_line(s: str | None) -> str | None:
    return (s.splitlines()[0] if s else None)


def _redact_rtl(r: dict) -> dict:
    """Answer-free per-capsule rtl_checks entry: verdict + FileCheck pass/diag + failing checks only."""
    fc = {k: {"ok": v.get("ok"), "diag": _first_line(v.get("diag"))}
          for k, v in (r.get("filecheck") or {}).items()}
    screen = r.get("screen") or {}
    fails = [{"id": c.get("id"), "severity": c.get("severity"), "message": c.get("message"),
              "expected": c.get("expected"), "got": c.get("got"), "ratio": c.get("ratio"),
              "fix_hint": c.get("fix_hint")}
             for c in screen.get("checks", []) if c.get("status") == "fail"]
    return {"capsule": r.get("capsule"), "verdict": r.get("verdict"),
            "filecheck": fc, "screen_verdict": screen.get("verdict"), "findings": fails}


def _rtl_block(runs_root) -> list[dict]:
    facts = json.loads(RUN._FACTS.read_text())
    index = RUN._capsule_index()
    fc = RUN.find_filecheck()
    bench = Path(runs_root) / "runs" / "gemmini-capsule-bench"
    out = []
    if bench.is_dir():
        dirs = sorted({p.parent.parent for p in bench.glob("*/generated/instruction_trace.json")})
        for d in dirs:
            r = RUN.screen_run(d, facts, index, fc, write=True)  # also writes rtl_checks.json sidecar
            if r:
                out.append(_redact_rtl(r))
    return out


def run(submission: str, capsules_root: str, runs_root, labels, no_oracle: bool, timeout: int) -> dict:
    verdict = _base.run(submission, capsules_root, runs_root, labels, no_oracle, timeout)
    try:
        verdict["rtl_checks"] = _rtl_block(runs_root)
        verdict["rtl_checks_note"] = (
            "ADVISORY RTL-derived checks (FileCheck over emitted MLIR + decoded RoCC trace; bounds from "
            "CIRCT-extracted hardware facts). Does NOT gate pass/fail. Fix encoding/tile/capacity findings "
            "before the RTL oracle would; a clean result means ISA structure is hardware-legal, not that "
            "numerics are correct.")
    except Exception as e:  # advisory must never break the gate
        verdict["rtl_checks_error"] = repr(e)
    return verdict


def main(argv: list[str] | None = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
