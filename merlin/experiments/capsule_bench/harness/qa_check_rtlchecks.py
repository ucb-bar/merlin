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
import _common as C        # the active target (MERLIN_TARGET_EXPERIMENT-aware): gemmini / atlas / …

import sys
_PKG = Path(__file__).resolve().parents[4] / "merlin" / "python"
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
    # SKIPPED checks are surfaced too, with their reason: a check that could not run must never be
    # indistinguishable from one that passed. The reasons are answer-free (they name a missing artifact
    # or an undecidable declaration, never a value).
    not_run = [{"id": k.get("id"), "reason": k.get("reason")} for k in (screen.get("skipped") or [])]
    return {"capsule": r.get("capsule"), "verdict": r.get("verdict"),
            "filecheck": fc, "screen_verdict": screen.get("verdict"), "findings": fails,
            "not_run": not_run}


def _rtl_block(runs_root) -> list[dict]:
    # RTL facts are the regenerated CIRCT artifact now (RUN._FACTS was retired in the facts-as-artifact
    # refactor); load_facts regenerates/reads it on demand. Same full-record shape screen_run expects.
    # TARGET-parameterized (C.TARGET honors MERLIN_TARGET_EXPERIMENT) + ENDPOINT-routed: a RoCC target
    # (gemmini, endpoint inline_asm_insn) gets the dialect+trace FileCheck over its RoCC stream; a
    # self-hosted-ISA target (atlas, endpoint external_backend) gets the kernel opcode-LEGALITY FileCheck
    # over its emitted kernel.S. compile_checks picks by the DERIVED endpoint_kind, never by
    # funct_decode_table presence (the mlc extractor synthesises one for a self-hosted decoder too). Never
    # gemmini facts/ops applied to another target.
    target = C.TARGET
    facts = RUN.load_facts(target)
    index = RUN._capsule_index()
    fc = RUN.find_filecheck()
    bench = Path(runs_root) / "runs" / f"{target}-capsule-bench"
    out = []
    if bench.is_dir():
        # Discover per-capsule run dirs by EITHER RTL-check input: a RoCC target emits
        # generated/instruction_trace.json; a self-hosted-ISA (external_backend, e.g. atlas) emits
        # generated/kernel.S (screen_run picks the right check by endpoint). Globbing only the RoCC trace
        # silently skipped every external_backend run — so the kernel opcode-legality check never fired.
        dirs = sorted({p.parent.parent for g in ("instruction_trace.json", "kernel.S")
                       for p in bench.glob(f"*/generated/{g}")})
        for d in dirs:
            r = RUN.screen_run(d, facts, index, fc, write=True, target=target)  # writes rtl_checks.json
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
            "numerics are correct. Four of the findings are STRUCTURAL and worth reading first, because "
            "each one is a defect the numeric plane can only report as a value error: "
            "T0.output_store_coverage (a declared output with no covering store, or a store past its "
            "declared extent), T0.extent_tile_legalization (a declared extent the RTL-derived array edge "
            "does not divide, with no tail sequence legalizing it), T0.conv_lowering (a declared "
            "convolution whose Kh*Kw window was never folded into the contraction), and "
            "T0.encoded_field_intent (does the DRAM base pointer of each emitted move carry the tensor "
            "the kernel ABI puts in that argument slot? The argument order is RESOLVED from "
            "kernel_abi.arg_order_by_command_shape for YOUR buffer's own command shape -- there is one "
            "order per shape and it is NOT the capsule's or the interface's declaration order, which "
            "coincides with it only for buffers that happen to declare the tensors in that same order. "
            "A buffer resolving against no contract shape, or against two, is reported UNKNOWN rather "
            "than screened against a guess. The check also compares each store's readout dtype, the "
            "store activation and accumulator scale, and each move's column extent against what your own "
            "declaration derives. Both command-buffer planes read the DECLARATION, so an encoding whose "
            "pointer binding disagrees with it passes them and diverges only on the oracle). Every bound "
            "is derived from your capsule's DECLARED shapes plus the array geometry extracted from the "
            "RTL, and every finding is computed from YOUR OWN emitted artifact. A field for which no "
            "intent is derivable produces NO finding and is listed under fields_not_derivable in the "
            "check's evidence instead; a check listed under not_run did NOT run and is not a pass.")
    except Exception as e:  # advisory must never break the gate
        verdict["rtl_checks_error"] = repr(e)
    return verdict


def main(argv: list[str] | None = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
