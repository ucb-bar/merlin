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
from merlin.targetgen import rtl_object_screen as _OS  # noqa: E402 — the emitted-words convention name


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
    out = {"capsule": r.get("capsule"), "verdict": r.get("verdict"),
           "filecheck": fc, "screen_verdict": screen.get("verdict"), "findings": fails}
    # The honesty ledger travels WITH the findings: which checks actually ran, and which could not and why.
    # Without it an empty `findings` list is indistinguishable from "every check passed", which is the exact
    # misreading that let this arm look healthy for 18 rounds. Answer-free — both are derivation names.
    if screen.get("grounded") or screen.get("dropped"):
        out["checks_grounded"] = sorted(screen.get("grounded") or {})
        out["checks_dropped"] = dict(sorted((screen.get("dropped") or {}).items()))
    if r.get("object_source"):
        out["screened_artifact"] = r["object_source"]
    return out


def _rtl_coverage(runs_root, target: str) -> dict:
    """How much of this round the RTL checks could actually LOOK at.

    ``_rtl_block`` returns one entry per capsule it screened, and an empty list is indistinguishable from
    "every capsule is clean" — which is exactly how it was read. Measured: 18 consecutive rounds across
    three repeats of the RTL-checks arm reported ``rtl_checks: []`` with no error, and the arm was taken to
    be working. It was not: ``screen_run`` has two check families, one keyed on ``generated/kernel.S`` and
    one on ``generated/instruction_trace.json``, and a target whose compiler emits an LLVM-dialect lowering
    (compiled fork-free) writes NEITHER, so every capsule returned ``None`` and the advisory said nothing at
    all in a shape that looks like assent.

    So report the denominator alongside the findings. A check that could not run must say so — never
    succeed by silence (repo rule; see the `checks-that-skip-and-report-success` class).
    """
    bench = Path(runs_root) / "runs" / f"{target}-capsule-bench"
    if not bench.is_dir():
        return {"run_dirs": 0, "screened": 0, "unscreenable": 0, "artifacts": {}}
    dirs = sorted(p.parent for p in bench.glob("*/generated") if p.is_dir())
    seen: dict[str, int] = {}
    for d in dirs:
        for f in sorted((d / "generated").iterdir()):
            if f.is_file():
                seen[f.name] = seen.get(f.name, 0) + 1
    return {"run_dirs": len(dirs), "screened": 0, "unscreenable": 0,
            # Only the names screen_run can consume, plus their real counts — so a reader can see at a
            # glance that the inputs the checks need are absent rather than that the checks passed. One
            # entry per CHECK FAMILY: decoded RoCC trace, self-hosted kernel assembly, recorded machine-code
            # words (the family a compiler that emits an MLIR lowering lands in).
            "artifacts": {n: seen.get(n, 0) for n in
                          ("kernel.S", "instruction_trace.json", _OS.WORDS_ARTIFACT)}}


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
        # Offer screen_run EVERY capsule run dir and let IT decide applicability — it already routes by the
        # derived endpoint and returns None when the artifact its check family needs is absent. Discovering
        # by a hardcoded pair of filenames put the same routing decision in two places, and the copy here
        # went stale: it has been extended once already (the RoCC trace alone skipped every
        # external_backend run) and was stale again for the fork-free lowering path. One owner, not two.
        dirs = sorted(p.parent for p in bench.glob("*/generated") if p.is_dir())
        for d in dirs:
            r = RUN.screen_run(d, facts, index, fc, write=True, target=target)  # writes rtl_checks.json
            if r:
                out.append(_redact_rtl(r))
    return out


def run(submission: str, capsules_root: str, runs_root, labels, no_oracle: bool, timeout: int,
        **kw) -> dict:
    # Forward anything else STRUCTURALLY (**kw) rather than restating the base signature. This wrapper is
    # a drop-in for qa_check.run, so every parameter the real gate grows has to arrive here too — and a
    # positional mirror silently rots: it kept working for every existing caller while raising TypeError
    # on exactly the one arm that routes through this wrapper, i.e. it would break the RTL-checks arm and
    # nothing else, which is the hardest shape of breakage to attribute.
    verdict = _base.run(submission, capsules_root, runs_root, labels, no_oracle, timeout, **kw)
    try:
        cov = _rtl_coverage(runs_root, C.TARGET)
        entries = _rtl_block(runs_root)
        cov["screened"] = len(entries)
        cov["unscreenable"] = max(0, cov["run_dirs"] - len(entries))
        verdict["rtl_checks"] = entries
        verdict["rtl_checks_coverage"] = cov
        if entries:
            # State the SCOPE of a clean result, per check, rather than claiming "hardware-legal": the
            # ledger below names which checks ran and which were dropped for want of a derivation, so a
            # capsule with no findings is read as "these checks found nothing", never as "all is well".
            ran = sorted({c for e in entries for c in (e.get("checks_grounded") or [])})
            skipped = sorted({c for e in entries for c in (e.get("checks_dropped") or {})})
            verdict["rtl_checks_note"] = (
                "ADVISORY RTL-derived checks (FileCheck + static screen over the emitted instruction "
                "stream; bounds from CIRCT-extracted hardware facts). Does NOT gate pass/fail. Fix "
                "encoding/tile/capacity findings before the RTL oracle would. A clean result covers ONLY "
                f"the checks that ran: {', '.join(ran) or 'none'}"
                + (f"; NOT run (no derivation): {', '.join(skipped)}" if skipped else "")
                + f". Screened {cov['screened']} of {cov['run_dirs']} capsule run(s). It says nothing "
                  "about numerics.")
        else:
            # NEVER let "checked nothing" render as "found nothing wrong". This is the whole reason the arm
            # read as working for 18 rounds: an empty findings list with an approving note beside it.
            verdict["rtl_checks_note"] = (
                f"RTL CHECKS DID NOT RUN — 0 of {cov['run_dirs']} capsule run(s) could be screened. The "
                "check families consume a self-hosted kernel assembly (generated/kernel.S), a decoded RoCC "
                f"trace (generated/instruction_trace.json), or the recorded machine-code words a compiled "
                f"lowering produces (generated/{_OS.WORDS_ARTIFACT}); this round produced "
                f"{cov['artifacts'].get('kernel.S', 0)}, "
                f"{cov['artifacts'].get('instruction_trace.json', 0)} and "
                f"{cov['artifacts'].get(_OS.WORDS_ARTIFACT, 0)} of those respectively. This is NOT a clean "
                "bill of health and NOT evidence the lowering is hardware-legal: nothing was inspected.")
    except Exception as e:  # advisory must never break the gate
        verdict["rtl_checks_error"] = repr(e)
    return verdict


def main(argv: list[str] | None = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
