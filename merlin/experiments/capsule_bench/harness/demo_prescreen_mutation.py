"""Demonstrate the RTL pre-screen's iteration-saving by MUTATION TESTING vs the real RTL sim (verilator).

The honest gap (from corroboration on 383 real runs): the corpus had no real RTL-tier codegen failures, so
the pre-screen was proven *safe* (0 false-rejects) but never shown *catching* a real RTL failure cheaply.
Here we inject realistic codegen-bug classes into a known-good kernel, and for each mutant run BOTH:
  * the real RTL sim (verilator) — the ground-truth verdict + wall time it costs, and
  * the RTL pre-screen — verdict + wall time (ms),
showing the pre-screen rejects the genuine RTL failures in ms (saving the verilator run), passes the
unmutated original (0 false-positive), and HONESTLY misses a pure-numerical mutation (needs the oracle).

Both the ELF (for verilator) and the decoded trace (for the pre-screen) derive from one artifact —
`generated/lowered.llvm.mlir` — so a single text mutation propagates consistently to both.

Usage: demo_prescreen_mutation.py [--pkg <dir with command_buffer.json+lowered.llvm.mlir>] [--timeout 400]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python
REPO = C.REPO

import yaml  # noqa: E402
from merlin.targetgen import rocc_decode as RD            # noqa: E402
from merlin.targetgen import rtl_check_compiler as CC     # noqa: E402
from merlin.targetgen import rtl_check_runner as RUN      # noqa: E402
from merlin.targetgen import rtl_checks as RC             # noqa: E402
from merlin.targetgen import capsule_golden as CG         # noqa: E402
from merlin.targetgen.contract import compile as COMPILE  # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402

_TE = load_target_experiment(C.EXP / "target_experiment.yaml")
DEFAULT_PKG = (REPO / "out/runs/grade_subset_check/runs" / f"{C.TARGET}-capsule-bench"
               / "A2_single_tile_matmul" / "generated")
CAPSULE = REPO / "merlin/contract/capsules/isa/A2_single_tile_matmul/capsule.yaml"
# rtl_facts_pin = merlin/targets/<target>/contracts/rtl_facts/ (derived from the descriptor's target).
FACTS = json.loads((REPO / _TE.rtl_facts_pin / "facts.json").read_text())


# ------------------------------------------------------------------------------------- mutators
def m_original(mlir: str) -> str:
    return mlir


def m_illegal_funct(mlir: str) -> str:
    # COMPUTE_PRELOADED (funct 4) -> funct 99 (outside the RTL legal set {0..25})
    return re.sub(r"(\.insn r 0x7b, 0x3, )4(,)", r"\g<1>99\g<2>", mlir, count=1)


def _drop_insn(mlir: str, funct: int) -> str:
    out = []
    dropped = False
    for ln in mlir.splitlines():
        if not dropped and re.search(rf"\.insn r 0x7b, 0x3, {funct},", ln):
            dropped = True
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def m_drop_compute(mlir: str) -> str:
    return _drop_insn(mlir, 4)   # remove the matrix COMPUTE


def m_drop_mvout(mlir: str) -> str:
    return _drop_insn(mlir, 3)   # remove the result store (MVOUT)


def m_swap_compute_operands(mlir: str) -> str:
    # numerical-only: swap the two operands of the COMPUTE .insn -> structurally valid, wrong math.
    def repl(mm):
        return f"{mm.group(1)}{mm.group(3)}, {mm.group(2)}{mm.group(4)}"
    return re.sub(r"(\.insn r 0x7b, 0x3, 4, x0, \$0, \$1\", \"r,r\" )(%\w+), (%\w+)( :)",
                  repl, mlir, count=1)


MUTATORS = [
    ("original", m_original, "ok", "pass"),
    ("illegal_funct", m_illegal_funct, "reject", "fail"),
    ("drop_compute", m_drop_compute, "reject", "fail"),
    ("drop_mvout", m_drop_mvout, "reject", "fail"),
    ("swap_compute_operands(numerical)", m_swap_compute_operands, "ok", "fail"),  # honest FN
]


# ------------------------------------------------------------------------------------- evaluators
def prescreen_verdict(mlir: str, capsule: dict, fc: str | None):
    t0 = time.perf_counter()
    trace = RD.decode_text(mlir, source="mutant", target=C.TARGET)
    cc = CC.compile_checks(FACTS, capsule)
    tr_ok = True
    if fc and cc["trace"]:
        tr_ok, _ = RUN.run_filecheck(fc, cc["trace"], RUN.render_trace(trace, FACTS), "TRACE")
    rep = RC.screen(trace, capsule, CC._facts_to_rc(FACTS), target=C.TARGET)
    verdict = "reject" if (tr_ok is False or rep.verdict == "reject") else (
        "warn" if rep.verdict == "warn" else "ok")
    ms = (time.perf_counter() - t0) * 1e3
    fails = [c.id for c in rep.checks if c.status == "fail"]
    if tr_ok is False:
        fails = ["TRACE.filecheck", *fails]
    return verdict, ms, fails


def verilator_verdict(cb: dict, mlir: str, gold_flat, timeout: int):
    t0 = time.perf_counter()
    try:
        r = COMPILE.run_on_oracle(cb, mlir, simulator="verilator", target=C.TARGET,
                                  timeout=timeout)
    except Exception as e:
        return "fail", round(time.perf_counter() - t0, 1), f"sim error: {type(e).__name__}: {str(e)[:80]}"
    sim_s = (r.get("timing") or {}).get("sim_active_s", round(time.perf_counter() - t0, 1))
    got = r.get("outputs", {}).get("Y0")
    if got is None:
        return "fail", sim_s, "no Y0 output committed"
    import numpy as np
    got_flat = np.asarray(got).flatten().astype(int).tolist()
    if got_flat == gold_flat:
        return "pass", sim_s, f"Y0==golden (cycles={r.get('cycles')})"
    return "fail", sim_s, f"Y0 != golden ({sum(1 for a,b in zip(got_flat,gold_flat) if a!=b)} mismatches)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=str(DEFAULT_PKG))
    ap.add_argument("--timeout", type=int, default=400)
    a = ap.parse_args(argv)
    pkg = Path(a.pkg)
    cb = json.loads((pkg / "command_buffer.json").read_text())
    base_mlir = (pkg / "lowered.llvm.mlir").read_text()
    capsule = yaml.safe_load(CAPSULE.read_text())
    import numpy as np
    gold_flat = np.asarray(CG.golden(capsule)["Y0"]).flatten().astype(int).tolist()
    fc = RUN.find_filecheck()

    rows = []
    print(f"{'mutant':34s} {'verilator':>10s} {'sim_s':>7s}   {'pre-screen':>10s} {'ms':>6s}  caught_first")
    for name, fn, exp_ck, exp_v in MUTATORS:
        mlir = fn(base_mlir)
        ck_v, ck_ms, ck_fails = prescreen_verdict(mlir, capsule, fc)
        v_v, v_s, v_note = verilator_verdict(cb, mlir, gold_flat, a.timeout)
        caught_first = (v_v == "fail" and ck_v != "ok")
        rows.append({"mutant": name, "verilator": v_v, "verilator_s": v_s, "verilator_note": v_note,
                     "prescreen": ck_v, "prescreen_ms": round(ck_ms, 1), "prescreen_fails": ck_fails,
                     "caught_before_rtl": caught_first})
        print(f"{name:34s} {v_v:>10s} {v_s:>7} {ck_v:>12s} {ck_ms:6.1f}  "
              f"{'YES' if caught_first else ('n/a(pass)' if v_v=='pass' else 'MISS(numeric)')}")

    # headline
    structural = [r for r in rows if r["mutant"] not in ("original",) and r["caught_before_rtl"]]
    saved = sum(float(r["verilator_s"]) for r in structural)
    fp = [r for r in rows if r["mutant"] == "original" and r["prescreen"] != "ok"]
    out = {"package": str(pkg), "rows": rows,
           "summary": {"structural_failures_caught_pre_RTL": len(structural),
                       "verilator_seconds_saved_est": round(saved, 1),
                       "false_positive_on_original": len(fp),
                       "numerical_miss_by_design": [r["mutant"] for r in rows
                                                    if r["verilator"] == "fail" and r["prescreen"] == "ok"]}}
    rep = C.REPORTS / "prescreen_mutation_demo.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=2))
    print(f"\nsummary: {out['summary']}\nwrote {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
