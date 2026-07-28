#!/usr/bin/env python3
"""Redacted QA gate for the capsule_bench_v0 raw_baseline pilot.

Grades a candidate ``submission/`` against the PUBLIC pilot capsules (A0/A2/A4/B0) through the
real ladder (L0 reference==simulate, spike, verilator + trace_check), then emits ONLY a redacted
verdict the agent is allowed to see:

    {all_pass, n_passed, n_capsules, integrity_status,
     per_capsule: [{capsule, status, numeric_status, mismatch_count, trace_status,
                    trace_violations:[class-name strings], tiers:{L*:status}, failure_plane,
                    failure_category, highest_tier}],
     first_failure_planes}

It DELIBERATELY omits every answer-bearing value: golden outputs, reference/oracle outputs,
numeric diffs (max_abs_diff / first_mismatch), command buffers, lowered MLIR. The full grading
work tree (which contains numeric_report.yaml etc.) is written under an OPERATOR-ONLY runs_root
that the agent never sees; only the scrubbed verdict crosses back.

The agent uses this as a pass/fail QA signal to iterate against — never as an answer key.

Usage:
  qa_check.py --submission <dir> --out <verdict.json> [--labels public,dev]
              [--runs-root <operator-only tmp>] [--no-oracle] [--timeout 900]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import _common as C

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.targetgen import capsule_grade as CG  # noqa: E402
from merlin.targetgen import capsule_runner as CR  # noqa: E402

PILOT_PUBLIC = C.EXP / "scripts" / "pilot_capsules"

# Allowed (answer-free) numeric fields. Everything else in the numeric block is dropped.
_SAFE_NUMERIC = {"status", "policy", "mismatch_count"}
# Strip any stray numeric literals from free-text detail so a value can never leak through.
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _redact_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    # keep the plane/category prose, scrub concrete numbers (could echo expected/actual values)
    return _NUM.sub("#", detail)[:240]


def _per_capsule_from_results(runs_root: Path) -> dict[str, dict]:
    """Read each capsule_result.json from the operator-only work tree and redact it."""
    out: dict[str, dict] = {}
    rr = runs_root / "runs" / CR.SUITE
    if not rr.exists():
        return out
    for cr in sorted(rr.glob("*/capsule_result.json")):
        try:
            r = json.loads(cr.read_text())
        except Exception:
            continue
        num = r.get("numeric") or {}
        fail = r.get("failure") or {}
        tiers = r.get("tiers") or {}
        out[r.get("capsule", cr.parent.name)] = {
            "status": r.get("status"),
            "numeric_status": num.get("status"),
            "mismatch_count": num.get("mismatch_count"),
            "trace_status": (r.get("trace_check") or {}).get("status"),
            "trace_violations": list((r.get("trace_check") or {}).get("violations") or []),
            "tiers": {t: (tiers.get(t) or {}).get("status") for t in tiers},
            "tier_cycles": {t: (tiers.get(t) or {}).get("cycles") for t in tiers
                            if (tiers.get(t) or {}).get("cycles") is not None},
            "failure_plane": fail.get("plane"),
            "failure_category": fail.get("category"),
            "failure_detail": _redact_detail(fail.get("detail")),
        }
    return out


def _loop_target_sim_via() -> tuple[str, str]:
    """Resolve (target, sim_via) for the loop-gate oracle from THIS experiment's descriptor (honors the
    MERLIN_TARGET_EXPERIMENT override baked into C.EXP). Falls back to C.TARGET + no bespoke sim if the
    descriptor is absent, so a descriptor-less invocation still grades on the RTL-derived (arc) tier."""
    desc = C.EXP / "target_experiment.yaml"
    if desc.is_file():
        from merlin.targetgen.target_experiment import load_target_experiment
        te = load_target_experiment(desc)
        return te.target, te.sim_via
    return C.TARGET, ""


def run(submission: str, capsules_root: str, runs_root: Path, labels: set[str],
        no_oracle: bool, timeout: int) -> dict:
    # Loop gate = L0+L1+trace + the target's FASTEST RTL oracle tier ONLY — for gemmini (sim_via=chipyard)
    # that is L2 (spike); the slower cycle-accurate tier (verilator L3) is the separate bounded checkpoint
    # (run_baseline_qa_loop). Per-round verilator on 20 capsules across 3 parallel arms is infeasible (CPU
    # storm). The adapters are resolved from the descriptor's target+sim_via via the shared factory, so a
    # non-chipyard target (arc/cyclotron) grades on its own RTL-derived tier with NO gemmini-specific path.
    _target, _sim_via = _loop_target_sim_via()
    _loop_adapters = {} if no_oracle else CR.qa_loop_adapters(_target, _sim_via)
    score = CG.grade(submission, capsules_root=capsules_root, runs_root=str(runs_root),
                     labels=labels, contract=str(C.REPO / "merlin/contract"),
                     oracle_adapters=_loop_adapters, timeout=timeout, target=_target)
    redacted = _per_capsule_from_results(runs_root)

    per_capsule = []
    for pc in score.get("per_capsule", []):
        name = pc["capsule"]
        rich = redacted.get(name, {})
        per_capsule.append({
            "capsule": name,
            "label": pc.get("label"),
            "status": pc.get("status"),
            "numeric_status": rich.get("numeric_status", pc.get("numeric")),
            "mismatch_count": rich.get("mismatch_count"),
            "trace_status": rich.get("trace_status", pc.get("trace")),
            "trace_violations": rich.get("trace_violations", []),
            "tiers": pc.get("tiers", {}),
            "tier_cycles": rich.get("tier_cycles", {}),
            "failure_plane": rich.get("failure_plane"),
            "failure_category": rich.get("failure_category"),
            "failure_detail": rich.get("failure_detail"),
        })

    n_caps = score.get("n_capsules", 0)
    n_pass = score.get("n_passed", 0)
    verdict = {
        "qa_gate": "capsule_bench_v0_pilot",
        "labels_graded": score.get("labels_graded"),
        "all_pass": bool(n_caps > 0 and n_pass == n_caps),
        "n_passed": n_pass,
        "n_capsules": n_caps,
        "integrity_status": score.get("integrity_status"),
        "highest_tier": score.get("highest_tier"),
        "first_failure_planes": score.get("first_failure_planes", {}),
        "per_capsule": per_capsule,
        "note": ("This is a QA pass/fail signal only. It contains NO expected/golden values. "
                 "Fix failures by capsule + failure_plane + trace_violations; never hardcode outputs."),
    }
    # top-level integrity failure (K0/K1 fail-closed)
    if "failure" in score:
        verdict["package_failure"] = {"plane": score["failure"]["plane"],
                                      "category": score["failure"]["category"],
                                      "detail": _redact_detail(score["failure"]["detail"])}
    return verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True, help="path to write the redacted verdict JSON")
    ap.add_argument("--capsules-root", default=str(PILOT_PUBLIC))
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--runs-root", default=None,
                    help="OPERATOR-ONLY grading work tree (must NOT be inside the agent workspace)")
    ap.add_argument("--no-oracle", action="store_true", help="L0 + trace only (skip spike/verilator)")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args(argv)

    runs_root = Path(a.runs_root) if a.runs_root else (C.EXP / "runs" / "_qa_work" / "scratch")
    runs_root.mkdir(parents=True, exist_ok=True)
    labels = set(a.labels.split(","))
    verdict = run(a.submission, a.capsules_root, runs_root, labels, a.no_oracle, a.timeout)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(verdict, indent=2))
    print(f"[qa_check] all_pass={verdict['all_pass']} "
          f"{verdict['n_passed']}/{verdict['n_capsules']} integrity={verdict['integrity_status']}")
    for pc in verdict["per_capsule"]:
        extra = "" if pc["status"] == "pass" else f"  <- plane={pc['failure_plane']} viol={pc['trace_violations']}"
        print(f"    [{pc['status']:10s}] {pc['capsule']}{extra}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
