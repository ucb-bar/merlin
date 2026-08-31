#!/usr/bin/env python3
"""Gate: does every catalogued compiler pass have a reason to exist, and does anything run it?

Two failure modes, neither of which a pass-rate or a phase label can express:

1. **An undischarged pass.** A generated backend that adds passes until the score goes up ends up
   with transforms nobody can name a purpose for. So a pass may exist only to discharge one of the
   four obligations in :data:`merlin.xdsl_dialects.lowering.passes.OBLIGATIONS`
   (``partition/eligibility``, ``target transformation``, ``target lowering``,
   ``boundary materialization``), and a concrete capsule must declare the pass's requirement class.
   Either missing declaration is rejected.
2. **A declared-but-dead pass.** A pass in the catalog that no capsule run ever reaches is not part
   of the compiler; it is furniture. That cannot be decided from the catalog, so it is MEASURED: the
   catalog's entry points record their invocations to a JSONL log
   (``MERLIN_PASS_LOG=<path>``; :func:`passes.install_pass_recorder` wraps them at import), and this
   gate reads that log.

**With no log, this gate reports UNMEASURED and refuses to call anything dead.** A check that could
not run and reported success is a failure this repo has hit repeatedly (`codegen_smoke n/a -> true`
burned 101 minutes); ``--fail-on-dead`` with no log therefore exits 2 ("cannot decide"), never 0.
The log also carries an *install* record, so "instrumented and never invoked" (dead) stays
distinguishable from "never wrapped, so we did not look" (not_instrumented).

Modes, mirroring the sibling gates in this directory:

  --log PATH                 an invocation log to read (repeatable; default: $MERLIN_PASS_LOG)
  --json                     machine-readable
  --ratchet PATH             pre-existing debt that MAY ONLY SHRINK; unlisted new gaps fail
  --fail-on-undischarged     exit non-zero when a production pass has no allowed obligation
  --fail-on-unrequired       exit non-zero when no concrete capsule requires a production pass
  --fail-on-dead             exit non-zero when a non-ratcheted pass is measured dead
  --fail-on-unknown-dialect  exit non-zero when a non-ratcheted pass declares UNKNOWN dialects

Reporting-only by default: the catalog predates the obligation field, and turning day-one debt into
a hard failure only teaches everyone to pass `--no-verify`.

Ratchet lines are ``<pass name> <axis>:<item>`` — scoped to BOTH the pass and the axis on purpose.
A bare pass name would let an accepted missing obligation silently excuse that same pass later going
dead, and a bare axis would let one pass's debt forgive every other pass's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.xdsl_dialects.lowering import passes as PS  # noqa: E402

# Statuses that mean a pass was measured and never ran. `not_instrumented` is NOT here: it means the
# measurement did not cover the pass, which is a different (and separately reported) fact.
_DEAD = ("dead",)


def _debt(pass_name: str, item: str, axis: str) -> str:
    """A ratchet entry, SCOPED TO ITS PASS AND ITS AXIS.

    One pass's accepted debt must not excuse another's, and one axis's must not excuse another's:
    accepting that a pass has no stated obligation says nothing about whether a capsule runs it.
    """
    return f"{pass_name} {axis}:{item}"


def _load_ratchet(p: Path | None) -> set[str]:
    if not p or not p.is_file():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def audit(logs: list[Path]) -> dict:
    """The catalog audit + the measured exercise report."""
    cat = PS.catalog()
    ex = PS.exercise_report(cat, logs=logs or None)
    rows = []
    for p in cat:
        st = ex["per_pass"][p.name]
        rows.append({
            "name": p.name,
            "stage": p.stage,
            "entry": p.entry,
            "input_dialect": p.input_dialect,
            "output_dialect": p.output_dialect,
            "obligation": p.obligation,
            "required_by": list(p.required_by),
            "discharges": p.discharges(),
            "is_required": p.is_required(),
            "exercise": st["status"],
            "capsules": st["capsules"],
            "required_hits": st["required_hits"],
            "install": st["install"],
        })
    return {
        "n_passes": len(rows),
        "obligations": list(PS.OBLIGATIONS),
        "logs_read": ex["logs_read"],
        "unreadable_log_lines": ex["unreadable"],
        "measured": bool(ex["logs_read"]),
        "passes": rows,
    }


def findings(rep: dict, ratchet: set[str]) -> dict[str, list[dict]]:
    """Group the audit into the three gated axes, marking each item ratcheted or new."""
    out: dict[str, list[dict]] = {"undischarged": [], "unrequired": [],
                                 "unknown_dialect": [], "dead": [],
                                 "not_instrumented": [], "unattributed": [],
                                 "wrong_capsule": []}
    for r in rep["passes"]:
        if not r["discharges"]:
            key = _debt(r["name"], "undeclared", "obligation")
            out["undischarged"].append({"pass": r["name"], "debt": key,
                                        "ratcheted": key in ratchet})
        if not r["is_required"]:
            key = _debt(r["name"], "no-capsule-declares-it", "requirement")
            out["unrequired"].append({"pass": r["name"], "debt": key,
                                      "ratcheted": key in ratchet})
        unknown = [k for k in ("input_dialect", "output_dialect") if r[k] == PS.UNKNOWN]
        if unknown:
            key = _debt(r["name"], ",".join(unknown), "dialect")
            out["unknown_dialect"].append({"pass": r["name"], "fields": unknown, "debt": key,
                                           "ratcheted": key in ratchet})
        if r["exercise"] in _DEAD:
            key = _debt(r["name"], "no-capsule-runs-it", "exercise")
            out["dead"].append({"pass": r["name"], "debt": key, "ratcheted": key in ratchet})
        elif r["exercise"] == "not_instrumented" and rep["measured"]:
            out["not_instrumented"].append({"pass": r["name"], "install": r["install"]})
        elif r["exercise"] == "exercised_unattributed":
            out["unattributed"].append({"pass": r["name"]})
        elif r["exercise"] == "exercised_wrong_capsule":
            out["wrong_capsule"].append({"pass": r["name"], "capsules": r["capsules"],
                                          "required_by": r["required_by"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", type=Path, default=[],
                    help="invocation log to read (repeatable); default: $" + PS.PASS_LOG_ENV)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=None)
    ap.add_argument("--fail-on-undischarged", action="store_true")
    ap.add_argument("--fail-on-unrequired", action="store_true")
    ap.add_argument("--fail-on-dead", action="store_true")
    ap.add_argument("--fail-on-unknown-dialect", action="store_true")
    a = ap.parse_args(argv)

    logs = list(a.log)
    if not logs:
        env = (os.environ.get(PS.PASS_LOG_ENV) or "").strip()
        if env:
            logs = [Path(env)]

    rep = audit(logs)
    ratchet = _load_ratchet(a.ratchet)
    f = findings(rep, ratchet)

    if a.json:
        print(json.dumps({"report": rep, "findings": f}, indent=2))
    else:
        print(f"== authored pass catalog: {rep['n_passes']} pass(es)")
        for r in rep["passes"]:
            mark = " " if r["discharges"] else "*"
            req = f" required_by={r['required_by']}" if r["required_by"] else " required_by=[]"
            print(f"  {mark} {r['name']:34s} {r['input_dialect']:>18s} -> "
                  f"{r['output_dialect']:<18s} {r['obligation']:<24s} {r['exercise']}{req}")
        print(f"\n  undischarged (no valid production obligation): {len(f['undischarged'])}")
        for it in f["undischarged"]:
            print(f"    {' ' if it['ratcheted'] else '*'} {it['pass']}")
        print(f"  no requiring capsule: {len(f['unrequired'])}")
        for it in f["unrequired"]:
            print(f"    {' ' if it['ratcheted'] else '*'} {it['pass']}")
        print(f"  UNKNOWN dialect field(s): {len(f['unknown_dialect'])}")
        for it in f["unknown_dialect"]:
            print(f"    {' ' if it['ratcheted'] else '*'} {it['pass']:34s} {it['fields']}")
        if not rep["measured"]:
            # The load-bearing sentence of this gate: silence here is ignorance, not health.
            print("\n  exercise: UNMEASURED — no invocation log was read, so NO pass can be called "
                  f"dead and none can be called live.\n    Record one with "
                  f"{PS.PASS_LOG_ENV}=<path> during a capsule run, then re-run with --log <path>.")
        else:
            print(f"\n  logs read: {rep['logs_read']}")
            live = [r for r in rep["passes"] if r["exercise"].startswith("exercised")]
            print(f"  exercised by a capsule run: {len(live)} / {rep['n_passes']}")
            print(f"  declared but DEAD (instrumented, never invoked): {len(f['dead'])}")
            for it in f["dead"]:
                print(f"    {' ' if it['ratcheted'] else '*'} {it['pass']}")
            if f["not_instrumented"]:
                print(f"  NOT INSTRUMENTED (we did not look — this is not 'dead'): "
                      f"{len(f['not_instrumented'])}")
                for it in f["not_instrumented"]:
                    print(f"    ? {it['pass']:34s} {it['install']}")
            if f["unattributed"]:
                print(f"  ran, but under no capsule context ({len(f['unattributed'])}): "
                      f"set {PS.PASS_LOG_CAPSULE_ENV} or use passes.pass_run_context()")
                for it in f["unattributed"]:
                    print(f"    ? {it['pass']}")
            if f["wrong_capsule"]:
                print(f"  WRONG CAPSULE ({len(f['wrong_capsule'])}): a pass ran, but none of the "
                      "capsules that require it caused the run")
                for it in f["wrong_capsule"]:
                    print(f"    ? {it['pass']}: ran under {it['capsules']}, requires "
                          f"{it['required_by']}")
        if rep["unreadable_log_lines"]:
            print(f"  UNREADABLE log entries: {rep['unreadable_log_lines']}")

    rc = 0
    if a.fail_on_undischarged:
        new = [it for it in f["undischarged"] if not it["ratcheted"]]
        if new:
            print(f"\nFAIL: {len(new)} pass(es) discharge no production obligation",
                  file=sys.stderr)
            rc = 1
    if a.fail_on_unrequired:
        new = [it for it in f["unrequired"] if not it["ratcheted"]]
        if new:
            print(f"\nFAIL: {len(new)} production pass(es) have no requiring capsule",
                  file=sys.stderr)
            rc = 1
    if a.fail_on_unknown_dialect:
        new = [it for it in f["unknown_dialect"] if not it["ratcheted"]]
        if new:
            print(f"\nFAIL: {len(new)} pass(es) declare an UNKNOWN dialect", file=sys.stderr)
            rc = 1
    if a.fail_on_dead:
        if not rep["measured"]:
            # Exit 2, not 0: "we could not measure" must never be spelled the same way as "clean".
            print("\nCANNOT DECIDE: --fail-on-dead needs an invocation log and none was read. "
                  f"Run a capsule with {PS.PASS_LOG_ENV}=<path> and pass --log <path>.",
                  file=sys.stderr)
            return 2
        new = [it for it in f["dead"] if not it["ratcheted"]]
        wrong = list(f["wrong_capsule"])
        if new or wrong:
            print(f"\nFAIL: {len(new)} declared pass(es) are dead and {len(wrong)} ran only under "
                  "non-requiring capsules", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
