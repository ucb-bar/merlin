#!/usr/bin/env python3
"""Gate: does every catalogued compiler pass have a reason to exist, and does anything run it?

Three failure modes, none of which a pass-rate or a phase label can express:

1. **An undischarged pass.** A generated backend that adds passes until the score goes up ends up
   with transforms nobody can name a purpose for. So a pass may exist only to discharge one of the
   four obligations in :data:`merlin.xdsl_dialects.lowering.passes.OBLIGATIONS`
   (``partition/eligibility``, ``target transformation``, ``target lowering``,
   ``boundary materialization``), and a concrete capsule must declare the pass's requirement class.
   Either missing declaration is rejected.
2. **An unverified pass.** A pass a capsule reaches, that transforms something, and that nobody
   has ever checked *does the thing it declares*. A capsule grade is an outcome: it says this
   program compiled correctly this time on this target, not that the pass which tiled it is
   correct. The static (lit/FileCheck) and formal (SMT refinement) layers answer that question, and
   their verdicts reach this gate through a second JSONL log
   (``MERLIN_VERIFY_LOG=<path>``; :func:`passes.record_verification` writes it).
3. **A declared-but-dead pass.** A pass in the catalog that no capsule run ever reaches is not part
   of the compiler; it is furniture. That cannot be decided from the catalog, so it is MEASURED: the
   catalog's entry points record their invocations to a JSONL log
   (``MERLIN_PASS_LOG=<path>``; :func:`passes.install_pass_recorder` wraps them at import), and this
   gate reads that log.

**With no log, this gate reports UNMEASURED and refuses to call anything dead or unverified.** A
check that could not run and reported success is a failure this repo has hit repeatedly
(`codegen_smoke n/a -> true` burned 101 minutes); ``--fail-on-dead`` and ``--fail-on-unverified``
with no log therefore exit 2 ("cannot decide"), never 0.
The log also carries an *install* record, so "instrumented and never invoked" (dead) stays
distinguishable from "never wrapped, so we did not look" (not_instrumented).

Modes, mirroring the sibling gates in this directory:

  --log PATH                 an invocation log to read (repeatable; default: $MERLIN_PASS_LOG)
  --verify-log PATH          a verdict log to read (repeatable; default: $MERLIN_VERIFY_LOG)
  --json                     machine-readable
  --ratchet PATH             pre-existing debt that MAY ONLY SHRINK; unlisted new gaps fail
  --fail-on-undischarged     exit non-zero when a production pass has no allowed obligation
  --fail-on-unrequired       exit non-zero when no concrete capsule requires a production pass
  --fail-on-dead             exit non-zero when a non-ratcheted pass is measured dead
  --fail-on-noop             exit non-zero when a pass RAN but every invocation transformed nothing
                             ("invoked" and "did its work" are different facts: measured here, the
                             boundary capstone invoked the outliner once and outlined ZERO kernels,
                             and this gate reported it exercised)
  --fail-on-unknown-dialect  exit non-zero when a non-ratcheted pass declares UNKNOWN dialects
  --fail-on-unverified       exit non-zero when a non-ratcheted pass is REACHED by a capsule but no
                             static or formal layer has reached a verdict about it (needs BOTH logs:
                             "reached" is measured from one and "verified" from the other, so either
                             one missing is "cannot decide", not "clean")

**A refutation is a hard failure, always, and no ratchet may forgive it.** It needs no flag and is
not listed in the axes above, because it is not an axis: a ratchet accepts *absent* evidence, and a
refuted verdict is evidence we HAVE — a FileCheck assertion that did not match, or an SMT model that
is a concrete counterexample to the pass. Allowing that to be ratcheted would convert the one thing
this layer can prove into a line in a debt file.

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


def audit(logs: list[Path], verify_logs: list[Path] | None = None) -> dict:
    """The catalog audit + the measured exercise report + the measured verification report.

    The two logs are read independently and NEITHER is inferred from the other. A pass can be
    reached and unverified, or verified and dead (a lit test proves it does its job; nothing in the
    pipeline calls it) — those are different defects with different fixes, and a single "healthy"
    column would hide both.
    """
    cat = PS.catalog()
    ex = PS.exercise_report(cat, logs=logs or None)
    vr = PS.verification_report(cat, logs=verify_logs or None)
    rows = []
    for p in cat:
        st = ex["per_pass"][p.name]
        vs = vr["per_pass"][p.name]
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
            "effects": st.get("effects", {}),
            "effect_evidence": st.get("effect_evidence", []),
            "verification": vs["status"],
            "verdicts": vs["verdicts"],
            "verify_methods": vs["methods"],
            "verify_targets": vs["targets"],
            "verify_requirement_classes": vs["requirement_classes"],
            "verify_capsules": vs["capsules"],
            "verdict_evidence": vs["evidence"],
        })
    return {
        "n_passes": len(rows),
        "obligations": list(PS.OBLIGATIONS),
        "verdicts_vocabulary": list(PS.VERDICTS),
        "verify_methods_vocabulary": list(PS.METHODS),
        "logs_read": ex["logs_read"],
        "unreadable_log_lines": ex["unreadable"],
        "measured": bool(ex["logs_read"]),
        "verify_logs_read": vr["logs_read"],
        "unreadable_verify_log_lines": vr["unreadable"],
        "verify_measured": bool(vr["logs_read"]),
        # A verdict aimed at a name the catalog does not carry stops counting silently, which looks
        # exactly like never having run the check. Surfaced, never dropped.
        "verdicts_for_unknown_passes": vr["unknown_passes"],
        "passes": rows,
    }


def findings(rep: dict, ratchet: set[str]) -> dict[str, list[dict]]:
    """Group the audit into the three gated axes, marking each item ratcheted or new."""
    out: dict[str, list[dict]] = {"undischarged": [], "unrequired": [],
                                 "unknown_dialect": [], "dead": [],
                                 "not_instrumented": [], "unattributed": [],
                                 "wrong_capsule": [], "noop": [],
                                 "unverified": [], "refuted": []}
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
        elif r["exercise"] == "exercised_noop":
            key = _debt(r["name"], "no-capsule-reaches-its-work", "exercise")
            out["noop"].append({"pass": r["name"], "capsules": r["capsules"],
                                "effects": r["effects"], "evidence": r["effect_evidence"],
                                "debt": key, "ratcheted": key in ratchet})

        # ``.get`` with the fail-closed default: a row assembled without a verification report is
        # UNMEASURED, which claims nothing verified and charges nothing unverified. The gate's own
        # `audit()` always fills these in; the default exists so a caller that builds a row by hand
        # cannot accidentally get "verified" out of a field it never set.
        verification = r.get("verification", "unmeasured")
        # A refutation is reported the moment a verdict log carries one, regardless of whether the
        # pass was reached: a pass disproved by a static or formal layer is disproved whether or not
        # this particular run happened to invoke it. It carries NO debt key — deliberately, so that
        # no ratchet line can ever mark it accepted.
        if verification == "refuted":
            out["refuted"].append({"pass": r["name"], "verdicts": r.get("verdicts", {}),
                                   "methods": r.get("verify_methods", []),
                                   "targets": r.get("verify_targets", []),
                                   "evidence": r.get("verdict_evidence", [])})
        # REACHED but unverified. Both measurements are required: "reached" comes from the invocation
        # log and "verified" from the verdict log, so with either missing this is not an empty list,
        # it is an undecidable question — which `main` spells as exit 2 rather than as a clean axis.
        elif (rep["measured"] and rep.get("verify_measured", False)
                and str(r["exercise"]).startswith("exercised")
                and verification != "verified"):
            key = _debt(r["name"], "no-static-or-formal-verdict", "verification")
            out["unverified"].append({"pass": r["name"], "verification": verification,
                                      "exercise": r["exercise"], "verdicts": r.get("verdicts", {}),
                                      "methods": r.get("verify_methods", []),
                                      "evidence": r.get("verdict_evidence", []),
                                      "debt": key, "ratcheted": key in ratchet})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", action="append", type=Path, default=[],
                    help="invocation log to read (repeatable); default: $" + PS.PASS_LOG_ENV)
    ap.add_argument("--verify-log", action="append", type=Path, default=[],
                    help="verdict log to read (repeatable); default: $" + PS.VERIFY_LOG_ENV)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=None)
    ap.add_argument("--fail-on-undischarged", action="store_true")
    ap.add_argument("--fail-on-unrequired", action="store_true")
    ap.add_argument("--fail-on-dead", action="store_true")
    ap.add_argument("--fail-on-noop", action="store_true",
                    help="exit non-zero when a pass ran but every invocation transformed nothing")
    ap.add_argument("--fail-on-unknown-dialect", action="store_true")
    ap.add_argument("--fail-on-unverified", action="store_true",
                    help="exit non-zero when a capsule-reached pass has no static or formal verdict")
    a = ap.parse_args(argv)

    logs = list(a.log)
    if not logs:
        env = (os.environ.get(PS.PASS_LOG_ENV) or "").strip()
        if env:
            logs = [Path(env)]

    verify_logs = list(a.verify_log)
    if not verify_logs:
        env = (os.environ.get(PS.VERIFY_LOG_ENV) or "").strip()
        if env:
            verify_logs = [Path(env)]

    rep = audit(logs, verify_logs)
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
                  f"{r['output_dialect']:<18s} {r['obligation']:<24s} {r['exercise']}"
                  f"/{r['verification']}{req}")
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
            live = [r for r in rep["passes"] if r["exercise"] == "exercised"]
            print(f"  exercised by a capsule run AND measured to do work: "
                  f"{len(live)} / {rep['n_passes']}")
            if f["noop"]:
                # The distinction this gate exists to keep: ran, and reached none of its work.
                print(f"  RAN TO NO EFFECT ({len(f['noop'])}): invoked, but every invocation left "
                      "the IR unchanged and produced nothing")
                for it in f["noop"]:
                    ev = it["evidence"][0] if it["evidence"] else {}
                    print(f"    {' ' if it['ratcheted'] else '*'} {it['pass']:34s} "
                          f"under {it['capsules']} effects={it['effects']} "
                          f"produced={ev.get('produced')} ({ev.get('product_read')})")
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
        # --- the verification layers ----------------------------------------------------------
        if not rep["verify_measured"]:
            # Same sentence as the exercise block above, for the same reason: nothing was checked is
            # not the same claim as everything checks out, and only one of them is what we know.
            print("\n  verification: UNMEASURED — no verdict log was read, so NO pass can be called "
                  f"verified and none can be called unverified.\n    Record one with "
                  f"{PS.VERIFY_LOG_ENV}=<path> while running the static/formal layers, then re-run "
                  "with --verify-log <path>.")
        else:
            print(f"\n  verdict logs read: {rep['verify_logs_read']}")
            proven = [r for r in rep["passes"] if r["verification"] == "verified"]
            print(f"  verified by a static or formal layer: {len(proven)} / {rep['n_passes']}")
            if f["refuted"]:
                print(f"  REFUTED ({len(f['refuted'])}): a layer DISPROVED the pass. This is not "
                      "debt and cannot be ratcheted.")
                for it in f["refuted"]:
                    ev = it["evidence"][0] if it["evidence"] else {}
                    print(f"    ! {it['pass']:34s} by {it['methods']} on {it['targets']} "
                          f"evidence={ev.get('evidence')}")
            if not rep["measured"]:
                print("  reached-but-unverified: CANNOT DECIDE — the invocation log says which "
                      "passes a capsule reaches, and none was read.")
            else:
                print(f"  reached but UNVERIFIED (no static or formal verdict): "
                      f"{len(f['unverified'])}")
                for it in f["unverified"]:
                    print(f"    {' ' if it['ratcheted'] else '*'} {it['pass']:34s} "
                          f"{it['exercise']} / {it['verification']} verdicts={it['verdicts']}")
            if rep["verdicts_for_unknown_passes"]:
                print(f"  verdicts against names the catalog does not carry (evidence that stopped "
                      f"counting): {rep['verdicts_for_unknown_passes']}")
        if rep["unreadable_log_lines"]:
            print(f"  UNREADABLE log entries: {rep['unreadable_log_lines']}")
        if rep["unreadable_verify_log_lines"]:
            print(f"  UNREADABLE verdict log entries: {rep['unreadable_verify_log_lines']}")

    rc = 0
    # Unconditional, and checked FIRST: a refutation is not an axis and not debt. No flag enables it
    # and no ratchet line forgives it — a gate that let a disproof be marked "accepted" would take
    # the only thing this layer can positively establish and file it away.
    if f["refuted"]:
        print(f"\nFAIL: {len(f['refuted'])} pass(es) were REFUTED by a static or formal layer "
              "(a failed check or a concrete counterexample); this is a disproof, not debt",
              file=sys.stderr)
        rc = 1
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
    if a.fail_on_noop:
        if not rep["measured"]:
            print("\nCANNOT DECIDE: --fail-on-noop needs an invocation log and none was read.",
                  file=sys.stderr)
            return 2
        new = [it for it in f["noop"] if not it["ratcheted"]]
        if new:
            print(f"\nFAIL: {len(new)} pass(es) ran but reached none of their work; a capsule that "
                  "invokes a pass to no effect does not certify it", file=sys.stderr)
            rc = 1
    if a.fail_on_unverified:
        if not rep["verify_measured"] or not rep["measured"]:
            # Exit 2, not 0 — verbatim the dead-pass rule. The axis is "REACHED but unverified", so
            # it needs both measurements: without the verdict log nothing is known to be verified,
            # and without the invocation log nothing is known to be reached. Either absence makes the
            # empty list a statement about our instrumentation, not about the compiler.
            missing = []
            if not rep["verify_measured"]:
                missing.append(f"a verdict log ({PS.VERIFY_LOG_ENV}=<path>, --verify-log <path>)")
            if not rep["measured"]:
                missing.append(f"an invocation log ({PS.PASS_LOG_ENV}=<path>, --log <path>)")
            print("\nCANNOT DECIDE: --fail-on-unverified needs " + " and ".join(missing)
                  + ", and none was read.", file=sys.stderr)
            # `rc or 2`: an undecidable axis must not swallow a refutation that WAS decided.
            return rc or 2
        new = [it for it in f["unverified"] if not it["ratcheted"]]
        if new:
            print(f"\nFAIL: {len(new)} pass(es) are reached by a capsule but no static or formal "
                  "layer has reached a verdict about them; a capsule grade is an outcome, not a "
                  "proof that the pass is correct", file=sys.stderr)
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
