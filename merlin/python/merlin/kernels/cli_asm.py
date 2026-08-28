"""``merlin-asm-audit`` — what we understand of a target's assembly, and what to try next.

Repeatable on purpose. The measurements this reports (how much of a stream carries meaning, which CCA
facets a target can populate, which optimizations its assembly implies) go stale whenever the compiler,
the corpus or a role table changes, so they belong in a command anyone can re-run rather than in a
report someone wrote once.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def _audits(target: str, patterns):
    from merlin.kernels import asm_audit as A
    out = []
    for pat in patterns or ():
        for path in sorted(glob.glob(pat)):
            try:
                # EVERY endpoint: auditing a multi-engine target through one hides the other engine's
                # work silently, which is the failure the engine model exists to prevent.
                out.extend(A.audit_every_endpoint(path, target, stream=path))
            except Exception as exc:  # noqa: BLE001 — a stream we cannot read is REPORTED, not dropped
                out.append(A.AsmAudit(target=target, notes=(f"{path}: {type(exc).__name__}: {exc}",)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True,
                    help="the target whose endpoints and assembly are audited")
    ap.add_argument("--stream", action="append", default=[],
                    help="glob of compiled objects/ELFs to audit (repeatable)")
    ap.add_argument("--asm", action="append", default=[],
                    help="glob of hand-written .S kernels to audit (repeatable)")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    a = ap.parse_args(argv)

    from merlin.kernels import asm_audit as A
    from merlin.kernels import asm_provenance as P

    audits = _audits(a.target, a.stream)
    for pat in a.asm:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8", errors="replace") as fh:
                audits.extend(A.audit_every_endpoint(fh.read().splitlines(), a.target,
                                                    text=True, stream=path))

    report = A.target_report(a.target, audits)
    hist: dict = {}
    engine = ""
    for x in audits:
        engine = x.engine or engine
        for k, v in x.role_histogram.items():
            hist[k] = hist.get(k, 0) + v
    # One stream read through several endpoints is still ONE stream. Summing per audit multiplies the
    # instruction count by the number of engines and deflates every coverage fraction by the same
    # factor -- a number that looks like a measurement and is an artefact of how it was gathered.
    merged = A.merge_audits(audits) if audits else {"per_engine": {}}
    # Counted ONCE per stream by the merge, rather than divided by the endpoint count: dividing is
    # only right when every endpoint read every stream, and it silently rounds when they did not.
    total = merged.get("total") or 0
    report["per_engine"] = merged.get("per_engine", {})
    # Opportunities are derived PER ENGINE: a lane engine's elementwise work and an array's accumulate
    # are different work on different silicon, and a pooled histogram reads as one machine doing all
    # of it -- which would propose an array optimization from a vector kernel's shape.
    opps = []
    for eng, slot in (merged.get("per_engine") or {}).items():
        opps += [o.to_dict() for o in P.opportunities(slot["roles"], engine=eng, total=total)]
    report["opportunities"] = opps
    # Kept for the JSON consumers, and explicitly NOT the headline: it double-counts roles that more
    # than one endpoint claims.
    report["role_totals_pooled"] = dict(sorted(hist.items()))

    if a.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"target: {a.target}   usable={report['usable']}   streams={len(audits)}")
    for e in report["endpoints"]:
        print(f"  endpoint {e['endpoint']} [{e['engine']}] via {e['source']}")
        print(f"    roles declared : {', '.join(e['roles_declared']) or '(none)'}")
        if e["identities_without_a_role"]:
            print(f"    NO ROLE for    : {', '.join(e['identities_without_a_role'])}")
        unreachable = [f for f, v in e["facets"].items() if not v["reachable"]]
        print(f"    facets reachable: {len(e['facets']) - len(unreachable)}/{len(e['facets'])}"
              + (f"   blocked: {', '.join(unreachable)}" if unreachable else ""))
    if audits:
        sem = report["observed"]["semantic_fraction"]
        print(f"  observed: {total} instruction(s), {sem:.1%} carry a role")
        # The four-way split is the invariant this command exists to state, so it is printed, not
        # only serialized. Per endpoint, because one stream read two ways yields two readings.
        cov = report.get("coverage") or {}
        for ep, c in sorted((cov.get("per_endpoint") or {}).items()):
            flag = "" if c.get("sums") else "   !! DOES NOT SUM"
            print(f"    {ep:16s} named={c['named_by_tool']:7d} roled={c['role_tagged']:6d} "
                  f"claimed_no_role={c['claimed_no_role']:5d} unaccounted={c['unaccounted']:6d}{flag}")
        una, frac = cov.get("unaccounted_by_every_endpoint"), cov.get("unaccounted_fraction")
        if una is None:
            print("    unaccounted by EVERY endpoint: UNKNOWN (a count derived by subtraction "
                  "cannot be intersected) — not zero")
        else:
            print(f"    unaccounted by EVERY endpoint: {una}"
                  + (f" ({frac:.2%})" if frac is not None else ""))
            widths = cov.get("unaccounted_widths") or {}
            if widths:
                # An entry narrower than the ISA's minimum instruction width is not an instruction at
                # all, so a single total conflates a decoder gap with bytes objdump could not form.
                parts = ", ".join(f"{b}b:{n}" for b, n in sorted(widths.items(), key=lambda kv: int(kv[0])))
                print(f"      by hex-column width: {parts}")
        # PER ENGINE, not pooled. Roles shared by two endpoints (every endpoint moves data) would be
        # counted once per endpoint in a pooled line, and a lane engine's elementwise work pooled with
        # an array's accumulate reads as one machine doing all of it.
        for eng, slot in sorted((report.get("per_engine") or {}).items()):
            print(f"    {eng:9} via {','.join(slot['endpoints'])}: {dict(sorted(slot['roles'].items()))}")
        if report["observed"]["declared_but_never_seen"]:
            print(f"    declared but never seen: "
                  f"{', '.join(report['observed']['declared_but_never_seen'])}")
    for b in report["blocking"]:
        print(f"  ! {b}")
    if report["opportunities"]:
        print("  optimizations implied by this assembly:")
        for o in report["opportunities"]:
            print(f"    [{o['confidence']}] {o['axis']}  ({o['status']})")
            print(f"        saw : {o['observation']}")
            print(f"        try : {o['change']}")
            if o["seam"]:
                print(f"        seam: {o['seam']}")
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
