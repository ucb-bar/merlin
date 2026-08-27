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
                out.append(A.audit_stream(path, target))
            except Exception as exc:  # noqa: BLE001 — a stream we cannot read is REPORTED, not dropped
                a = A.AsmAudit(target=target, notes=(f"{path}: {type(exc).__name__}: {exc}",))
                out.append(a)
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
                audits.append(A.audit_text(fh.read().splitlines(), a.target))

    report = A.target_report(a.target, audits)
    hist: dict = {}
    total = 0
    engine = ""
    for x in audits:
        total += x.total
        engine = x.engine or engine
        for k, v in x.role_histogram.items():
            hist[k] = hist.get(k, 0) + v
    report["opportunities"] = [o.to_dict() for o in P.opportunities(hist, engine=engine, total=total)]
    report["role_totals"] = dict(sorted(hist.items()))

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
        print(f"    roles: {report['role_totals']}")
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
