#!/usr/bin/env python3
"""Gate: does a target's capability MANIFEST still describe the hardware its own sources evidence?

The manifest is prose somebody wrote; the RTL and the target's own pinned ISA header are not. When the
two drift, nothing notices, and both directions cost a real number:

  UNDER-DECLARED  the sources evidence a capability the manifest omits. This does NOT show up as a
                  failing capsule — the conformance requirement is ``admitted INTERSECT observed``
                  (:mod:`merlin.targetgen.conformance`), so an unadmitted family is EXCLUDED from the
                  requirement rather than missed, and the bar for the whole corpus quietly drops.
                  Measured here: a systolic target's contract says in a comment "No general reduction /
                  softmax / attention / normalization hardware" while its own ISA header enumerates
                  ``LAYERNORM`` and ``SOFTMAX`` as accumulator-readout activation modes.

  OVER-DECLARED   the manifest claims a capability and the rung that could have evidenced it RAN and
                  found nothing. That becomes permanent ``false_fallback`` no compiler change can clear,
                  and ships capsules the hardware may not be able to execute.

  ENCODABLE       the ISA encodes it and the ELABORATED design does not contain it. Its own list, and
  NOT BUILT       neither of the two above: adding it to the manifest would be over-declaration, and
                  calling it absent would deny an encoding that demonstrably exists. Measured here:
                  three accumulator-readout activation modes with a `#define` apiece, all gated on a
                  configuration field whose declared default is false.

  UNDETERMINABLE  no rung capable of deciding was available (no fact bundle, no readable ISA source, no
                  parseable contract). Reported in its OWN list and never folded into either direction:
                  "we could not look" must never license deleting a capability from a manifest.

Three rungs, and the ranking matters: `rtl_facts` (extracted hardware) and `build_config` (the
configuration the fact bundle itself names as the one its RTL was elaborated from) both outrank
`isa_header`, because a header states what the ISA can ENCODE and only the configuration says what was
BUILT.

Modes, mirroring the other gates in this directory:

  --target NAME              audit one target (repeatable); default: every target with an RTL fact bundle
  --json                     machine-readable
  --ratchet PATH             pre-existing debt that MAY ONLY SHRINK; unlisted new drift is what fails
  --fail-on-under-declared   exit non-zero on non-ratcheted under-declaration
  --fail-on-over-declared    exit non-zero on non-ratcheted over-declaration
  --require-pin              refuse to report a surface whose hardware pin does not verify (default:
                             report it, loudly, with the mismatch in the notes — a gate that dies on a
                             colleague's checkout stops being run)

Reporting-only by default. The derivation is new and every manifest in the tree predates it; turning a
day-one delta into a hard failure only teaches everyone to pass ``--no-verify``.

Ratchet entries are SCOPED TO THEIR TARGET AND DIRECTION -- ``<target> <direction>:<kind>:<name>`` --
because a bare family name would let one target's accepted debt silently excuse another's, and
``contraction`` is a real entry on more than one target here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.targetgen import capability_discovery as CD  # noqa: E402


def audit(target: str, *, require_pin: bool = False) -> dict:
    """The declared-vs-discovered delta for one target, or the reason there is none."""
    try:
        return CD.delta(target, require_pin=require_pin)
    except CD.ProvenanceRefused as e:
        return {"target": target, "status": "pin_unverified", "detail": str(e),
                "under_declared": [], "over_declared": [],
                "undeterminable": [{"kind": "provenance", "name": target, "detail": str(e)}],
                "discovered_families": [], "declared_families": [], "datapath_dtypes": {},
                "rungs_ran": [], "notes": [], "sources": []}


def _debt(target: str, direction: str, kind: str, name: str) -> str:
    return f"{target} {direction}:{kind}:{name}"


def _load_ratchet(p: Path | None) -> set[str]:
    if not p or not p.is_file():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def _print_report(r: dict, ratchet: set[str]) -> None:
    t = r["target"]
    status = r.get("status", "ok")
    if status != "ok":
        print(f"== {t}: {status} — {r.get('detail', '')[:200]}")
        for u in r.get("undeterminable", []):
            print(f"   ? {u['kind']}:{u['name']}  {u.get('detail', '')[:140]}")
        return
    print(f"== {t}   rungs that ran: {r.get('rungs_ran') or ['none']}")
    bc = r.get("build_config") or {}
    if bc:
        print(f"   build config        : {bc.get('name')} -> {bc.get('instantiated')} "
              f"({len(bc.get('fields') or {})} field(s), "
              f"{sum(1 for f in (bc.get('fields') or {}).values() if f.get('origin') == 'set')} set / "
              f"{sum(1 for f in (bc.get('fields') or {}).values() if f.get('origin') == 'declared_default')}"
              f" declared default)")
    pins = r.get("source_pin_status") or {}
    off = {k: v for k, v in pins.items() if v not in ("pinned", "nested_pinned")}
    if off:
        print(f"   NOT PINNED          : {off} — header-derived claims from these files are real but "
              f"are NOT claims about the pinned revision")
    print(f"   discovered families : {r['discovered_families'] or '-'}")
    print(f"   declared   families : {r['declared_families'] or '-'}")
    for role, dts in sorted((r.get("datapath_dtypes") or {}).items()):
        print(f"   datapath {role:16s}: {dts}")
    if r["under_declared"]:
        print(f"   UNDER-DECLARED ({len(r['under_declared'])}) — evidenced, not admitted; this LOWERS "
              f"the conformance bar rather than failing a capsule")
        for u in r["under_declared"]:
            mark = " " if _debt(t, "under", u["kind"], u["name"]) in ratchet else "*"
            ev = (u.get("evidence") or [{}])[0]
            where = ev.get("locator", "")
            if ev.get("line"):
                where = f"{where}:{ev['line']}"
            print(f"     {mark} {u['kind']:14s} {u['name']:22s} {ev.get('observed', '')[:70]}")
            if where:
                print(f"       evidence: {where}")
    if r["over_declared"]:
        print(f"   OVER-DECLARED ({len(r['over_declared'])}) — claimed, and the deciding rung ran and "
              f"found nothing")
        for o in r["over_declared"]:
            mark = " " if _debt(t, "over", o["kind"], o["name"]) in ratchet else "*"
            print(f"     {mark} {o['kind']:14s} {o['name']:22s} {o.get('detail', '')[:80]}")
    enb = r.get("encodable_not_built") or []
    if enb:
        print(f"   ENCODABLE BUT NOT BUILT ({len(enb)}) — the ISA encodes it, this elaboration does "
              f"not contain it; declaring any of these would be OVER-declaration")
        for f in enb:
            g = f.get("gate") or {}
            print(f"     ! {f['axis']:14s} {f['name']:22s} gated on {g.get('off')} = false in "
                  f"{g.get('config')}")
            ev = (f.get("evidence") or [{}])[-1]
            if ev.get("locator"):
                print(f"       evidence: {ev['locator']}"
                      + (f":{ev['line']}" if ev.get("line") else "")
                      + f"  {ev.get('observed', '')[:70]}")
    if r["undeterminable"]:
        print(f"   undeterminable ({len(r['undeterminable'])}) — no rung capable of deciding ran; NOT "
              f"absence, and never a licence to remove a declaration")
        for u in r["undeterminable"][:12]:
            print(f"     ? {u['kind']:14s} {u['name'][:40]}")
        if len(r["undeterminable"]) > 12:
            print(f"     ... {len(r['undeterminable']) - 12} more (use --json)")
    for n in r.get("notes", []):
        print(f"   note: {n[:200]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path,
                    default=_HERE.parent / "capability_drift_ratchet.txt")
    ap.add_argument("--fail-on-under-declared", action="store_true")
    ap.add_argument("--fail-on-over-declared", action="store_true")
    ap.add_argument("--require-pin", action="store_true",
                    help="refuse to report a surface whose hardware pin does not verify")
    a = ap.parse_args(argv)

    # DEFAULT TARGET SET IS DISCOVERED, not named: every target that has an RTL fact bundle. A hardcoded
    # default here would make this gate silently about one target forever, which is the overfitting the
    # whole module exists to prevent (and the no-target-name gate rightly rejects it).
    targets = a.target or CD.targets_with_facts()
    if not targets:
        print("no --target given and no target in this checkout has a readable RTL fact bundle; pass "
              "--target NAME, or regenerate facts first", file=sys.stderr)
        return 2

    ratchet = _load_ratchet(a.ratchet)
    reports = [audit(t, require_pin=a.require_pin) for t in targets]
    if a.json:
        print(json.dumps(reports, indent=2, default=str))
    else:
        for r in reports:
            _print_report(r, ratchet)

    new_under = [_debt(r["target"], "under", u["kind"], u["name"])
                 for r in reports for u in r.get("under_declared", [])
                 if _debt(r["target"], "under", u["kind"], u["name"]) not in ratchet]
    new_over = [_debt(r["target"], "over", o["kind"], o["name"])
                for r in reports for o in r.get("over_declared", [])
                if _debt(r["target"], "over", o["kind"], o["name"]) not in ratchet]

    if not a.json:
        n_undet = sum(len(r.get("undeterminable", [])) for r in reports)
        n_enb = sum(len(r.get("encodable_not_built", [])) for r in reports)
        print(f"\n{len(targets)} target(s): {len(new_under)} un-ratcheted under-declaration(s), "
              f"{len(new_over)} un-ratcheted over-declaration(s), {n_enb} encodable-but-not-built, "
              f"{n_undet} undeterminable entr(ies)")

    rc = 0
    if new_under and a.fail_on_under_declared:
        print(f"\nFAIL: {len(new_under)} under-declared capability(ies) not in the ratchet:\n  "
              + "\n  ".join(new_under), file=sys.stderr)
        rc = 1
    if new_over and a.fail_on_over_declared:
        print(f"\nFAIL: {len(new_over)} over-declared capability(ies) not in the ratchet:\n  "
              + "\n  ".join(new_over), file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
