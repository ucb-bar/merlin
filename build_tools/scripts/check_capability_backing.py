#!/usr/bin/env python3
"""Gate: a capability claim that NARROWS a target must cite what it was read off.

THE HOLE THIS CLOSES. An op-coverage check asks "is there an op the manifest ADMITS that no capsule
DEMANDS?" Such a check could not have caught the defect it was written for, because it reads the
manifest: a manifest that wrongly says a capability does NOT exist produces silence, not a finding.
One hand-written contract line -- ``{family: elementwise_map, composed_with: [contraction]}``, with two
lines of prose behind it -- was inverted in both directions against the RTL, and the consequences were
invisible all the way through: no standalone elementwise capsule was ever derived, ``residual_add`` was
demanded by 0 of 103 graded capsules, the agent was graded 101/103 "complete", and the compiler that
produced refused 16 of ResNet-50's 71 device groups.

So this gate is about the DECLARATION, not the verdict. It cannot decide whether a restriction is true
-- nothing reading YAML can. It enforces that the restriction is CHECKABLE:

    a claim that narrows must carry a resolvable citation -- an RTL source in the read set of one of
    that target's OWN declared hardware pins, an in-repo test or module that pins it, or a derivation
    entry point -- and every citation it carries must still resolve.

Both halves matter. The second is why the first does not rot: a claim may cite exactly the right Chisel
file while the pin that names the checkout never reads it, so an edit to that file changes the verdict
and ``verify`` still calls the pin clean. That is not hypothetical -- it was the state of BOTH systolic
pins in this repo when this gate was written.

Only NARROWING is gated. Over-declaring produces capsules the hardware refuses, which is loud;
under-declaring removes rows from a denominator, which is silent and flatters every recall number
computed afterwards. See :mod:`merlin.targetgen.capability_backing` for the shape table, and its
``DEFERRED_SHAPES`` for the narrowing shapes this gate does NOT yet decide -- named so their absence is
not mistaken for coverage.

RATCHETED, NOT A FLAG DAY. Pre-existing unbacked claims live in ``capability_backing_ratchet.txt``,
which MAY ONLY SHRINK; a NEW unbacked claim fails immediately. The ledger is held in BOTH directions
like ``check_wiring.py``'s: an entry that is no longer a gap ALSO fails, so backing a claim
forces its line to be deleted rather than left to rot into a permanent allowlist.

Modes:

  --target NAME     audit one target (repeatable); default: every target with a contract document
  --tree            print the checkout this gate audits and stop
  --json            machine-readable document per target
  --list            print every claim with its citations and stop
  --shapes          print the gated and deferred shape tables and stop
  --ratchet PATH    pre-existing debt that may only shrink (default: the ledger beside this script)
  --no-ratchet      ignore the ledger entirely -- what the gate would say on a clean sheet
  --write-ratchet   regenerate the ledger from today's findings; review the diff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This gate audits ITS OWN checkout, so it seats that checkout's package ahead of anything an
# editable install or an inherited PYTHONPATH would resolve. That is right for the script and wrong
# to leave implicit: a worktree sharing a venv with another clone imports `merlin` from the OTHER
# clone by default, so a caller holding the library and this script at once can be looking at two
# different trees while both report green. `--tree` publishes which one this run read, and
# `merlin/tests/infra/test_capability_backing.py` asserts the two match rather than assuming it.
_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[2] / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import repo_root  # noqa: E402
from merlin.targetgen import capability_backing as CB  # noqa: E402

LEDGER = _HERE.parent / "capability_backing_ratchet.txt"

_HEADER = """\
# Capability claims that NARROW a target and carry no citation anyone can follow.
#
# NEVER ADD A LINE. This ledger may only shrink: cite the RTL source (and put it in the read set of a
# hardware pin this target declares), the test that pins it, or the derivation that produces it -- or
# drop the restriction -- and delete the entry. The gate fails on a NEW entry and equally on a STALE
# one, so an entry that is no longer a gap must be removed in the same change that closes it.
#
# Regenerate (and review the diff) with:
#   python build_tools/scripts/check_capability_backing.py --write-ratchet
#
# Format: <target> <axis>:<shape>:<detail>.
#   axis    `unbacked` (no citation resolves) or `rotted` (backed, but a citation no longer resolves)
#   shape   which kind of narrowing -- see merlin.targetgen.capability_backing.NARROWING_SHAPES
#   detail  the family, or the compute unit for `no_semantic_capabilities`
# Target-scoped because a claim is debt on one target and evidence on another; a flat entry would
# forgive every target at once.
#
# WHAT IS NOT HERE. A narrowing shape this gate does not yet decide (a unit's `ops` enumeration, a
# `legality` clause, a bare `false` feature flag, a family declared nowhere at all) is listed in
# `capability_backing.DEFERRED_SHAPES`, not here -- an empty ledger would otherwise read as "every
# narrowing in this repo is evidenced", which is not what it means.
#
# growth-accepted: the gate is new; these are the unbacked narrowing claims it found on the tree,
# not claims added. They include the shape the gate was written for -- an elementwise map declared
# reachable only behind a contraction, on prose alone -- which is still declared on one target here.
"""


def _load_ratchet(path: Path | None) -> set[str]:
    """Entries, one per line, ``#`` starting a comment. A missing ledger is an empty one."""
    if not path or not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            out.add(entry)
    return out


def _write_ratchet(path: Path, entries) -> None:
    body = "\n".join(sorted(entries))
    path.write_text(_HEADER + (body + "\n" if body else ""), encoding="utf-8")


def _print(doc: dict) -> None:
    print(f"\n=== {doc['target']} — capability-backing contract ===")
    pins = doc["pins"]
    print(f"  pins         {', '.join(sorted(pins)) or '(none declared)'}")
    print(f"  claims       {doc['n_claims']} narrowing claim(s) in {len(doc['sources'])} contract document(s)")
    for claim in doc["claims"]:
        mark = "ok  " if claim["backed"] and not claim["rotted"] else ("ROT " if claim["backed"] else "BARE")
        print(f"    [{mark}] {claim['shape']}:{claim['detail']}  {claim['source']}:{claim['line']}")
        print(f"           {claim['why']}")
    if doc["ratcheted"]:
        print(f"  ratcheted    {len(doc['ratcheted'])} known-unbacked claim(s)")
    for entry in doc["stale_ratchet_entries"]:
        print(f"  STALE        {entry} — no longer a gap; delete this ledger line")
    for problem in doc["problems"]:
        print(f"  FAIL         {problem}")
    if not doc["problems"] and not doc["stale_ratchet_entries"]:
        print("  OK")


def _decorate(doc: dict) -> dict:
    """Fields the printer needs that the library does not owe a JSON consumer."""
    rotted = {k.split(":", 1)[1] for k in doc["rotted"]}
    for claim in doc["claims"]:
        claim["rotted"] = f"{claim['shape']}:{claim['detail']}" in rotted
    doc["sources"] = sorted({c["source"] for c in doc["claims"]})
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=None, help="audit this target (repeatable)")
    ap.add_argument("--json", action="store_true", help="machine-readable document per target")
    ap.add_argument("--list", action="store_true", help="print every claim with its citations and stop")
    ap.add_argument("--shapes", action="store_true", help="print the gated and deferred shapes and stop")
    ap.add_argument("--tree", action="store_true", help="print the checkout this gate audits and stop")
    ap.add_argument("--ratchet", type=Path, default=LEDGER)
    ap.add_argument("--no-ratchet", action="store_true")
    ap.add_argument("--write-ratchet", action="store_true")
    args = ap.parse_args(argv)

    if args.tree:
        print(repo_root())
        return 0

    if args.shapes:
        print("GATED narrowing shapes:")
        for key, why in CB.NARROWING_SHAPES.items():
            print(f"  {key:26} {why}")
        print("\nDEFERRED (narrowing, not yet decided by this gate):")
        for key, why in CB.DEFERRED_SHAPES.items():
            print(f"  {key:26} {why}")
        return 0

    targets = args.target or list(CB.gated_targets())
    ratchet: set[str] = set() if args.no_ratchet else _load_ratchet(args.ratchet)

    docs = []
    for target in targets:
        try:
            doc = _decorate(CB.audit(target, ratchet=ratchet))
        except Exception as exc:  # noqa: BLE001 -- an unreadable contract is a finding, not a crash
            docs.append(
                {
                    "target": target,
                    "n_claims": 0,
                    "claims": [],
                    "sources": [],
                    "pins": {},
                    "unbacked": [],
                    "rotted": [],
                    "ratcheted": [],
                    "stale_ratchet_entries": [],
                    "problems": [f"contract could not be read structurally: {exc}"],
                }
            )
            continue
        docs.append(doc)

    if args.write_ratchet:
        entries = {e for doc in docs for e in doc["unbacked"] + doc["rotted"]}
        if args.target:
            keep = {e for e in _load_ratchet(args.ratchet) if e.split(" ", 1)[0] not in set(args.target)}
            entries |= keep
        _write_ratchet(args.ratchet, entries)
        print(f"wrote {len(entries)} entrie(s) to {args.ratchet}")
        return 0

    if args.json:
        print(json.dumps(docs, indent=2, sort_keys=True))
    elif args.list:
        for doc in docs:
            print(f"\n=== {doc['target']} ===")
            for claim in doc["claims"]:
                print(f"  {claim['shape']}:{claim['detail']}  {claim['source']}:{claim['line']}")
                print(f"    {claim['text']}")
                for cite in claim["citations"]:
                    print(f"      {cite['kind']:22} {cite['token']}  -> {cite['detail']}")
                if not claim["citations"]:
                    print("      (no citation)")
        return 0
    else:
        for doc in docs:
            _print(doc)

    failed = sum(len(doc["problems"]) + len(doc["stale_ratchet_entries"]) for doc in docs)
    if failed and not args.json:
        print(f"\n{failed} capability-backing problem(s). See the module docstring for what backing means.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
