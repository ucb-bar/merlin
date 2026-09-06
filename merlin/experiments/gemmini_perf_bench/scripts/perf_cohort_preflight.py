#!/usr/bin/env python3
"""Refuse a campaign whose family cohorts would produce no verdict.

A family's claim is decided by ONE analyzer, and that analyzer publishes a single ``preflight_*``
precondition check.  A cohort that fails it does not crash the campaign: the run proceeds, looks
healthy, measures every member, and then yields no verdict for that family.  Measured 2026-09-06 on
the gemmini performance corpus, a scope assembled by hand would have produced family verdicts for 14
of 38 members while all six families read as "fine" by inspection.

The failure mode this guards is one member short of a predeclared cohort.  A capture that fails drops
its member from the measured selection silently, and an analyzer that predeclares sixteen members
refuses at fifteen -- so the loss of one capture costs the whole family.

DISPATCH IS ON THE DECLARED ANALYZER, NEVER ON THE FAMILY NAME.  An earlier operator-local version of
this check carried a hardcoded table (``if fam in ("PC","PL","PQ") ... elif fam == "PM" ...``), which
is the exact shape :mod:`perf_claim_dispatch` was written to end: "adding a family meant adding a
caller -- and nobody added one", and two families consequently shipped declaring a claim that no code
path ever evaluated.  Here the cohort is grouped by the analyzer identity each capsule's own frozen
contract names, and an identity the registry cannot resolve is REFUSED and named rather than skipped.

Exit 0 = every analyzer cohort in scope preflights READY.
"""
from __future__ import annotations

import argparse
import collections
import inspect
import sys
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import perf_claim_dispatch as DISPATCH  # noqa: E402


def _descriptors(names: list[str], capsule_root: Path) -> list[dict[str, Any]]:
    """Each named capsule's frozen descriptor, or a refusal naming the ones that do not resolve."""
    loaded, missing = [], []
    for name in names:
        manifest = capsule_root / name / "capsule.yaml"
        if not manifest.is_file():
            missing.append(name)
            continue
        loaded.append(yaml.safe_load(manifest.read_text(encoding="utf-8")))
    if missing:
        raise SystemExit(
            f"cohort preflight: {len(missing)} capsule(s) have no descriptor under {capsule_root}: "
            + " ".join(sorted(missing)))
    return loaded


def _call_preflight(resolved: Any, descriptors: list[dict[str, Any]],
                    replicates: list[str]) -> dict[str, Any]:
    """Invoke the analyzer's own precondition check.

    Analyzers differ in whether their claim is defined over replicates, so the parameter is passed
    only when the resolved callable accepts it. Inspecting the signature keeps this file free of the
    per-family branching the module docstring refuses.
    """
    try:
        takes_replicates = "replicates" in inspect.signature(resolved.preflight).parameters
    except (TypeError, ValueError):                       # builtin or unintrospectable callable
        takes_replicates = False
    if takes_replicates:
        return resolved.preflight(descriptors, replicates=replicates)
    return resolved.preflight(descriptors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--members", required=True,
                        help="file holding the comma-separated capsule names in campaign scope")
    parser.add_argument("--capsule-root", required=True, type=Path,
                        help="directory holding <capsule>/capsule.yaml for every member")
    parser.add_argument("--replicate", action="append", default=None, metavar="ID",
                        help="replicate id (repeatable); analyzers defined over replicates get these")
    args = parser.parse_args(argv)

    replicates = args.replicate or ["r000", "r001"]
    names = [n.strip() for n in Path(args.members).read_text(encoding="utf-8").split(",")
             if n.strip()]
    if not names:
        print("cohort preflight: the member selection is empty", file=sys.stderr)
        return 2
    descriptors = _descriptors(names, args.capsule_root)

    # GROUP BY THE DECLARED ANALYZER. `resolve` refuses a mixed cohort by design -- one campaign
    # seals one claim -- so each group is resolved on its own and reported separately.
    from merlin.perf import claim_reach

    # KEYED BY (declared analyzer, declared family). Dispatch is on the analyzer, as the registry
    # requires -- but an analyzer's own precondition check is defined over ONE family, and several
    # families may declare the same analyzer. Measured 2026-09-06: PC, PL and PQ all declare
    # `perf_paired_claim`, and handing that analyzer its own 12-member cohort is refused with
    # "descriptors span 3 families". Grouping on the analyzer alone therefore reports a cohort the
    # analyzer cannot decide, while grouping on the family alone reintroduces the name-based dispatch
    # this module exists to avoid. Both keys are read from the capsule's own frozen contract.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    undeclared: list[str] = []
    for descriptor in descriptors:
        name = str(descriptor.get("name") or "<unnamed>")
        performance = descriptor.get("performance")
        try:
            identity = claim_reach.analyzer_identity(
                performance if isinstance(performance, dict) else {})
        except ValueError as exc:
            undeclared.append(f"{name}: unusable analyzer declaration ({exc})")
            continue
        if identity is None:
            undeclared.append(f"{name}: declares no acceptance.analyzer")
            continue
        family = str((performance if isinstance(performance, dict) else {}).get("family") or "")
        if not family:
            undeclared.append(f"{name}: declares no performance.family")
            continue
        grouped[(identity.declared, family)].append(descriptor)

    refusals = list(undeclared)
    decidable = 0
    for (declared, family), members in sorted(grouped.items()):
        try:
            resolved = DISPATCH.resolve(members)
            outcome = _call_preflight(resolved, members, replicates)
        except Exception as exc:                          # noqa: BLE001 - report, never skip
            refusals.append(f"{family} [{declared}] ({len(members)} member(s)): "
                            f"{type(exc).__name__}: {str(exc)[:200]}")
            continue
        status = outcome.get("status")
        print(f"  {family:4s} n={len(members):2d} [{declared:42s}] -> {status}")
        if status == "READY":
            decidable += len(members)
        else:
            reason = (outcome.get("refusal_reasons") or [""])[0]
            refusals.append(f"{family} [{declared}] ({len(members)} member(s)): "
                            f"{str(reason)[:200]}")

    print(f"\nmembers with a decidable claim: {decidable}/{len(descriptors)}")
    if refusals:
        print("\nCOHORT PREFLIGHT FAIL -- these cohorts would produce NO verdict:", file=sys.stderr)
        for line in refusals:
            print(f"  - {line}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
