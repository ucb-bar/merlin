#!/usr/bin/env python3
"""Gate: a capsule whose work the hardware ADMITS must assert that the work lands there.

`semantic.must_accelerate: true` is the demand that a capsule's work reach the accelerator, and
`false` permits a host fallback. Flipping it is the cheapest possible way to make a failing capsule
pass, and it is exactly what a regeneration does by accident: the generator emits its own default,
the hand-authored assertion is not in its vocabulary, and the field decays. This repo has already
paid for that once -- regeneration silently deleted `pass_requirements` from the only two capsules
that had it -- and it recurred. Measured 2026-09-02: six HAND-AUTHORED microscaling capsules
(M0/M1/M2 mxfp8, M3/M4 mxfp4, M5 mxfp6) went from

    must_accelerate: true,  fallback_allowed: false        (tracked)
to  must_accelerate: false, eligible: auto                 (working tree)

on a target whose own capability contract admits `contraction` at mxfp4, mxfp6 AND mxfp8. So the work
is eligible, the demand was right, and the weaker value was not a decision about the hardware.

WHY THIS IS AN INVARIANT AND NOT A DIFF. Comparing against git would only catch a weakening that
happened recently, and would call every legitimate authoring change a violation. The property that
actually holds is derivable from the target itself: if the capability contract admits a capsule's
`(semantic_family, operand dtype)`, then the accelerator can take that work, and a capsule declining
to demand it owes a REASON. `not_asserted_reason` is that field and 24 capsules already carry it. A
capsule whose cell is NOT admitted may say `must_accelerate: false` freely -- there the fallback is
the correct answer and forcing the demand would be the bug.

Modes, mirroring the sibling gates here:

  --target NAME        restrict to one target (repeatable)
  --json               machine-readable
  --ratchet PATH       pre-existing debt that MAY ONLY SHRINK
  --fail-on-weakened   exit non-zero when an admitted capsule declines the demand with no reason
  --fail-on-unresolved
                       exit non-zero when a target's contract cannot be read, since then nothing about
                       its capsules has been established either way
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (_REPO / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402

from merlin.common.paths import merlin_dir  # noqa: E402

#: Capsules written by hand are the reference the automation is being built to reproduce, so a
#: weakening there is worth naming separately from one in a generated capsule.
_HAND_ROLES = frozenset({"handauthored_compiler_test", "uplifted_from_bareMetalC"})


def _operand_dtype(doc: dict) -> str | None:
    """The dtype the capsule's work is done in, from its declared operands."""
    for role in ("input", "weight"):
        for spec in (doc.get("inputs") or ()):
            if str((spec or {}).get("role")) == role and spec.get("dtype"):
                return str(spec["dtype"])
    return None


def _target_of(path: Path) -> str | None:
    """The target a capsule belongs to, from its position under the corpus root.

    A capsule directly under a category (`isa/`, `layers/`, ...) belongs to the default target, which
    is not named here: the caller resolves it from the descriptors, because a target literal in this
    file is exactly the overfit the repo's cardinal rule forbids.
    """
    parts = path.parts
    try:
        i = parts.index("capsules")
    except ValueError:
        return None
    rest = parts[i + 1:]
    return rest[0] if len(rest) > 2 else None


def audit(targets=()) -> dict:
    from merlin.targetgen import conformance as CF

    root = merlin_dir() / "contract" / "capsules"
    default_targets = []
    tdir = _REPO / "merlin" / "experiments" / "capsule_bench" / "targets"
    if tdir.is_dir():
        for desc in sorted(tdir.glob("*/target_experiment.yaml")):
            try:
                doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            default_targets.append((desc.parent.name, str(doc.get("target") or desc.parent.name)))

    admitted_cache: dict[str, tuple[dict, str]] = {}

    def _admitted(t: str):
        if t not in admitted_cache:
            try:
                admitted_cache[t] = CF.admitted_with_reason(t)
            except Exception as exc:               # noqa: BLE001
                admitted_cache[t] = ({}, f"unresolvable: {type(exc).__name__}: {exc}")
        return admitted_cache[t]

    # The corpus root with no target directory is the default target's; resolve it as the one whose
    # descriptor directory name is not itself a directory under the corpus.
    dir_names = {p.name for p in root.iterdir() if p.is_dir()}
    default = next((declared for d, declared in default_targets if d not in dir_names), None)

    weakened, unresolved, checked = [], {}, 0
    for cy in sorted(root.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        sem = doc.get("semantic") or {}
        fam = sem.get("semantic_family")
        if not fam:
            continue
        dir_target = _target_of(cy)
        declared = next((d for n, d in default_targets if n == dir_target), dir_target) or default
        if not declared:
            continue
        if targets and declared not in targets:
            continue
        adm, why = _admitted(declared)
        if why != "resolved":
            unresolved[declared] = why
            continue
        dtype = _operand_dtype(doc)
        dtypes = {str(x) for x in (adm.get(str(fam)) or ())}
        if not dtypes or (dtype and dtype not in dtypes):
            continue                               # not admitted: a fallback is the right answer
        checked += 1
        if sem.get("must_accelerate") is True:
            continue
        if doc.get("not_asserted_reason") or sem.get("not_asserted_reason"):
            continue                               # declined WITH a stated reason
        weakened.append({
            "capsule": cy.parent.name, "target": declared, "family": str(fam), "dtype": dtype,
            "must_accelerate": sem.get("must_accelerate"),
            "source_role": str(doc.get("source_role") or "(none)"),
            "hand_authored": str(doc.get("source_role") or "") in _HAND_ROLES,
            "path": str(cy.parent.relative_to(_REPO)),
        })
    return {"n_admitted_capsules_checked": checked, "unresolved_targets": unresolved,
            "weakened": sorted(weakened, key=lambda r: (not r["hand_authored"], r["capsule"]))}


def _load_ratchet(p: Path | None) -> set[str]:
    if not p or not p.is_file():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=None)
    ap.add_argument("--fail-on-weakened", action="store_true")
    ap.add_argument("--fail-on-unresolved", action="store_true")
    a = ap.parse_args(argv)

    rep = audit(tuple(a.target or ()))
    ratchet = _load_ratchet(a.ratchet)
    new = [r for r in rep["weakened"] if r["capsule"] not in ratchet]
    hand = [r for r in new if r["hand_authored"]]

    if a.json:
        print(json.dumps({"report": rep, "new": new}, indent=2))
    else:
        print(f"== mesh assertion vs declared capability")
        print(f"   capsules whose cell the hardware admits : {rep['n_admitted_capsules_checked']}")
        print(f"   declining the demand with no reason     : {len(rep['weakened'])}"
              f"  ({len(hand)} of them HAND-AUTHORED)")
        for r in rep["weakened"][:25]:
            mark = " " if r["capsule"] in ratchet else "*"
            tag = "HAND" if r["hand_authored"] else "    "
            print(f"   {mark} {tag} {r['capsule']:36s} {r['target']:12s} "
                  f"{r['family']}/{r['dtype']} must_accelerate={r['must_accelerate']}")
        if rep["unresolved_targets"]:
            print(f"   UNRESOLVED ({len(rep['unresolved_targets'])}) — nothing established either way:")
            for t, why in rep["unresolved_targets"].items():
                print(f"     ? {t}: {why[:90]}")

    rc = 0
    if a.fail_on_weakened and new:
        print(f"\nFAIL: {len(new)} capsule(s) decline must_accelerate on work their target's "
              f"contract admits, with no not_asserted_reason ({len(hand)} hand-authored)",
              file=sys.stderr)
        rc = 1
    if a.fail_on_unresolved and rep["unresolved_targets"]:
        print(f"\nCANNOT DECIDE: {len(rep['unresolved_targets'])} target(s) have no resolvable "
              f"contract, so their capsules established nothing", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
