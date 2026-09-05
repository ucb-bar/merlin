#!/usr/bin/env python3
"""Report which phase each capsule can serve, and refuse a corpus that cannot say why.

THE RULE THIS GATE ENFORCES. A capsule that serves only one phase is not a defect -- a movement family
contracts nothing and belongs to phase 1, and a member too large to certify belongs to phase 2 resting
on a certified sibling. What IS a defect is a single-phase member whose reason nobody recorded, because
that is indistinguishable from a member somebody forgot to size. Every single-phase verdict here
therefore carries a derived reason, and a verdict with none fails the gate.

WHY IT CANNOT GATE ON THE RATIO. The healthy state is a large ``both`` set, and it would be easy to
write ``fail if both < x``. That gate would be satisfiable by DELETING the members that serve one phase,
which improves the ratio and destroys coverage -- the same trap ``check_semantic_coverage`` already
documents for ARR, where gating on the score makes the rational response to a hard family to remove it
from the contract. So this gate checks that the split is EXPLAINED, never that it is favourable.

UNMEASURED IS NOT CLEAN. A target with no certification history cannot have its phase-1 membership
decided at all. That is reported as ``undetermined`` and, with ``--strict``, is a non-zero exit --
never a pass. The remedy is to certify that target's corpus once, not to weaken this check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common.paths import merlin_dir  # noqa: E402
from merlin.targetgen import cert_cost as CC  # noqa: E402
from merlin.targetgen import phase_policy as PP  # noqa: E402

#: The budget a capsule is sized against. A declared choice, not a measurement -- it is the number the
#: corpus is willing to spend per member, and it is stated here so a reader can see it rather than
#: finding it inside a fit.
DEFAULT_BUDGET_S = 300.0


def _corpus_root() -> Path:
    return merlin_dir() / "contract" / "capsules"


def _targets(root: Path) -> list[str]:
    """Every target with a public profile. Derived from what is on disk so a new target needs no edit
    here -- the profile IS the declaration that a target exists."""
    return sorted(p.stem for p in (root / "profiles").glob("*.yaml")
                  if not p.stem.startswith("_") and "." not in p.stem)


def _capsules_for(root: Path, target: str, subtrees: set[str]) -> list[Path]:
    """A target owns a subtree, except the one whose corpus sits at the corpus root. Which one that is
    is found by elimination rather than written down, so this file names no target."""
    if target in subtrees:
        return sorted((root / target).rglob("capsule.yaml"))
    excluded = subtrees | {"profiles"}
    return sorted(p for p in root.rglob("capsule.yaml")
                  if p.relative_to(root).parts[0] not in excluded)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", help="only this target (default: every target with a public profile)")
    ap.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("--strict", action="store_true",
                    help="an undecidable target (no certification history) is a failure, not a note")
    args = ap.parse_args()

    import yaml

    root = _corpus_root()
    targets = _targets(root)
    subtrees = {t for t in targets if (root / t).is_dir()}
    if args.target:
        if args.target not in targets:
            print(f"[FAIL] phase-split: {args.target!r} has no public profile; known: {targets}")
            return 2
        targets = [args.target]

    report: dict[str, dict] = {}
    unexplained: list[str] = []
    undecided: list[str] = []
    orphaned: list[str] = []

    for t in targets:
        caps = []
        for p in _capsules_for(root, t, subtrees):
            try:
                doc = yaml.safe_load(p.read_text())
            except Exception:  # noqa: BLE001 - an unreadable capsule is reported, never skipped silently
                unexplained.append(f"{t}: {p} could not be read")
                continue
            if isinstance(doc, dict):
                caps.append(doc)
        fit = CC.fit_for(t)
        rep = PP.split_report(caps, target=t, fit=fit, budget_s=args.budget_s)
        counts = rep["counts"]
        report[t] = {"n_capsules": rep["n_capsules"], "counts": counts,
                     "cert_fit_samples": getattr(fit, "n_samples", None),
                     "single_phase_reasons": rep["single_phase_reasons"]}

        anc = PP.anchors(caps, target=t, fit=fit, budget_s=args.budget_s)
        report[t]["obligations"] = anc["n_obligations"]
        report[t]["paired"] = anc["n_paired"]
        report[t]["orphaned"] = anc["n_orphaned"]
        if anc["n_orphaned"]:
            orphaned.append(f"{t}: {anc['n_orphaned']} phase-2 member(s) rest on nothing "
                            f"({anc['orphaned'][0]['why']})")

        for v in rep["verdicts"]:
            if v.phase in (PP.PHASE1, PP.PHASE2, PP.NEITHER) and not v.reason.strip():
                unexplained.append(f"{t}: {v.name} is {v.phase} with no recorded reason")
        if counts[PP.UNDETERMINED]:
            undecided.append(f"{t}: {counts[PP.UNDETERMINED]} of {rep['n_capsules']} undecidable "
                             f"({'no measured certification history' if fit is None else 'a predicate could not answer'})")

    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        print(f"{'target':<16}{'caps':>5}{'both':>6}{'p1':>5}{'p2':>5}{'neither':>9}{'undet':>7}"
              f"{'oblig':>7}{'anchored':>10}{'orphan':>8}  cert-fit")
        for t, r in report.items():
            c = r["counts"]
            n = r["cert_fit_samples"]
            print(f"{t:<16}{r['n_capsules']:>5}{c[PP.BOTH]:>6}{c[PP.PHASE1]:>5}{c[PP.PHASE2]:>5}"
                  f"{c[PP.NEITHER]:>9}{c[PP.UNDETERMINED]:>7}{r['obligations']:>7}{r['paired']:>10}"
                  f"{r['orphaned']:>8}  {('n=%d' % n) if n else 'none'}")

    if unexplained:
        print("\n[FAIL] phase-split: a single-phase verdict with no recorded reason is indistinguishable "
              "from a member nobody sized:")
        for line in unexplained[:20]:
            print(f"  - {line}")
        return 1

    if orphaned:
        head = "[FAIL]" if args.strict else "[note]"
        print(f"\n{head} phase-split: a phase-2 member is admissible only as an EXTENSION of a sibling "
              "that WAS certified; one resting on nothing is an L2 pass on a shape nothing ever "
              "certified cycle-accurately:")
        for line in orphaned:
            print(f"  - {line}")

    if undecided:
        head = "[FAIL]" if args.strict else "[note]"
        print(f"\n{head} phase-split: a target with no certification history cannot have its phase-1 "
              "membership decided; certify its corpus once rather than weakening this check:")
        for line in undecided:
            print(f"  - {line}")
        if args.strict:
            return 1
    if orphaned and args.strict:
        return 1

    print("\n[  ok] phase-split: every single-phase verdict carries a derived reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
