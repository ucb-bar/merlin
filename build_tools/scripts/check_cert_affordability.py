#!/usr/bin/env python3
"""Gate: can anyone afford to certify the capsules that demand certification?

A capsule declaring ``L3`` in ``required_oracle_tiers`` is asking for a cycle-accurate run, and that
run has a price. Nothing checked the price against a budget, so the corpus could -- and did -- grow
capsules whose certification nobody would ever pay for, which is the same as not certifying them
while reporting that they must be.

MEASURED, not assumed. A calibration ladder that held the lhs at one tile while the weight grew 64x
found cycle-accurate seconds scaling with the COMMITTED OUTPUT and not with the operands: x1.98 then
x2.06 against output x2 and x2, r2 0.9998, at a near-constant 0.347 s per committed element with no
fixed floor. Confirmed outside the ladder on a two-commit resident-reuse capsule at 0.3409 s/element.
Measured over the whole corpus: 295 capsules demand L3 for a predicted 95.3 hours, of which TEN
capsules are 83.4 hours -- 87% of the bill -- and the largest four are 75% of it. The median capsule
costs 89 seconds. So this is not a "too many capsules" problem; it is a handful of very large ones.

WHAT AN OVER-BUDGET CAPSULE OWES. Not deletion -- a large shape is often the representative one, and
shrinking every capsule to fit a budget would buy affordability with generality. The corpus already
has the right shape for this: an affordable capsule certified at L3, plus a larger L2-only capsule
that ``extends`` it and rests on its guarantee. So an over-budget capsule must either declare
``max_oracle_tier: L2`` (it is not asking for certification after all) or ``extends`` (it rests on a
sibling that is certified). Declaring neither, while demanding L3 at a size nobody can run, is the
gap this gate reports.

Modes, mirroring the sibling gates in this directory:

  --target NAME        restrict to one target's capsule roots (repeatable)
  --budget-s SECONDS   the affordability bar (default: the conformance default)
  --json               machine-readable
  --ratchet PATH       pre-existing debt that MAY ONLY SHRINK
  --fail-on-unaffordable
                       exit non-zero when a non-ratcheted capsule demands L3 over budget and declares
                       neither an L2 cap nor an extends
  --fail-on-unpriceable
                       exit non-zero when a capsule's cost CANNOT be computed. A capsule whose price
                       is unknown has not been shown to be affordable, and this repo has repeatedly
                       paid for a check that could not run reporting success.
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
from merlin.targetgen import cert_cost as CC  # noqa: E402

#: The same default the conformance derivation sizes against, read from it rather than restated.
try:
    from merlin.targetgen.conformance import _DEFAULT_CERT_BUDGET_S as _BUDGET
except ImportError:                                # pragma: no cover - keep the gate runnable
    _BUDGET = 300.0


def _price(fit, output_elements: int) -> tuple[float, str]:
    """Seconds for a capsule committing ``output_elements``, and what that rests on."""
    if fit is not None and getattr(fit, "metric", "") == "output_elements":
        secs = CC.predict_seconds(fit, output_elements)
        if secs is not None:
            return secs, f"fitted ({fit.n_samples} samples, r2 {fit.r2:.2f})"
    return (output_elements * CC.MEASURED_S_PER_OUTPUT_ELEMENT,
            f"measured default {CC.MEASURED_S_PER_OUTPUT_ELEMENT} s/element")


def audit(*, budget_s: float = _BUDGET, targets=()) -> dict:
    root = merlin_dir() / "contract" / "capsules"
    rows, unpriceable = [], []
    for cy in sorted(root.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            unpriceable.append({"capsule": cy.parent.name, "why": f"unreadable capsule.yaml: {exc}"})
            continue
        if "L3" not in (doc.get("required_oracle_tiers") or ()):
            continue                               # not asking to be certified
        if targets and not any(t in cy.parts for t in targets):
            continue
        ifc = cy.parent / str(doc.get("interface_mlir") or "capsule.interface.mlir")
        if not ifc.is_file():
            unpriceable.append({"capsule": cy.parent.name, "why": "no interface to price"})
            continue
        try:
            out = CC.capsule_output_elements(ifc.read_text(encoding="utf-8"))
        except Exception as exc:                   # noqa: BLE001 -- an unpriceable capsule is reported
            unpriceable.append({"capsule": cy.parent.name,
                                "why": f"{type(exc).__name__}: {exc}"})
            continue
        if out <= 0:
            unpriceable.append({"capsule": cy.parent.name, "why": "commits nothing measurable"})
            continue
        secs, basis = _price(None, out)
        rows.append({"capsule": cy.parent.name, "output_elements": out,
                     "predicted_s": round(secs, 1), "basis": basis,
                     "max_oracle_tier": doc.get("max_oracle_tier"),
                     "extends": doc.get("extends"),
                     "path": str(cy.parent.relative_to(_REPO))})
    over = [r for r in rows
            if r["predicted_s"] > budget_s and not r["extends"]
            and str(r["max_oracle_tier"] or "").upper() != "L2"]
    total = sum(r["predicted_s"] for r in rows)
    return {"budget_s": budget_s, "n_demanding_l3": len(rows),
            "total_predicted_s": round(total, 1),
            "total_predicted_hours": round(total / 3600.0, 2),
            "over_budget": sorted(over, key=lambda r: -r["predicted_s"]),
            "unpriceable": unpriceable,
            "s_per_output_element": CC.MEASURED_S_PER_OUTPUT_ELEMENT}


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
    ap.add_argument("--budget-s", type=float, default=_BUDGET)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=None)
    ap.add_argument("--fail-on-unaffordable", action="store_true")
    ap.add_argument("--fail-on-unpriceable", action="store_true")
    a = ap.parse_args(argv)

    rep = audit(budget_s=a.budget_s, targets=tuple(a.target or ()))
    ratchet = _load_ratchet(a.ratchet)
    new = [r for r in rep["over_budget"] if r["capsule"] not in ratchet]

    if a.json:
        print(json.dumps({"report": rep, "new_over_budget": new}, indent=2))
    else:
        print(f"== certification affordability at {rep['budget_s']:.0f}s "
              f"({rep['s_per_output_element']} s per committed element, measured)")
        print(f"   capsules demanding L3        : {rep['n_demanding_l3']}")
        print(f"   predicted total              : {rep['total_predicted_s']:,.0f}s "
              f"({rep['total_predicted_hours']}h)")
        print(f"   over budget with no L2 cap and no extends: {len(rep['over_budget'])}")
        for r in rep["over_budget"][:20]:
            mark = " " if r["capsule"] in ratchet else "*"
            print(f"   {mark} {r['predicted_s']:9,.0f}s  {r['output_elements']:9,} out  "
                  f"{r['capsule']}")
        if len(rep["over_budget"]) > 20:
            print(f"     ... and {len(rep['over_budget']) - 20} more")
        if rep["unpriceable"]:
            # NOT counted as affordable. An unknown price is not a small one.
            print(f"   UNPRICEABLE ({len(rep['unpriceable'])}) — these establish nothing either way:")
            for u in rep["unpriceable"][:10]:
                print(f"     ? {u['capsule']}: {u['why']}")

    rc = 0
    if a.fail_on_unaffordable and new:
        print(f"\nFAIL: {len(new)} capsule(s) demand L3 at a size over the {rep['budget_s']:.0f}s "
              f"budget and declare neither max_oracle_tier: L2 nor extends", file=sys.stderr)
        rc = 1
    if a.fail_on_unpriceable and rep["unpriceable"]:
        print(f"\nCANNOT DECIDE: {len(rep['unpriceable'])} capsule(s) could not be priced, so their "
              f"affordability is unknown rather than acceptable", file=sys.stderr)
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
