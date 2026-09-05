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
from merlin.targetgen import cert_affordability as CA  # noqa: E402
from merlin.targetgen import cert_cost as CC  # noqa: E402

#: The same default the conformance derivation sizes against, read from it rather than restated.
try:
    from merlin.targetgen.conformance import _DEFAULT_CERT_BUDGET_S as _BUDGET
except ImportError:                                # pragma: no cover - keep the gate runnable
    _BUDGET = 300.0


#: The size metric ``cert_cost`` actually fits against, read off its own dataclass rather than spelled
#: here. Restating it as a literal is how the fitted branch below went dead: it compared against
#: ``"output_elements"`` while every fit carries ``"written_output_elements"``, so no fit was ever used
#: and every price silently came from the global law.
_FITTED_METRIC = CC.CostFit.__dataclass_fields__["metric"].default


def _price(fit, output_elements: int) -> tuple[float, str]:
    """Seconds for a capsule committing ``output_elements``, and what that rests on.

    A fit is used only when it is a fit over the SAME metric this prices in. Everything else falls
    through to the global calibration law, and the basis SAYS SO -- a price that came from a law
    calibrated once, on one engine, on one target, must never read as a measurement of the target it is
    being quoted for.
    """
    if fit is not None and getattr(fit, "metric", "") == _FITTED_METRIC:
        secs = CC.predict_seconds(fit, output_elements)
        if secs is not None:
            engine = getattr(fit, "engine", None)
            where = f"{fit.target}/{engine}" if engine else str(fit.target)
            return secs, f"fitted on {where} ({fit.n_samples} samples, r2 {fit.r2:.2f})"
    secs, extrapolated = CC.predict_seconds_from_output(output_elements)
    basis = (f"no measured (target, engine) basis; global calibration law "
             f"{CC.MEASURED_COEFFICIENT_S} * out^{CC.MEASURED_EXPONENT}"
             + (" EXTRAPOLATED past the calibrated range" if extrapolated else ""))
    return (secs or 0.0, basis)


def _default_corpus_target() -> str:
    """The target whose corpus sits at the corpus ROOT rather than in a subtree of its own.

    Found by elimination -- the one public profile with no directory beside it -- so this file names no
    target. Needed because that target's capsules have one fewer path component than everyone else's,
    which is exactly what the label rule below has to get right.
    """
    from merlin.common.paths import merlin_dir

    root = merlin_dir() / "contract" / "capsules"
    names = [f.stem for f in (root / "profiles").glob("*.yaml")
             if not f.stem.startswith("_") and "." not in f.stem]
    rootless = [n for n in names if not (root / n).is_dir()]
    return rootless[0] if len(rootless) == 1 else "(default)"


def _target_label_for(cy) -> str:
    """Which target a capsule belongs to, from its path.

    ⚠️ THE ROOT-CORPUS TARGET HAS ONE FEWER COMPONENT. A subtree capsule is
    ``capsules/<target>/<category>/<name>/capsule.yaml`` (4 parts after ``capsules``); the root
    corpus's is ``capsules/<category>/<name>/capsule.yaml`` (3). The previous rule took the first
    component whenever there were more than two, so every capsule of the root-corpus target was
    labelled with its CATEGORY -- ``isa``, ``_perf``, ``model`` -- and priced against a target of that
    name, which has no certification history at all. That target's 34 measured certifications have
    therefore never priced its own capsules; every one silently fell back to the global law.
    """
    parts = cy.parts
    i = parts.index("capsules") if "capsules" in parts else -1
    rest = parts[i + 1:] if i >= 0 else ()
    return rest[0] if len(rest) > 3 else _default_corpus_target()


def _resolved_target(label: str) -> str:
    """The TARGET NAME behind a corpus directory label.

    A descriptor sits in a short directory and declares a configuration-qualified name, and every
    artifact path -- including the measurement roots this prices from -- uses the declared one. Looking
    up the directory name would find no history for a target that has plenty.
    """
    try:
        from merlin.targetgen.target_registry import declared_target_for
        return declared_target_for(label) or label
    except Exception:                              # noqa: BLE001 -- no registry here: the label as given
        return label


def _engine_fit(target: str, cache: dict):
    """The ONE measured ``(target, engine)`` fit to price this target's capsules with, or ``None``.

    ``None`` for a target with no measured certification history at all, and ``None`` -- deliberately --
    for a target measured on SEVERAL engines: the two answers differ by more than an order of magnitude
    (3.31 s vs 86.83 s for the same capsule), so "the cost on this target" is not a question with one
    answer and picking either engine would be inventing the choice. Such a target is priced per engine in
    ``measured_basis`` instead, where the reader sees both.
    """
    if target not in cache:
        try:
            cache[target] = CA.fits_for(_resolved_target(target))
        except Exception:                          # noqa: BLE001 -- unreadable history is no history
            cache[target] = {"engines": {}, "sample_counts": {}, "unattributed_samples": 0,
                             "unsized_samples": 0}
    fits = [f for f in cache[target]["engines"].values() if f is not None]
    return fits[0] if len(fits) == 1 else None


def audit(*, budget_s: float = _BUDGET, targets=()) -> dict:
    root = merlin_dir() / "contract" / "capsules"
    rows, unpriceable, extends_rows = [], [], []
    fit_cache: dict = {}
    for cy in sorted(root.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            unpriceable.append({"capsule": cy.parent.name, "why": f"unreadable capsule.yaml: {exc}"})
            continue
        # ⚠️ COLLECTED BEFORE THE L3 FILTER, DELIBERATELY. A capsule resting on a sibling is exactly the
        # one that does NOT demand L3 -- it was capped to the cheap tier precisely because it could not
        # afford certification -- so verifying `extends` after this filter examines only capsules that
        # never carry one. Placed after it, the check reported zero every time while 19 capsules on disk
        # declared the field.
        if doc.get("extends") and (not targets or any(t in cy.parts for t in targets)):
            extends_rows.append({"capsule": cy.parent.name,
                                 "target": _resolved_target(_target_label_for(cy)),
                                 "extends": str(doc["extends"]),
                                 "max_oracle_tier": doc.get("max_oracle_tier")})
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
        perf = doc.get("performance") or {}
        instrument = str(((perf.get("gate") or {}).get("instrument")) or "")
        _target = _target_label_for(cy)
        # PRICED WITH THIS TARGET'S OWN MEASURED ENGINE where one exists. The whole corpus used to be
        # priced by the single global law, so a target nobody had ever certified and one with 34
        # measured certifications produced the same number and neither said which.
        secs, basis = _price(_engine_fit(_target, fit_cache), out)
        rows.append({"capsule": cy.parent.name,
                     "target": _target,
                     "output_elements": out,
                     "extrapolated": out > CC.MEASURED_MAX_OUTPUT_ELEMENTS,
                     "predicted_s": round(secs, 1), "basis": basis,
                     "max_oracle_tier": doc.get("max_oracle_tier"),
                     "extends": doc.get("extends"),
                     "perf_family": perf.get("family"), "instrument": instrument or None,
                     # ⚠️ AN L2 CAP IS NOT AVAILABLE TO EVERY CAPSULE. A performance capsule whose
                     # gate instrument is a cycle count NEEDS a cycle-accurate tier -- capping it at
                     # L2 would not make it cheap, it would destroy the measurement the capsule
                     # exists to take. Advising that remedy would be advising an impossible fix, so
                     # these are reported apart with the remedy that IS available to them.
                     "needs_cycle_accurate": "cycle" in instrument,
                     "path": str(cy.parent.relative_to(_REPO))})
    # ⚠️ AN UNVERIFIED `extends` IS NOT A REMEDY. A non-empty field was read as one, so a capsule
    # naming a sibling that was never certified counted as remedied -- the failure the field exists to
    # prevent, arrived at through the field itself. verify_extends asks the sibling's own results
    # whether it earned a deeper tier and fails closed: no result on disk records UNVERIFIED, which is
    # weaker than naming nobody, because an unchecked extends reads as certified.
    for r in extends_rows:
        try:
            from merlin.targetgen import tier_policy as TP

            v = TP.verify_extends(r["target"],
                                  {"extends": r["extends"],
                                   "required_oracle_tiers": ["L0", "L1", "L2", "L3"]},
                                  str(r["max_oracle_tier"] or "L2"))
            r["verified"] = bool(getattr(v, "verified", False))
            r["reason"] = str(getattr(v, "reason", ""))
        except Exception as exc:  # noqa: BLE001 - unaskable is UNKNOWN, never a pass
            r["verified"], r["reason"] = False, f"could not be verified here: {type(exc).__name__}"
    unverified_extends = [r for r in extends_rows if not r.get("verified")]
    unremedied = [r for r in rows
                  if r["predicted_s"] > budget_s and not r["extends"]
                  and str(r["max_oracle_tier"] or "").upper() != "L2"]
    over = [r for r in unremedied if not r["needs_cycle_accurate"]]
    needs_cycles = [r for r in unremedied if r["needs_cycle_accurate"]]
    total = sum(r["predicted_s"] for r in rows)

    # WHAT EACH TARGET'S NUMBERS ACTUALLY REST ON, per engine, stated beside them. A cohort total is
    # only as good as the engine it was priced for, and a target whose whole history is unattributed has
    # no per-engine basis at all -- which is a finding, not a gap to be filled with the global law.
    by_target: dict = {}
    for r in rows:
        by_target.setdefault(r["target"], {})[r["capsule"]] = r["output_elements"]
    measured: dict = {}
    for target, sizes in sorted(by_target.items()):
        got = fit_cache.get(target)
        if got is None:
            continue
        measured[target] = {
            "n_l3_capsules": len(sizes),
            "unattributed_samples": got.get("unattributed_samples", 0),
            "unsized_samples": got.get("unsized_samples", 0),
            "engines": {engine: {"n_samples": got["sample_counts"].get(engine, 0),
                                 "fit": f.to_dict() if f is not None else None,
                                 "cohort": CA.cohort_price(f, sizes)}
                        for engine, f in sorted(got["engines"].items())},
        }
    # AND the same question asked of the REGISTERED ROSTER rather than of the corpus's directory
    # labels, because the two do not coincide: the default target's capsules sit directly under
    # `capsules/<category>/`, so they are labelled by category above and no per-target basis would ever
    # be reported for them. The roster comes from the registry, never from a list written here.
    roster: dict = {}
    try:
        from merlin.targetgen.target_registry import all_targets
        names = list(all_targets())
    except Exception:                              # noqa: BLE001 -- no registry: no roster, not a guess
        names = []
    # Plus every target capsule-bench has a descriptor for: a target can be benched (and certified)
    # without being in the compiler's own registry, and its measured history is exactly as real.
    _descs = _REPO / "merlin" / "experiments" / "capsule_bench" / "targets"
    if _descs.is_dir():
        names += [_resolved_target(d.name) for d in _descs.iterdir()
                  if (d / "target_experiment.yaml").is_file()]
    _seen = {_resolved_target(t): got for t, got in fit_cache.items()}
    for name in sorted(set(names) | set(_seen)):
        got = _seen.get(name)
        if got is None:
            try:
                got = CA.fits_for(name)
            except Exception:                      # noqa: BLE001 -- unreadable history is no history
                continue
        if not (got["engines"] or got["unattributed_samples"] or got["unsized_samples"]):
            continue
        roster[name] = {"engines": {e: (f.to_dict() if f is not None else None)
                                    for e, f in sorted(got["engines"].items())},
                        "sample_counts": got["sample_counts"],
                        "unattributed_samples": got["unattributed_samples"],
                        "unsized_samples": got["unsized_samples"]}
    return {"budget_s": budget_s, "n_demanding_l3": len(rows),
            "measured_basis": measured, "roster_basis": roster,
            "n_extrapolated": sum(1 for r in rows if r["extrapolated"]),
            "extrapolated_hours": round(sum(r["predicted_s"] for r in rows
                                            if r["extrapolated"]) / 3600.0, 2),
            "total_predicted_s": round(total, 1),
            "total_predicted_hours": round(total / 3600.0, 2),
            "n_extends_declared": len(extends_rows),
            "unverified_extends": [{"capsule": r["capsule"], "target": r["target"],
                                    "extends": r["extends"], "reason": r.get("reason", "")}
                                   for r in unverified_extends],
            "over_budget": sorted(over, key=lambda r: -r["predicted_s"]),
            # Over budget, but an L2 cap is not a remedy they can take.
            "over_budget_needs_cycle_accurate": sorted(needs_cycles,
                                                       key=lambda r: -r["predicted_s"]),
            "unpriceable": unpriceable,
            "s_per_output_element": CC.MEASURED_S_PER_OUTPUT_ELEMENT}


def _debt_key(row: dict) -> str:
    """``<target>/<capsule>`` -- the ratchet key, SCOPED TO THE TARGET.

    A bare capsule name is not unique: `SY_micro_model` exists under both atlas and
    saturn_opu_rvv, and each is a different capsule with its own cost. Keying the ratchet on the name
    alone meant accepting one target's debt silently excused the other's -- the same flaw
    check_pass_obligations scopes away by keying on both the pass and the axis.
    """
    return f"{row.get('target') or '?'}/{row['capsule']}"


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
    new = [r for r in rep["over_budget"] if _debt_key(r) not in ratchet]

    if a.json:
        print(json.dumps({"report": rep, "new_over_budget": new}, indent=2))
    else:
        print(f"== certification affordability at {rep['budget_s']:.0f}s "
              f"({rep['s_per_output_element']} s per element at the flat rungs; law is superlinear)")
        print(f"   capsules demanding L3        : {rep['n_demanding_l3']}")
        print(f"   predicted total              : {rep['total_predicted_s']:,.0f}s "
              f"({rep['total_predicted_hours']}h)")
        print(f"   priced beyond the calibrated range : {rep['n_extrapolated']} "
              f"({rep['extrapolated_hours']}h of the total — an unstated guess if not said out loud)")
        print(f"   over budget with no L2 cap and no extends: {len(rep['over_budget'])}")
        print(f"   resting on an UNVERIFIED extends          : "
              f"{len(rep.get('unverified_extends') or [])} "
              "(names a sibling with no certification on disk)")
        for r in rep["over_budget"][:20]:
            mark = " " if _debt_key(r) in ratchet else "*"
            print(f"   {mark} {r['predicted_s']:9,.0f}s  {r['output_elements']:9,} out  "
                  f"{_debt_key(r)}")
        if len(rep["over_budget"]) > 20:
            print(f"     ... and {len(rep['over_budget']) - 20} more")
        if rep["over_budget_needs_cycle_accurate"]:
            print(f"   over budget but CANNOT be capped at L2 "
                  f"({len(rep['over_budget_needs_cycle_accurate'])}): a performance capsule whose "
                  f"instrument is a cycle count needs a cycle-accurate tier, so its remedy is a "
                  f"smaller shape or an accepted cost, never an L2 cap")
            for r in rep["over_budget_needs_cycle_accurate"][:10]:
                print(f"     ! {r['predicted_s']:9,.0f}s  {r['output_elements']:9,} out  "
                      f"{r['capsule']}  [{r['perf_family']}: {r['instrument']}]")
        print("   measured (target, engine) basis for these prices:")
        for target, blk in sorted(rep["measured_basis"].items()):
            if not blk["engines"]:
                n = blk["unattributed_samples"]
                why = (f"{n} cycle-accurate measurement(s) on disk, none of which names its engine"
                       if n else "no cycle-accurate certification history under this name")
                print(f"     - {target}: NONE -- {why}, so no per-engine cost can be fitted and "
                      f"every price above is the global calibration law")
                continue
            for engine, row in sorted(blk["engines"].items()):
                fit = row["fit"]
                if fit is None:
                    print(f"     - {target}/{engine}: {row['n_samples']} sample(s), too few to fit")
                    continue
                tot = row["cohort"]["total_s"]
                print(f"     - {target}/{engine}: n={fit['n_samples']} r2={fit['r2']} over "
                      f"{fit['measured_range_elements']} {fit['metric']}; cohort "
                      f"{tot:,.0f}s over {row['cohort']['priced']} capsule(s), "
                      f"{len(row['cohort']['beyond_evidence'])} beyond the evidence")
        if rep["roster_basis"]:
            print("   registered targets with any cycle-accurate history (the roster, not the "
                  "corpus's directory labels):")
            for name, blk in sorted(rep["roster_basis"].items()):
                fitted = {e: f for e, f in blk["engines"].items() if f}
                print(f"     - {name}: {sum(blk['sample_counts'].values())} engine-attributed "
                      f"sample(s) across {len(blk['sample_counts'])} engine(s); "
                      f"{blk['unattributed_samples']} unattributed; "
                      f"fits: {sorted(fitted) or 'NONE'}")
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
