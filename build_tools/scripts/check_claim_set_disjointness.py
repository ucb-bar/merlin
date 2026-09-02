#!/usr/bin/env python3
"""Gate: is the generalization claim measured over models that did not help build the corpus?

`conformance.required_cells(target, captures)` derives the requirement -- which
`(family, dtype, alignment)` cells a target owes -- from what real captures CONTAIN. The synthesized
corpus covers those cells. Coverage is then reported over captured models. If one capture does both
jobs the claim is circular: the corpus was built from the model it is said to generalize to.

Measured before the split was declared: `check_conformance_coverage` fed EVERY bundle under
`out/artifacts/recaptures/` into the derivation, the four claim models included, and lstmnetvit was
already in both roles.

FOUR CHECKS, because "disjoint" alone is not the property that matters:

1. **Disjointness** -- no bundle both derives and is claimed. Structural, from
   `merlin.targetgen.claim_models.partition`.
2. **Every claim model is captured.** A claim measured over a model nobody captured is a claim about
   nothing, and reads identically to a passing one.
3. **The derivation set is non-empty.** Holding out everything would trivially satisfy check 1 and
   derive no requirement at all.
4. **The requirement does not DEPEND on the claim set** -- the substantive property. Derive it twice,
   once from every capture and once from the derivation set alone, and compare. A cell present only
   with the claim models included is a requirement that a held-out model demanded, which is
   circularity in substance rather than in form. Measured on this corpus: the two derivations are
   IDENTICAL, so the split costs no requirement -- but that is a fact to keep checking, not to assume,
   because it stops being true the moment a claim model is the only capture carrying some family.

WHAT MAY BE PRINTED. A requirement CELL is public -- it is `admitted x observed` over family, dtype and
alignment, and the tracked conformance specs already state it. A claim model's SHAPES are not, and this
gate never reads them: it compares requirements, never capsule points. (The sibling
`check_holdout_disjointness.py` guards the other hazard -- there the specification of a holdout IS an
answer, so that gate reports counts only.)

Modes, mirroring the other gates in this directory:

  --target NAME        audit one target (repeatable); default: every target with a conformance spec
  --json               machine-readable
  --fail-on-circular   exit non-zero when a requirement cell depends on a held-out model
  --fail-on-unverifiable
                       exit non-zero when the comparison COULD NOT RUN (no captures, no declaration,
                       an unresolvable target). A check that could not run has established nothing,
                       and this repo has shipped one reporting success more than once.
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

from merlin.common.paths import artifacts_dir  # noqa: E402
from merlin.targetgen import claim_models as CM  # noqa: E402


def _bundles() -> dict[str, Path]:
    """Every captured bundle that carries a compilable module, keyed by its directory name.

    The directory name is kept VERBATIM -- the claim matcher works on token boundaries, and the
    prettified label other callers use (`_fp32_consistent` stripped) would erase the tokens it needs.
    """
    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return {}
    return {d.name: d / "model.mlir" for d in sorted(root.iterdir())
            if (d / "model.mlir").is_file()}


def _spec_targets() -> list[str]:
    """The targets tracked conformance specs are FOR, read from each spec's own ``target``.

    Not the filename. A spec's stem is a label and its `target:` is the key everything else resolves
    by -- measured: `conformance/saturn_opu.yaml` declares `target: saturn_opu_mxv256d128`, and
    auditing the stem asked about a target with no generated contract while the spec belonged to one
    that resolves. Falls back to the stem when a spec declares nothing, so an unreadable file still
    gets audited rather than silently skipped.
    """
    import yaml

    d = _REPO / "merlin" / "contract" / "capsules" / "conformance"
    if not d.is_dir():
        return []
    out = []
    for path in sorted(d.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            doc = {}
        out.append(str(doc.get("target") or path.stem))
    return sorted(set(out))


def audit(target: str, bundles: dict[str, Path] | None = None) -> dict:
    from merlin.targetgen import conformance as CF

    bundles = bundles if bundles is not None else _bundles()
    if not bundles:
        return {"target": target, "status": "no_captures",
                "detail": "no bundle under out/artifacts/recaptures/ carries a model.mlir, so no "
                          "requirement can be derived and nothing about circularity is established"}
    derivation, claim = CM.partition(bundles)
    covered = CM.covered_claim_models(bundles)
    uncaptured = sorted(m for m, bs in covered.items() if not bs)
    overlap = sorted(set(derivation) & set(claim))

    row: dict = {
        "target": target,
        "n_derivation": len(derivation),
        "n_claim": len(claim),
        "claim_models": {m: len(bs) for m, bs in sorted(covered.items())},
        "uncaptured_claim_models": uncaptured,
        "overlap": overlap,
    }
    if overlap:
        row["status"] = "overlap"
        row["detail"] = f"{len(overlap)} bundle(s) both derive and are claimed"
        return row
    if not derivation:
        row["status"] = "empty_derivation"
        row["detail"] = ("every capture is held out, so the requirement is derived from nothing; "
                         "disjointness is satisfied trivially and means nothing")
        return row

    def _cells(caps):
        cells, diag = CF.required_cells(target, caps)
        return sorted({(c.family, c.dtype, c.alignment) for c in cells}), diag

    try:
        with_claim, _ = _cells(bundles)
        without, diag = _cells(derivation)
        row["admitted_status"] = diag.get("admitted_status", "unknown")
    except Exception as exc:                       # noqa: BLE001 -- an unresolvable target establishes nothing
        row["status"] = "unverifiable"
        row["detail"] = f"could not derive the requirement: {type(exc).__name__}: {exc}"
        return row

    dependent = [c for c in with_claim if c not in without]
    row.update({
        "n_cells_all_captures": len(with_claim),
        "n_cells_derivation_only": len(without),
        "requirement_is_independent": not dependent,
        # A cell present ONLY when the held-out models are included: a requirement a claim model
        # demanded. Printable -- a cell is `admitted x observed`, which the tracked spec already states.
        "cells_depending_on_a_claim_model": [list(c) for c in dependent],
        "known_derivation_gaps": [dict(g) for g in CM.known_derivation_gaps()],
        # ⚠️ A ZERO-CELL DERIVATION IS NOT A PASS. Independence is vacuously true when the requirement
        # is empty -- nothing can depend on a held-out model if nothing is required at all -- so
        # reporting `ok` there is the "a check that could not run reported success" failure this repo
        # has paid for repeatedly. Measured: saturn_opu and saturn_opu_rvv derive no cell from any
        # capture, and this gate called both of them clean.
        "status": ("no_requirement" if not without else
                   "circular" if dependent else
                   "claim_model_uncaptured" if uncaptured else "ok"),
    })
    if row.get("admitted_status", "resolved") != "resolved":
        # Distinguished from "admits nothing a capture contains", because they license opposite
        # actions: generate the target's package, versus accept that the families do not intersect.
        row["status"] = "contract_unresolved"
        row["detail"] = (f"this target's capability contract did not resolve "
                         f"({row['admitted_status']}), so the requirement is UNKNOWN, not empty")
    elif not without:
        row["detail"] = ("the derivation set yields NO requirement cell, so independence is vacuous "
                         "and nothing about circularity is established: this target's manifest admits "
                         "no family any derivation capture contains")
    elif dependent:
        row["detail"] = (f"{len(dependent)} requirement cell(s) exist only because a held-out model "
                         f"was read; the corpus would be built from what it claims to generalize to")
    elif uncaptured:
        row["detail"] = (f"claim model(s) {uncaptured} have no capture, so the claim is not measured "
                         f"over them at all")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on-circular", action="store_true")
    ap.add_argument("--fail-on-unverifiable", action="store_true")
    a = ap.parse_args(argv)

    try:
        declared = CM.claim_models()
    except Exception as exc:                       # noqa: BLE001
        msg = f"claim-model declaration unreadable: {type(exc).__name__}: {exc}"
        print(f"[FAIL] {msg}", file=sys.stderr)
        return 2 if a.fail_on_unverifiable else 0

    targets = a.target or _spec_targets()
    if not targets:
        print("[claim-set] no target ships a conformance spec; nothing to audit")
        return 2 if a.fail_on_unverifiable else 0

    bundles = _bundles()
    rows = [audit(t, bundles) for t in targets]
    report = {"claim_models": list(declared), "exclusion_rule": CM.exclusion_rule(),
              "forbidden_sources": list(CM.forbidden_sources()), "targets": rows}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"[claim-set] held out: {', '.join(declared)}")
        for r in rows:
            head = f"  {r['target']:14} {r['status']:22}"
            if r["status"] in ("ok", "circular", "claim_model_uncaptured"):
                print(f"{head} derivation={r['n_derivation']} claim={r['n_claim']} "
                      f"cells={r['n_cells_derivation_only']} "
                      f"independent={r['requirement_is_independent']}")
            elif r.get("detail"):
                # One line per target: the detail was printed here AND again below, so every
                # non-ok target reported its reason twice.
                print(f"{head}")
                print(f"                 -> {r['detail']}")
            else:
                print(f"{head}")
            if r["status"] in ("circular", "claim_model_uncaptured", "no_requirement") \
                    and r.get("detail"):
                print(f"                 -> {r['detail']}")
        for g in CM.known_derivation_gaps():
            print(f"  [gap] {g.get('family')}/{g.get('shape_class')}: {g.get('reason', '').strip()}")

    circular = [r for r in rows if r["status"] == "circular"]
    unverifiable = [r for r in rows if r["status"] in
                    ("no_captures", "unverifiable", "empty_derivation", "overlap",
                     "no_requirement", "contract_unresolved")]
    if a.fail_on_circular and circular:
        print(f"[FAIL] {len(circular)} target(s) derive a requirement from a held-out model",
              file=sys.stderr)
        return 1
    if a.fail_on_unverifiable and unverifiable:
        print(f"[FAIL] {len(unverifiable)} target(s) could not be verified", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
