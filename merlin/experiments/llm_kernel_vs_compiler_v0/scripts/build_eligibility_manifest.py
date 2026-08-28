#!/usr/bin/env python3
"""Build the eligibility manifest -- the denominator Merlin is graded against.

Coverage is `accelerated / eligible`, so whoever sets `eligible` sets the score. If Merlin's own
capability declaration supplied it, the ratio would be partly self-graded. This script therefore builds
the denominator from evidence that exists independently of Merlin -- kernels a HUMAN wrote against this
hardware, in the hardware's own repo (`eligibility/family_evidence.yaml`, itself gated on the
provenance audit) -- and reports Merlin's declaration beside it as a cross-check, never as the source.

Three columns per family, and the disagreements are the point:

  independent   a hand-written kernel for this hardware implements the family
  declared      the target contract claims the family
  derived       what Merlin's own evidence pipeline could actually corroborate

The union is what a fair denominator uses. Note which way the bias runs: a SMALLER denominator makes
Merlin look BETTER, and Merlin's derivation currently corroborates the fewest families of the three, so
deferring to it would flatter the result. Any family that is independently evidenced counts, whatever
Merlin thinks of it.

Usage:
    build_eligibility_manifest.py --target radiance --out radiance_eligibility_manifest.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def _provenance_hand_kernels(prov_path: Path) -> set[str]:
    doc = yaml.safe_load(prov_path.read_text())
    return {k["name"] for k in doc.get("kernels", []) if k.get("verdict") == "hand"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--evidence", type=Path, default=EXP / "eligibility" / "family_evidence.yaml")
    ap.add_argument("--provenance", type=Path,
                    default=EXP / "eligibility" / "provenance" / "kernel_provenance.yaml")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    ev_doc = yaml.safe_load(a.evidence.read_text())
    if ev_doc.get("target") != a.target:
        raise SystemExit(f"evidence file is for {ev_doc.get('target')!r}, not {a.target!r}")

    hand = _provenance_hand_kernels(a.provenance)
    required = ev_doc.get("evidence_repo", {}).get("provenance_verdict_required", "hand")
    if required != "hand":
        raise SystemExit(f"unsupported provenance_verdict_required: {required!r}")

    from merlin.targetgen import eligibility as EL

    cap_map = EL.capability_map_for_target(a.target)
    undetermined = set(EL.undetermined_families_for_target(a.target))
    providers = EL.providers_for_target(a.target)

    families: dict[str, dict] = {}
    unciteable: dict[str, list[str]] = {}
    for fam, spec in sorted(ev_doc.get("families", {}).items()):
        cited = list(spec.get("evidenced_by") or [])
        # Only kernels the audit called `hand` may carry evidence. A citation to anything else is
        # dropped and reported -- silently keeping it would let generated code define the denominator.
        good = [k for k in cited if k in hand]
        bad = [k for k in cited if k not in hand]
        if bad:
            unciteable[fam] = bad

        declared = fam in cap_map
        derived_ok = declared and fam not in undetermined
        cap = cap_map.get(fam)
        families[fam] = {
            "independently_evidenced": bool(good),
            "evidence_kernels": good,
            "evidence_dropped_not_hand": bad,
            "declared_by_target_contract": declared,
            "merlin_derivation_undetermined": fam in undetermined,
            "engines": list(cap.engines) if cap is not None else [],
            "dtypes": list(cap.dtypes) if cap is not None else [],
            "ranks": list(cap.ranks) if cap is not None else [],
            "providers": [list(p) for p in providers.get(fam, ())],
            "in_denominator": bool(good) or derived_ok,
            "denominator_basis": (
                "independent_hand_kernel" if good
                else "target_contract_declaration" if derived_ok
                else "excluded"
            ),
            "note": spec.get("note", "").strip(),
        }

    in_denom = sorted(f for f, v in families.items() if v["in_denominator"])
    only_independent = sorted(
        f for f, v in families.items()
        if v["independently_evidenced"] and v["merlin_derivation_undetermined"]
    )

    doc = {
        "schema": "eligibility_manifest/v1",
        "target": a.target,
        "generated_by": "llm_kernel_vs_compiler_v0/scripts/build_eligibility_manifest.py",
        "denominator_source": (
            "independent hand-written kernels for this hardware, unioned with the target "
            "contract's own declaration where Merlin's derivation corroborates it"
        ),
        "independence_argument": (
            "The families in the denominator are established by kernels a human wrote against this "
            "hardware in the hardware's own repository, classified `hand` by the provenance audit. "
            "That evidence exists whether or not Merlin does, so Merlin is not setting the "
            "denominator it is graded against."
        ),
        "bias_direction": (
            "Coverage is accelerated/eligible, so a SMALLER denominator flatters the compiler. "
            f"Merlin's own derivation leaves {sorted(undetermined)} undetermined; the independent "
            "evidence covers them. Using the larger independent denominator is the conservative "
            "choice and is what this manifest does."
        ),
        "sources": {
            "family_evidence": str(a.evidence.relative_to(EXP)),
            "provenance": str(a.provenance.relative_to(EXP)),
            "evidence_repo": ev_doc.get("evidence_repo", {}),
            "n_hand_kernels_available": len(hand),
        },
        "requires_human_review": bool(ev_doc.get("requires_human_review", True)),
        "families_in_denominator": in_denom,
        "families_excluded": sorted(set(families) - set(in_denom)),
        "families_only_independent_evidence": only_independent,
        "families": families,
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(doc, sort_keys=False, width=100))

    print(f"wrote {a.out}")
    print(f"  denominator families ({len(in_denom)}): {', '.join(in_denom)}")
    print(f"  excluded: {doc['families_excluded'] or 'none'}")
    print(f"  independently evidenced but UNDETERMINED in merlin's own derivation: "
          f"{', '.join(only_independent) or 'none'}")
    if unciteable:
        print("  ⚠️ dropped citations (kernel is not provenance-`hand`):")
        for fam, ks in sorted(unciteable.items()):
            print(f"      {fam}: {ks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
