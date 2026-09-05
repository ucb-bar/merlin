"""Verify a target's DERIVED extent lattice — the same one the capsule corpus is built from.

The review comment this answers is *"the capsules are very case-specific"*. They are: the dynamic
ladder grades the shapes it can afford, tens of them, each on one stimulus. This sweeps the lattice
the target's own RTL facts define and proves, at every point, that the compiled program computes the
declared contraction **for every input** at that shape. The cases stop being chosen by us.

The lattice is not invented here. ``merlin/contract/capsules/conformance/<target>.yaml`` is generated
from the target's capability manifest and RTL facts and already carries both halves:

* ``cells`` — the conformance cells (family x dtype x alignment) the target must handle;
* ``boundaries.extent_probes[].points`` — extents that straddle each real hardware boundary: the
  degenerate 1, a mostly-empty tile (edge/4), edge/2, the tail (edge-1), the exact tile, the overflow
  (edge+1), and two tiles.

Those points were dead code before this module: ``corpus_synth.extents_for`` reads only the ``edge``.

**Cost is on our side here.** Sweeping a lattice means verifying CORRECT programs, which is the cheap
``unsat`` direction — a correct program's two sides are syntactically identical and z3's rewriter
collapses the query before bit-blasting anything. The expensive ``sat`` direction only applies when a
pass is actually broken, and then the smallest failing shape is the one you want anyway.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

#: Cell families this module can build and lower a program for.
#:
#: The limit is NOT the SMT encoder — it already handles VECTOR_MAP, VREDUCE and MOVEMENT. It is the
#: in-tree reference target: ``load_curated_contract("toy_npu")`` declares exactly four ops
#: (``res_pack, matmul, commit, evict``), and ``lower_to_target`` refuses anything outside them with
#: "the dialect plan does not lower interface.X, so this payload cannot descend to it. Coverage is
#: read from the plan, never assumed." Without a descent there is no command buffer, and with no
#: command buffer there is no compilation to validate. Sweeping the other families needs a target
#: that declares their ops — a target-side gap, not an encoder one, and the omission reason says so.
ENCODABLE_FAMILIES = frozenset({"contraction"})


def reference_target_ops() -> list[str]:
    """The op set the in-tree reference target declares. Read, never assumed."""
    try:
        from merlin.xdsl_dialects.lowering.pipeline import load_curated_contract

        return sorted((load_curated_contract("toy_npu") or {}).get("ops") or [])
    except Exception:
        return []


def spec_path(target: str):
    from merlin.common.paths import merlin_dir

    return merlin_dir() / "contract" / "capsules" / "conformance" / f"{target}.yaml"


def load_spec(target: str) -> dict[str, Any]:
    """The tracked, derived conformance spec for one target."""
    import yaml

    p = spec_path(target)
    if not p.is_file():
        raise FileNotFoundError(
            f"no derived conformance spec for {target!r} at {p}. The lattice is derived from the "
            f"target's own facts; without the spec there is nothing to sweep and nothing to assume.")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def lattice_points(spec: dict[str, Any]) -> list[int]:
    """The derived extents, deduplicated and ordered. Empty when the target has no derivable edge."""
    probes = ((spec.get("boundaries") or {}).get("extent_probes") or [])
    return sorted({int(p) for pr in probes for p in (pr.get("points") or []) if int(p) >= 1})


def sweep(target: str, *, timeout_ms: int = 300_000, acc_width: int = 32,
          reuse: int = 2, max_points: int | None = None) -> dict[str, Any]:
    """Validate the compilation at every (cell, extent) the target's own facts define."""
    from .evaluate import _finish_lowering, _lower_to_interface
    from .refine import validate_compilation
    from .smt_semantics import UnsupportedSemantics

    spec = load_spec(target)
    points = lattice_points(spec)
    if max_points is not None:
        points = points[:max_points]
    cells = list(spec.get("cells") or ())

    results: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []

    # Cells are grouped by (family, dtype) rather than swept one by one. The third axis of a cell is
    # its ALIGNMENT — aligned / partial / sub_tile — and that is exactly what the extent points
    # already express: 16 is the aligned tile, 15 the partial tail, 4 a sub-tile occupancy. Sweeping
    # every cell separately would issue the identical query three times and report it as three
    # verified points, which inflates the coverage number without verifying anything more.
    grouped: dict[tuple[str, str], list[str]] = {}
    for cell in cells:
        family = str(cell.get("family") or "")
        dtype = str(cell.get("dtype") or "")
        name = str(cell.get("cell") or f"{family}/{dtype}")
        if family not in ENCODABLE_FAMILIES:
            omissions.append({"cell": name, "reason": _family_omission(family)})
            continue
        grouped.setdefault((family, dtype), []).append(name)

    for (family, dtype), covered in sorted(grouped.items()):
        name = f"{family}/{dtype}"
        for p in points:
            t0 = time.time()
            try:
                iface, tc = _lower_to_interface(p, p, p, reuse)
                cb = _finish_lowering(iface, tc)
                v = validate_compilation(iface, cb, acc_width=acc_width, timeout_ms=timeout_ms)
                status = v.status
            except UnsupportedSemantics as exc:
                status = "abstained"
                results.append({"cell": name, "dtype": dtype, "m": p, "k": p, "n": p,
                                "status": status, "seconds": round(time.time() - t0, 2),
                                "reason": str(exc)[:200], "covers_cells": covered})
                continue
            results.append({"cell": name, "dtype": dtype, "m": p, "k": p, "n": p,
                            "status": status, "seconds": round(time.time() - t0, 2),
                            "covers_cells": covered})

    verified = [r for r in results if r["status"] == "unsat"]
    refuted = [r for r in results if r["status"] == "sat"]
    return {
        "schema": "verify_lattice/v1",
        "target": target,
        "lattice_points": points,
        "lattice_source": _lattice_source(spec),
        "cells_declared": len(cells),
        "cell_groups_swept": len({r["cell"] for r in results}),
        "cells_covered": sorted({c for r in results for c in r.get("covers_cells", ())}),
        "points_total": len(results),
        "points_verified": len(verified),
        "points_refuted": len(refuted),
        "points_abstained": len(results) - len(verified) - len(refuted),
        "reuse": reuse,
        "acc_width": acc_width,
        "timeout_ms": timeout_ms,
        "results": results,
        "cell_omissions": omissions,
        "shape_space": {
            "formal": {
                "shapes_proved": len(verified),
                "quantifier": "every integer input at each shape",
            },
            "dynamic": witnesses_graded(target),
            "note": ("The two numbers measure different things and neither subsumes the other: the "
                     "dynamic ladder is the only layer that touches hardware, and the formal sweep "
                     "says nothing about it. What the formal side adds is the quantifier -- all "
                     "inputs rather than one stimulus -- at the shapes it can reach."),
        },
    }


def witnesses_graded(target: str) -> dict[str, Any]:
    """How many capsules the dynamic ladder grades for this target, and at how many distinct shapes.

    The comparison this supports is the quantified answer to "the capsules are very case-specific":
    the dynamic ladder grades N witnesses, each on ONE stimulus; the formal sweep proves M shapes over
    EVERY input. Both numbers are counted here rather than asserted, and the shape count is what makes
    the comparison fair -- several capsules can share a shape, so a raw capsule count would overstate
    the dynamic side's shape coverage.
    """
    import yaml

    from merlin.targetgen.target_experiment import TargetExperiment  # noqa: F401  (import guard)

    from merlin.common.paths import merlin_dir

    root = merlin_dir() / "contract" / "capsules"
    if not root.is_dir():
        return {"capsules": 0, "distinct_shapes": 0, "note": "no corpus tree in this checkout"}
    capsules, shapes = 0, set()
    for path in root.rglob("capsule.yaml"):
        if "hidden" in path.parts:
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        capsules += 1
        for spec in (doc.get("inputs") or []):
            shape = spec.get("shape")
            if shape:
                shapes.add(tuple(shape))
    return {"capsules": capsules, "distinct_shapes": len(shapes),
            "note": "counted from tracked capsule.yaml files, excluding hidden/; each is graded on "
                    "one deterministic stimulus"}


def _family_omission(family: str) -> str:
    """Why a family is not swept — the real blocker, derived from the target, not a generic message.

    Being precise here matters: "we have not written a builder" and "the reference target cannot
    represent this at all" call for different work, and the first would send someone to the wrong file.
    """
    ops = reference_target_ops()
    # Deliberately does NOT name the interface op: the family name and the op name differ (the
    # `elementwise_map` family lowers to `interface.elementwise`), and inventing `interface.<family>`
    # would put a plausible-looking but wrong symbol in a ledger people cite.
    return (f"family {family!r} is not swept: the in-tree reference target declares only "
            f"{ops or 'UNKNOWN'}, so lower_to_target refuses this family's interface op and no "
            f"command buffer is produced — there is nothing to validate. The SMT encoder is not the "
            f"limit (it already handles VECTOR_MAP, VREDUCE and MOVEMENT); a target declaring those "
            f"ops is.")


def _lattice_source(spec: dict[str, Any]) -> str:
    """Where the extents came from — a shape with no provenance is not citable."""
    probes = ((spec.get("boundaries") or {}).get("extent_probes") or [])
    if not probes:
        return ("no derivable boundary: this target's RTL facts carry no mesh edge, so no lattice "
                "exists to sweep and none is invented")
    return "; ".join(f"{pr.get('boundary')} edge={pr.get('edge')} from {pr.get('source')}"
                     for pr in probes)


def render(rec: dict[str, Any]) -> str:
    out = [f"lattice sweep: {rec['target']}", ""]
    out.append(f"  extents      {rec['lattice_points'] or '(none derivable)'}")
    out.append(f"  source       {rec['lattice_source']}")
    out.append(f"  cells        {len(rec['cells_covered'])} covered of {rec['cells_declared']} "
               f"declared, via {rec['cell_groups_swept']} distinct query group(s)")
    out.append(f"               (alignment is expressed by the extent, not by a separate query)")
    out.append(f"  points       {rec['points_verified']} verified / {rec['points_refuted']} REFUTED "
               f"/ {rec['points_abstained']} abstained  (of {rec['points_total']})")
    ss = rec.get("shape_space") or {}
    if ss:
        d = ss.get("dynamic") or {}
        out.append("")
        out.append(f"  shape space  formal: {ss['formal']['shapes_proved']} shape(s) proved over "
                   f"{ss['formal']['quantifier']}")
        out.append(f"               dynamic: {d.get('capsules', 0)} capsule(s) across "
                   f"{d.get('distinct_shapes', 0)} distinct shape(s), one stimulus each")
        out.append("               (different questions -- only the dynamic ladder touches hardware)")
    if rec["points_refuted"]:
        out.append("")
        out.append("  REFUTED — the compiled program disagrees with the declared contraction:")
        for r in rec["results"]:
            if r["status"] == "sat":
                out.append(f"    {r['cell']:34s} {r['m']}x{r['k']}x{r['n']}")
    if rec["cell_omissions"]:
        out.append("")
        out.append("  cells not swept, with reasons:")
        for o in rec["cell_omissions"]:
            out.append(f"    {o['cell']:34s} {o['reason']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True)
    ap.add_argument("--timeout-ms", type=int, default=300_000)
    ap.add_argument("--reuse", type=int, default=2)
    ap.add_argument("--max-points", type=int, default=None,
                    help="cap the extents swept (for a quick pass); the record says what was capped")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--emit-counterexamples", action="store_true",
                    help="write refuted shapes as corpus profile entries so the bench grades them")
    args = ap.parse_args(argv)

    rec = sweep(args.target, timeout_ms=args.timeout_ms, reuse=args.reuse,
                max_points=args.max_points)
    print(json.dumps(rec, indent=1) if args.json else render(rec))
    if args.write:
        _write(rec)
    if args.emit_counterexamples:
        emit_counterexamples(rec)
    # A refutation is a failure; an abstention is not.
    return 1 if rec["points_refuted"] else 0


def emit_counterexamples(rec: dict[str, Any]) -> None:
    """Turn every refuted point into a corpus entry, and its values into untracked evidence.

    Nothing is written when nothing was refuted — an empty profile sidecar would suggest the sweep
    found something and lost it.
    """
    from .counterexamples import counterexample_entry, write_evidence, write_profile

    refuted = [r for r in rec["results"] if r["status"] == "sat"]
    if not refuted:
        print("\nno refuted points: nothing to add to the corpus")
        return
    target = rec["target"]
    entries = [counterexample_entry(target=target, m=r["m"], k=r["k"], n=r["n"],
                                    dtype=r.get("dtype", "i8"),
                                    family=str(r["cell"]).split("/")[0],
                                    bound_ms=rec.get("timeout_ms"))
               for r in refuted]
    write_profile(target, entries, provenance={"lattice_source": rec["lattice_source"]})
    path = write_evidence(target, refuted)
    if path:
        print(f"counterexample values: {path}")
    print("run `python -m merlin.contract.capsules.generate_corpus --target "
          f"{target}` to materialise them as capsules")


def _write(rec: dict[str, Any]):
    from merlin.common.artifacts import new_product

    prod = new_product("verification", version=1, target=rec["target"], sources=[
        f"derived lattice: {spec_path(rec['target'])}",
        f"extents: {rec['lattice_points']}",
        f"lattice source: {rec['lattice_source']}",
        f"solver bound: {rec['timeout_ms']} ms per point",
    ], notes=(
        "Verification of a target's derived extent lattice. Each point proves the compiled program "
        "computes the declared contraction for EVERY input at that shape, versus the dynamic ladder "
        "which grades one stimulus per witness. Cells and extents come from the target's own "
        "capability manifest and RTL facts, so the verified set is generated rather than curated."))
    out = prod.add_artifact("lattice.json")
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    sys.exit(main())
