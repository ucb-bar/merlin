"""Verify a target's DERIVED extent lattice — the same one the capsule corpus is built from.

The review comment this answers is *"the capsules are very case-specific"*. They are: the dynamic
ladder grades the shapes it can afford, tens of them, each on one stimulus. This attempts to sweep
the lattice the target's own RTL facts define. A successfully verified point proves that the
compiled program computes the declared contraction **for every input** at that shape.

The lattice is not invented here. A reviewed conformance requirement is generated
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
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---- lattice -----------------------------------------------------------------------------------

#: The only workload this module currently builds is quantized i8 contraction. A cell with a
#: different family or dtype cannot borrow that workload's proof.
ENCODABLE_FAMILIES = frozenset({"contraction"})
ENCODABLE_DTYPES = frozenset({"i8"})


def spec_path(target: str, *, conformance_spec: str | Path | None = None) -> Path:
    if conformance_spec is None:
        raise ValueError(f"select the generated conformance requirement for {target!r} explicitly with --spec PATH")
    return Path(conformance_spec)


def load_spec(target: str, *, conformance_spec: str | Path | None = None) -> dict[str, Any]:
    """The explicitly selected derived conformance requirement for one target."""
    spec, _ = _selected_spec(target, conformance_spec=conformance_spec)
    return spec


def _selected_spec(target: str, *, conformance_spec: str | Path | None) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse and fingerprint the same bytes; selection alone does not approve their scope."""
    import yaml

    p = spec_path(target, conformance_spec=conformance_spec)
    if not p.is_file():
        raise FileNotFoundError(
            f"no derived conformance spec for {target!r} at {p}. The lattice is derived from the "
            f"target's own facts; without the spec there is nothing to sweep and nothing to assume."
        )
    source = p.read_bytes()
    return yaml.safe_load(source) or {}, {
        "path": str(p.resolve()),
        "sha256": hashlib.sha256(source).hexdigest(),
        "scope": "caller_selected_requirement; independent approval not established by this sweep",
    }


def lattice_points(spec: dict[str, Any]) -> list[int]:
    """The derived extents, deduplicated and ordered. Empty when the target has no derivable edge."""
    probes = (spec.get("boundaries") or {}).get("extent_probes") or []
    return sorted({int(p) for pr in probes for p in (pr.get("points") or []) if int(p) >= 1})


def sweep(
    target: str,
    *,
    conformance_spec: str | Path | None = None,
    timeout_ms: int = 300_000,
    acc_width: int = 32,
    reuse: int = 2,
    max_points: int | None = None,
) -> dict[str, Any]:
    """Validate the compilation at every (cell, extent) the target's own facts define."""
    from merlin.verification.translation.fault_injection import _finish_lowering, _lower_to_interface
    from merlin.verification.translation.refinement import validate_compilation

    spec, selected = _selected_spec(target, conformance_spec=conformance_spec)
    available_points = lattice_points(spec)
    points = available_points
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
        if dtype not in ENCODABLE_DTYPES:
            omissions.append({"cell": name, "reason": _dtype_omission(dtype)})
            continue
        grouped.setdefault((family, dtype), []).append(name)

    for (family, dtype), covered in sorted(grouped.items()):
        name = f"{family}/{dtype}"
        for p in points:
            t0 = time.time()
            try:
                iface, tc = _lower_to_interface(p, p, p, reuse, target=target)
                cb = _finish_lowering(iface, tc)
                if cb.get("target") != target:
                    raise ValueError(f"lowered command buffer names {cb.get('target')!r}, expected {target!r}")
                v = validate_compilation(iface, cb, acc_width=acc_width, timeout_ms=timeout_ms)
                status = v.status
            except Exception as exc:  # noqa: BLE001 — a failed lowering/solver cannot count as proof
                from merlin.compiler.dialects.lowering.interface_stage import LoweringError
                from merlin.verification.translation.smt_encoding import UnsupportedSemantics

                status = "abstained" if isinstance(exc, (LoweringError, UnsupportedSemantics)) else "error"
                results.append(
                    {
                        "cell": name,
                        "dtype": dtype,
                        "m": p,
                        "k": p,
                        "n": p,
                        "status": status,
                        "seconds": round(time.time() - t0, 2),
                        "reason": f"{type(exc).__name__}: {exc}"[:200],
                        "covers_cells": covered,
                    }
                )
                continue
            results.append(
                {
                    "cell": name,
                    "dtype": dtype,
                    "m": p,
                    "k": p,
                    "n": p,
                    "status": status,
                    "seconds": round(time.time() - t0, 2),
                    "covers_cells": covered,
                }
            )

    verified = [r for r in results if r["status"] == "unsat"]
    refuted = [r for r in results if r["status"] == "sat"]
    cells_verified_at_sampled_points = sorted(
        c
        for c in {c for r in results for c in r["covers_cells"]}
        if points and all(q["status"] == "unsat" for q in results if c in q["covers_cells"])
    )
    cells_covered = cells_verified_at_sampled_points if points == available_points else []
    errors = [r for r in results if r["status"] == "error"]
    return {
        "conformance_spec": selected,
        "schema": "verify_lattice/v2",
        "target": target,
        "lattice_points": points,
        "lattice_points_available": available_points,
        "max_points": max_points,
        "lattice_source": _lattice_source(spec),
        "cells_declared": len(cells),
        "cell_groups_swept": len({r["cell"] for r in results}),
        "cells_swept": sorted({c for r in results for c in r.get("covers_cells", ())}),
        "cells_verified_at_sampled_points": cells_verified_at_sampled_points,
        "cells_covered": cells_covered,
        "points_total": len(results),
        "points_verified": len(verified),
        "points_refuted": len(refuted),
        "points_abstained": len(results) - len(verified) - len(refuted) - len(errors),
        "points_error": len(errors),
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
            "dynamic_declared": witnesses_declared(target),
            "note": (
                "Formal proofs are measured here; dynamic capsules are only counted from the "
                "selected target's declared suite, not from grade receipts. The dynamic ladder "
                "touches hardware when run; the formal sweep does not."
            ),
        },
    }


def witnesses_declared(target: str) -> dict[str, Any]:
    """Count declared capsules and distinct shapes in this target's selected graded suite.

    This measures available witnesses, not completed grading. Execution requires grade receipts and
    cannot be inferred from a tracked capsule.yaml. Several capsules can share one shape.
    """
    import yaml

    from merlin.target.registry.descriptor_loading import graded_capsule_roots

    roots = graded_capsule_roots(target)
    if not roots:
        return {"capsules": 0, "distinct_shapes": 0, "note": "no descriptor-selected graded roots for this target"}
    capsules, shapes, unreadable = 0, set(), []
    paths = {path for root in roots for path in root.rglob("capsule.yaml") if "hidden" not in path.parts}
    for path in sorted(paths):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            # Counted, never dropped: an unreadable declaration is not an absent one.
            unreadable.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        capsules += 1
        for spec in doc.get("inputs") or []:
            shape = spec.get("shape")
            if shape:
                shapes.add(tuple(shape))
    return {
        "capsules": capsules,
        "distinct_shapes": len(shapes),
        "unreadable": unreadable,
        "note": "counted from this target's descriptor-selected graded roots, excluding hidden/; "
        "availability only, with no grade receipt checked",
    }


def _family_omission(family: str) -> str:
    """Explain the unbuilt workload without inferring backend capability."""
    return (
        f"family {family!r} is not swept: this verifier has no workload builder for it; "
        f"no target capability or SMT verdict was inferred."
    )


def _dtype_omission(dtype: str) -> str:
    return f"dtype {dtype!r} is not swept: this verifier builds only i8 quantized matmul workloads"


def _lattice_source(spec: dict[str, Any]) -> str:
    """Where the extents came from — a shape with no provenance is not citable."""
    probes = (spec.get("boundaries") or {}).get("extent_probes") or []
    if not probes:
        return (
            "no derivable boundary: this target's RTL facts carry no mesh edge, so no lattice "
            "exists to sweep and none is invented"
        )
    return "; ".join(f"{pr.get('boundary')} edge={pr.get('edge')} from {pr.get('source')}" for pr in probes)


def render(rec: dict[str, Any]) -> str:
    out = [f"lattice sweep: {rec['target']}", ""]
    out.append(f"  extents      {rec['lattice_points'] or '(none derivable)'}")
    out.append(f"  source       {rec['lattice_source']}")
    out.append(
        f"  cells        {len(rec['cells_covered'])} fully verified of {rec['cells_declared']} "
        f"declared, via {rec['cell_groups_swept']} distinct query group(s)"
    )
    if rec.get("max_points") is not None:
        out.append(
            f"               {len(rec['cells_verified_at_sampled_points'])} cell(s) verified at sampled points; "
            f"{len(rec['lattice_points'])} of {len(rec['lattice_points_available'])} extent(s) selected"
        )
    out.append("               (alignment is expressed by the extent, not by a separate query)")
    out.append(
        f"  points       {rec['points_verified']} verified / {rec['points_refuted']} REFUTED "
        f"/ {rec['points_abstained']} abstained / {rec['points_error']} ERROR "
        f"(of {rec['points_total']})"
    )
    unavailable = [r for r in rec["results"] if r["status"] in {"abstained", "error", "unknown"}]
    if unavailable:
        out.append("  no proof for:")
        for r in unavailable:
            out.append(f"    {r['cell']:34s} {r['m']}x{r['k']}x{r['n']}: {r.get('reason', r['status'])}")
    ss = rec.get("shape_space") or {}
    if ss:
        d = ss.get("dynamic_declared") or {}
        out.append("")
        out.append(
            f"  shape space  formal: {ss['formal']['shapes_proved']} shape(s) proved over {ss['formal']['quantifier']}"
        )
        out.append(
            f"               dynamic available: {d.get('capsules', 0)} declared capsule(s) across "
            f"{d.get('distinct_shapes', 0)} distinct shape(s); execution not measured"
        )
        out.append("               (only the dynamic ladder touches hardware when it runs)")
        if d.get("unreadable"):
            out.append(
                f"               {len(d['unreadable'])} declared capsule(s) could not be read and are not counted"
            )
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--target", required=True)
    ap.add_argument("--spec", type=Path, required=True, help="explicit generated conformance requirement")
    ap.add_argument("--timeout-ms", type=int, default=300_000)
    ap.add_argument("--reuse", type=int, default=2)
    ap.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="cap the extents swept (for a quick pass); the record says what was capped",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument(
        "--emit-counterexamples",
        action="store_true",
        help="write refuted shapes as corpus profile entries so the bench grades them",
    )
    ap.add_argument("--smt-profile", type=Path, help="explicit experiment-declared SMT sidecar output")
    args = ap.parse_args(argv)
    if args.emit_counterexamples != (args.smt_profile is not None):
        ap.error("--emit-counterexamples and --smt-profile must be supplied together")
    if args.smt_profile is not None and args.smt_profile.exists() and not args.smt_profile.is_file():
        ap.error("--smt-profile must name a file, not a directory")

    rec = sweep(
        args.target,
        conformance_spec=args.spec,
        timeout_ms=args.timeout_ms,
        reuse=args.reuse,
        max_points=args.max_points,
    )
    print(json.dumps(rec, indent=1) if args.json else render(rec))
    if args.write:
        _write(rec)
    if args.emit_counterexamples:
        emit_counterexamples(rec, profile=args.smt_profile)
        # A refutation or unexpected tooling error is a failure; a recorded abstention is not.
    return 1 if rec["points_refuted"] or rec["points_error"] else 0


def emit_counterexamples(rec: dict[str, Any], *, profile: Path) -> None:
    """Turn every refuted point into a corpus entry, and its values into untracked evidence.

    Nothing is written when nothing was refuted — an empty profile sidecar would suggest the sweep
    found something and lost it.
    """

    refuted = [r for r in rec["results"] if r["status"] == "sat"]
    if not refuted:
        print("\nno refuted points: nothing to add to the corpus")
        return
    target = rec["target"]
    entries = [
        counterexample_entry(
            target=target,
            m=r["m"],
            k=r["k"],
            n=r["n"],
            dtype=r.get("dtype", "i8"),
            family=str(r["cell"]).split("/")[0],
            bound_ms=rec.get("timeout_ms"),
        )
        for r in refuted
    ]
    write_profile(
        profile,
        entries,
        provenance={"lattice_source": rec["lattice_source"], "conformance_spec": rec["conformance_spec"]},
    )
    path = write_evidence(target, refuted)
    if path:
        print(f"counterexample values: {path}")
    print(f"regenerate the experiment's Phase0 corpus with --smt-profile {profile} to materialise these shapes")


def _write(rec: dict[str, Any]):
    from merlin.common.artifacts import new_product

    prod = new_product(
        "verification",
        version=1,
        target=rec["target"],
        sources=[
            f"derived lattice: {rec['conformance_spec']['path']} (sha256={rec['conformance_spec']['sha256']})",
            f"extents: {rec['lattice_points']}",
            f"lattice source: {rec['lattice_source']}",
            f"solver bound: {rec['timeout_ms']} ms per point",
        ],
        notes=(
            "Verification attempts over a target's derived extent lattice. A verified point proves the "
            "compiled program computes the declared contraction for EVERY input at that shape, "
            "while declared dynamic capsules are counted without grade receipts. Cells and extents "
            "come from the target's own capability manifest and RTL facts."
            " The selected requirement fixes this proof's scope; this run does not independently approve it "
            "or establish hardware execution."
        ),
    )
    out = prod.add_artifact("lattice.json")
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nwrote {out}")
    return out

    # Preserve historical object lookup; implementation and completion state live here.


# ---- counterexamples ---------------------------------------------------------------------------
# Turn refutations into corpus entries the normal generator builds.
#
# When the lattice sweep refutes a point, the shape that exposed the defect should not stay in a log —
# it should become a capsule the bench grades from then on. This module writes those shapes as PROFILE
# ENTRIES rather than as capsule directories, which is the difference between joining the corpus and
# sitting beside it.
#
# **Why a profile entry and not a directory.** ``merlin.experiments.phase0`` is what
# turns an entry into a capsule: it writes ``capsule.yaml``, ``capsule.interface.mlir`` and the README,
# computes and writes the untracked ``golden.yaml``, scrubs the directory, and records the capsule in
# ``MANIFEST.yaml`` as *generated*. An earlier version of this work wrote the directory itself and got
# none of that — no golden for the grader, and ``update_provenance_manifest`` would have classified a
# solver-produced capsule as ``hand_authored``, which is exactly backwards.
#
# The experiment declares the SMT sidecar path explicitly. This core writer neither discovers
# recipes nor imports the experiment catalog. The authoring edge passes the same path to Phase0's
# ``load_profile(..., smt_profile=...)``; writing an arbitrary sidecar does not enroll it automatically.
# The historical explicit ``profiles_root`` adapter still reads its fixed sidecar chain, not a glob.
#
# **What the entry carries, and what it does not.** It carries the SHAPE and configuration the solver
# found. It does not carry the solver's input values, because a capsule has nowhere to put them:
# ``capsule.schema.json``'s ``inputs[]`` has no values field and sets ``additionalProperties: false``,
# and ``capsule_golden.materialize_capsule_leaves`` fills every leaf unconditionally with
# ``Tensor.deterministic``. The values are written to an untracked artifact instead, where they are
# evidence for a human rather than an input to the grader. Claiming otherwise would overstate what this
# path does — see the 2026-09-05 correction in docs/design/compiler_verification.md.

#: Prefix for solver-produced capsules. Distinct from ``corpus_synth.SYNTH_PREFIX`` ("SY") so a name
#: collision is impossible and a reader can tell at a glance which generator produced a capsule.
CX_PREFIX = "CX"

#: The schema's existing enum value for this provenance — no schema change is needed.
CX_ROLE = "smt_counterexample"


def entry_name(m: int, k: int, n: int, *, dtype: str, family: str) -> str:
    """A stable, collision-free name. Schema requires ``^[A-Za-z0-9_]+$``."""
    return f"{CX_PREFIX}_{family}_{dtype}_{m}x{k}x{n}".replace("-", "_")


def counterexample_entry(
    *,
    target: str,
    m: int,
    k: int,
    n: int,
    dtype: str = "i8",
    family: str = "contraction",
    obligation: str = "",
    solver: str = "",
    bound_ms: int | None = None,
    evidence_path: str | None = None,
) -> dict[str, Any]:
    """One profile entry for a refuted lattice point.

    Extents are CONCRETE integers rather than the tile-relative spellings the synthesized entries
    use ("tile", "2*tile-1"). The point of a counterexample is the specific shape that broke, so
    re-deriving it from a tile edge later would lose exactly the information worth keeping.
    """
    where = f"{m}x{k}x{n}"
    reference = (
        f"refuted by SMT translation validation at {where}"
        + (f" for obligation {obligation!r}" if obligation else "")
        + (f"; solver {solver}" if solver else "")
        + (f", bound {bound_ms} ms" if bound_ms is not None else "")
        + ". The SHAPE is the solver's; the stimulus is the corpus's own deterministic fill, because "
        "a capsule has no field for input values"
        + (f". Counterexample values: {evidence_path}" if evidence_path else "")
    )
    return {
        "cat": "isa",
        "kind": "isa",
        "name": entry_name(m, k, n, dtype=dtype, family=family),
        "op": "matmul",
        "operand_dtype": dtype,
        "out": "Y0",
        "lhs": "A0",
        "weight": "W",
        "source_role": CX_ROLE,
        "source_reference": reference,
        "label": "public",
        "modes": {},
        "M": int(m),
        "K": int(k),
        "N": int(n),
        "pass_requirements": ["target-isa-lowering", "tile-schedule"],
    }


def merge_entries(existing: list[dict], new: list[dict]) -> tuple[list[dict], int]:
    """Union by name, newest winning. Returns the merged list and how many were added.

    De-duplicating by name matters because the same shape can refute on successive runs; without it
    the profile would grow without bound and ``expand_sweeps`` would raise on the duplicate.
    """
    by_name = {e["name"]: e for e in existing}
    added = sum(1 for e in new if e["name"] not in by_name)
    by_name.update({e["name"]: e for e in new})
    return [by_name[k] for k in sorted(by_name)], added


def write_profile(path: Path, entries: list[dict], *, provenance: dict | None = None) -> Path:
    """Write or extend the experiment's explicitly declared SMT sidecar."""
    import yaml

    path = Path(path)
    existing: list[dict] = []
    if path.is_file():
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        existing = list(doc.get("capsules") or [])
    merged, added = merge_entries(existing, entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "provenance": dict(
                    provenance or {},
                    generator="merlin.verification.translation.extent_lattice",
                    note=(
                        "shapes refuted by SMT translation validation; the stimulus is the "
                        "corpus's deterministic fill, not the solver's values"
                    ),
                ),
                "capsules": merged,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"{path}: {len(merged)} entr{'y' if len(merged) == 1 else 'ies'} ({added} new)")
    return path


def write_evidence(target: str, records: list[dict]) -> Path | None:
    """Record the counterexample VALUES as an untracked artifact.

    Deliberately not in the corpus tree: a capsule directory's tracked files are its public contract,
    and input values that expose a defect are neither part of that contract nor something the grader
    can consume. Here they are auditable evidence for whoever reads the refutation.
    """
    if not records:
        return None
    import json

    from merlin.common.artifacts import new_product

    prod = new_product(
        "verification",
        version=1,
        target=target,
        sources=[
            f"{len(records)} refuted lattice point(s) for {target}",
            "values are the solver's model; the corpus grades the deterministic fill instead",
        ],
        notes=(
            "Counterexample values for refuted lattice points. Evidence for a human reading the "
            "refutation -- a refutation without its counterexample is an assertion -- and NOT an "
            "input to the grader, which has no field for input values."
        ),
    )
    out = prod.add_artifact("counterexamples.json")
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    prod.write_manifest()
    return out

    # Preserve historical object lookup; implementation and completion state live here.


if __name__ == "__main__":
    sys.exit(main())
