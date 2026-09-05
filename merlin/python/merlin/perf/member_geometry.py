"""Which shape class an OBJECTIVE member occupies, and whether real models present that class.

A perf member exists to make the generated code get faster on shapes that matter. Nothing in the
corpus said which shapes those were, so the question "does this member represent anything" was
answerable only by someone who went and measured it. MEASURED, once: of the 29 classifiable OBJECTIVE
members on this repo's own corpus, 27 fall in a geometric class the target's measured census does not
contain at all, the other two fall in the class the census marks UNREACHABLE, and all four reachable
classes have no members. The corpus was not under-sampling the models; it was off their manifold.

This module makes that fact travel WITH the capsule instead of living in a script somebody has to
remember to run. It answers three things and refuses rather than guesses on each:

* the member's own ``(M, K, N)``, read from its declared operands -- never recomputed from the entry
  that produced them, because two derivations of one quantity eventually disagree;
* the geometric class, from :mod:`merlin.dse_guidance.shape_taxonomy`, whose thresholds are fixed and
  documented so a reader can re-derive the label independently;
* whether the target's census (``conformance/<target>.yaml`` -> ``shape_geometry.required``, itself
  derived from real captures) contains that class, and with what share of the MAC mass.

⚠️ IT DOES NOT GATE. A member off the census is a finding, not an error: the census's mass-carrying
class is explicitly unreachable on this machine, so a corpus that covered every class would be
impossible to build and a generator that refused every off-census member would emit nothing. The
stamp exists so the gap is READABLE from the tracked capsule -- which is the difference between a
known hole and a silent one.

Nothing here names a target. The census is looked up by the ``target`` it is handed, exactly as the
conformance lattice does.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["census_classes", "declared_geometry", "stamp_for"]

#: Ops whose declared operands carry a contraction's (M, K, N). An op absent here is not
#: "not a contraction" -- it is one this reader cannot price, and it says so with a reason.
_CONTRACTION_OPS = ("matmul", "linear", "fused_matmul_bias", "resident_reuse")

#: Operand roles, in the ABI's own vocabulary: the activation supplies (M, K), the weight (K, N).
_ACTIVATION_ROLE = "input"
_WEIGHT_ROLE = "weight"


def _rank2(shapes: Mapping[str, Any], name: str) -> tuple[int, int] | None:
    shape = shapes.get(name)
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return None
    try:
        rows, cols = int(shape[0]), int(shape[1])
    except (TypeError, ValueError):
        return None
    return (rows, cols) if rows > 0 and cols > 0 else None


def declared_geometry(capsule: object) -> dict[str, Any]:
    """``(M, K, N)`` from a built capsule's own declared operands, or a refusal carrying its reason.

    Reads the capsule, not the profile entry that generated it. The entry states extents in tiles and
    the builder turns them into operands; reading the entry again here would be a second derivation of
    one quantity, and the two would drift the first time a builder padded or transposed.
    """
    if not isinstance(capsule, Mapping):
        return {"status": "refused", "reason": "the capsule is not a mapping"}
    op = str(((capsule.get("operation") or {}) if isinstance(capsule.get("operation"), Mapping)
              else {}).get("op") or "")
    if op not in _CONTRACTION_OPS:
        return {"status": "refused",
                "reason": (f"op {op!r} declares no contraction operands this reader can price; its "
                           f"geometry is not (M, K, N) and inventing one would misattribute it"),
                "op": op}
    rows = capsule.get("inputs")
    if not isinstance(rows, (list, tuple)):
        return {"status": "refused", "reason": "the capsule declares no inputs", "op": op}
    shapes: dict[str, Any] = {}
    roles: dict[str, str] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("name"):
            shapes[str(row["name"])] = row.get("shape")
            roles[str(row["name"])] = str(row.get("role") or "")
    weights = [n for n, r in roles.items() if r == _WEIGHT_ROLE]
    activations = [n for n, r in roles.items() if r == _ACTIVATION_ROLE]
    if len(weights) != 1 or not activations:
        return {"status": "refused",
                "reason": (f"expected exactly one {_WEIGHT_ROLE!r} operand and at least one "
                           f"{_ACTIVATION_ROLE!r}; found {len(weights)} and {len(activations)}"),
                "op": op}
    weight = _rank2(shapes, weights[0])
    # Every activation of a resident-weight member shares one (M, K); taking the first is not a choice
    # between them, and a member whose activations disagreed would be a different claim entirely.
    activation = _rank2(shapes, sorted(activations)[0])
    if weight is None or activation is None:
        return {"status": "refused",
                "reason": "an operand is not a positive rank-2 shape, so no (M, K, N) is declared",
                "op": op}
    K_w, N = weight
    M, K_a = activation
    if K_w != K_a:
        return {"status": "refused",
                "reason": (f"the activation reduces over {K_a} and the weight over {K_w}; the operands "
                           f"do not describe one contraction"), "op": op}
    return {"status": "derived", "op": op, "M": M, "K": K_w, "N": N,
            "basis": "the capsule's own declared operand shapes (activation MxK, weight KxN)"}


def census_classes(target: str) -> dict[str, Any]:
    """``{class: census entry}`` from the target's derived conformance spec, or ``{}`` when absent.

    An empty mapping is a real answer with a real consequence: without a census there is nothing to
    place a member against, and :func:`stamp_for` records that rather than pretending the member is
    representative.
    """
    from merlin.verify.lattice import load_spec

    try:
        spec = load_spec(target)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    geometry = spec.get("shape_geometry")
    required = (geometry or {}).get("required") if isinstance(geometry, Mapping) else None
    if not isinstance(required, (list, tuple)):
        return {}
    return {str(e["class"]): dict(e) for e in required
            if isinstance(e, Mapping) and e.get("class")}


def stamp_for(capsule: object, *, target: str) -> dict[str, Any] | None:
    """The block to embed on a member, or ``None`` when its geometry cannot be read at all.

    ``None`` and a block carrying ``in_census: false`` mean different things and must never collapse:
    the first says this member's shape is unreadable here, the second says it was read and real models
    do not present it.
    """
    from merlin.dse_guidance.shape_taxonomy import classify_geometry

    declared = declared_geometry(capsule)
    if declared.get("status") != "derived":
        return None
    M, K, N = int(declared["M"]), int(declared["K"]), int(declared["N"])
    label = classify_geometry(M, N, K)
    census = census_classes(target)
    block: dict[str, Any] = {
        "M": M, "K": K, "N": N, "out_elements": M * N,
        "geometry_class": label,
        "classifier": "merlin.dse_guidance.shape_taxonomy.classify_geometry",
        "basis": declared["basis"],
    }
    if not census:
        block["in_census"] = None
        block["census_note"] = ("this target has no derived shape_geometry census, so whether real "
                                "models present this class is unknown rather than false")
        return block
    entry = census.get(label)
    block["in_census"] = entry is not None
    block["census_classes"] = sorted(census)
    if entry is None:
        block["census_note"] = ("no captured model presents this geometric class, so improving this "
                                "member improves a shape the models do not contain")
        return block
    block["census_mac_fraction"] = entry.get("mac_fraction")
    block["census_out_elements"] = entry.get("out_elements")
    block["census_regions"] = entry.get("n_regions")
    unreachable = entry.get("unreachable")
    if unreachable:
        block["census_unreachable"] = str(unreachable)
        block["census_note"] = ("this class is in the census but the census itself records it as "
                                "unbuildable on this target, so membership is by aspect ratio only")
    return block
