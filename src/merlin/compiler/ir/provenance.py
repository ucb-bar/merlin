"""Carry source provenance through actual IR replacements.

Preserve source ordering, numerical/effect obligations and explicit selection.
"""

from __future__ import annotations


def carry_provenance(new_op, src_op, applied: str) -> None:
    """Carry model-layer provenance from a rewritten op onto its replacement + record WHAT ran.

    Three threads, so a lowering that rewrites an op does not sever it from its source layer:
      1. copy every ``prov.*`` attribute (region_id/fqn/op/... — the join key the compare + slicer use);
      2. propagate the MLIR ``Location`` (the in-IR "which original file/line" carrier — a no-op until
         model2MLIR emits locations, but then it survives passes for free);
      3. APPEND ``applied`` to an ordered ``prov.transforms`` breadcrumb (comma-joined) so the trace can
         see the transformation sequence a region actually went through ("what was actually applied").
    """
    from xdsl.dialects.builtin import StringAttr

    chain: list[str] = []
    for key, val in src_op.attributes.items():
        if key.startswith("prov."):
            new_op.attributes[key] = val
            if key == "prov.transforms" and isinstance(val, StringAttr):
                chain = [t for t in val.data.split(",") if t]
    chain.append(applied)
    new_op.attributes["prov.transforms"] = StringAttr(",".join(chain))
    loc = getattr(src_op, "location", None)
    if loc is not None:
        try:
            new_op.location = loc
        except Exception:  # noqa: BLE001 - location is best-effort in-IR provenance, never load-bearing
            pass
