"""Derive complete source operation demands and preserve exact linalg/interface correspondence."""

from __future__ import annotations

import hashlib
import json

from merlin.capture.artifacts import M2MUnavailable
from merlin.capture.carriers import (
    _OP_MARKER,
    _carrier_extents,
    _carrier_families,
    _carrier_formats,
    _carrier_input_formats,
    _carrier_reduces,
    _carrying_ops,
    _generic_form_linalg_summary,
    _tag_lines,
    _tensor_types,
)


def linalg_summary(linalg_mlir: str) -> dict:
    """Read a linalg-on-tensors module STRUCTURALLY (str tokenizer, NO regex): the ``@forward`` signature's
    operand + result tensor types and the set of ``prov.op`` / ``prov.family`` tags m2m stamps on each region.
    This is the structural view used to verify a mapped op's interface against its actual lowering."""
    if '"linalg.' in linalg_mlir or linalg_mlir.lstrip().startswith('"builtin.module"('):
        return _generic_form_linalg_summary(linalg_mlir)
    prov_ops = [c.split('"', 1)[0] for c in linalg_mlir.split('prov.op = "')[1:]]
    # Per TAG, not per family-occurrence: the two tags are not in one-to-one correspondence, and
    # splitting on them separately misaligns the lists wherever an operation states an op and no family.
    prov_families = _carrier_families(linalg_mlir)
    inputs: list[tuple[list[int], str]] = []
    output = None
    head = linalg_mlir.split("func.func @forward(", 1)
    if len(head) > 1:
        sig = head[1].split("{", 1)[0]  # the signature line, before the body brace
        argpart, _, rest = sig.partition(") ->")
        inputs = _tensor_types(argpart)
        outs = _tensor_types(rest)
        output = outs[0] if outs else None
    return {
        "prov_ops": prov_ops,
        "prov_families": prov_families,
        "carrier_region_ids": [
            line.split('prov.region_id = "', 1)[1].split('"', 1)[0] if 'prov.region_id = "' in line else None
            for line in _tag_lines(linalg_mlir)
        ],
        "inputs": inputs,
        "output": output,
        # WHICH OPERATION carries each tag, so a caller can tell a contraction from the shape ops
        # the same captured region also stamped.
        "carrying_ops": _carrying_ops(linalg_mlir),
        # …and whether that operation REDUCES, which is what separates a contraction from the im2col
        # gather the same region tags identically. Aligned with the lists above by construction.
        "carrier_reduces": _carrier_reduces(linalg_mlir),
        # …its contraction extents as (batch, M, K, N), for BOTH spellings a capture uses, and the
        # element format its own operands carry. Same construction, so same alignment.
        "carrier_extents": _carrier_extents(linalg_mlir),
        "carrier_formats": _carrier_formats(linalg_mlir),
        "carrier_input_formats": _carrier_input_formats(linalg_mlir),
    }


def model_op_demands(linalg_mlir: str, in_fmt: str, weight_fmt: str | None = None) -> list:
    """Per-op routing demands from a captured model's linalg (structural prov.op/prov.family read, NO regex).
    A contraction op (matmul/conv — ``prov.family == "contraction"``) carries a weight format AND its real
    (M, K, N) extents (so a whole-model matmul layer compiles at its true shape); normalization / elementwise
    / reduction ops are unary. Feeds ``routing.route_target`` so a whole model can be split across a target's
    compute units (matmul tiles -> the systolic mesh, the rest -> vector/scalar lanes).

    ``in_fmt``/``weight_fmt`` are the formats the compile was ASKED for and stay on every demand.
    Each demand ALSO carries ``elem_fmt``, the format its own operands are captured in; legality
    follows that observed format because the request alone cannot prove a conversion. A whole-model
    capture is routinely mixed -- see :attr:`routing.OpDemand.elem_fmt`.
    """
    from merlin.targetgen import semantic_families as sf
    from merlin.targetgen.routing import OpDemand

    summ = linalg_summary(linalg_mlir)
    ops, fams = summ["prov_ops"], summ["prov_families"]
    carriers = summ["carrying_ops"]
    reduces = summ["carrier_reduces"]
    extents = summ["carrier_extents"]
    formats = summ["carrier_formats"]
    input_formats = summ["carrier_input_formats"]
    region_ids = summ["carrier_region_ids"]
    wf = weight_fmt or in_fmt
    demands: list = []
    for i, op in enumerate(ops):
        fam = fams[i] if i < len(fams) else ""
        carrier = carriers[i] if i < len(carriers) else ""
        if op == "fill":  # linalg.fill init — not a routable compute op
            continue
        # ONE DEMAND PER OPERATION, NOT PER TAG. `prov.op` names the captured region an op came from, and
        # one region stamps its shape and metadata ops with the same tag. A `tensor.expand_shape` carrying
        # a convolution's tag is not a convolution: asking a compute unit to support one, and counting the
        # refusal onto the scalar lane, is how 702 of this capture's 1199 tags became "scalar demands".
        # The op still appears -- it costs real cycles -- but under ITS OWN name and with no weight format,
        # so it is judged as the movement it is.
        computes = carrier.startswith("linalg.")
        if not computes:
            demands.append(
                OpDemand(
                    op=carrier or op,
                    in_fmt=in_fmt,
                    weight_fmt=None,
                    site=carrier or op,
                    m=None,
                    k=None,
                    n=None,
                    elem_fmt=formats[i],
                    captured_input_formats=input_formats[i],
                    region_id=region_ids[i],
                    carrier_op=carrier or None,
                    form=sf.operation_form(carrier or op, carrier_op=carrier),
                )
            )
            continue
        # THE REGION'S family, corrected by THIS operation's own iteration space. `prov.family` is
        # stamped on every op a captured region produced, so an im2col GATHER carries `contraction`
        # because the convolution it belongs to is one -- and unlike the region's `tensor.*` ops it
        # survives the `computes` test above, being a `linalg.generic`. An op with no reduction
        # contracts nothing, whatever its region is called. Downgraded to UNKNOWN rather than renamed:
        # unknown never widens, so such an op is matched by NAME alone, exactly as it would be if the
        # capture had never stated a family for it.
        fam = sf.carrier_provenance_family(fam, carrier_op=carrier, reduces=reduces[i])
        # ``prov.family`` is a capture vocabulary (layout, cast, elementwise, ...),
        # while a compute-unit contract speaks canonical semantic families.  Only
        # normalize a present, operation-valid tag; the deliberately erased
        # contraction tag above must not be recovered by an op-name guess.
        canonical_family = sf.from_prov(fam, op) if fam else None
        # THE OPERATION'S OWN EXTENTS, not the next unclaimed named matmul's. This used to walk a
        # separate list of `linalg.matmul` text occurrences with its own counter, on the assumption that
        # the demands tagged `prov.op = "matmul"` are exactly those operations. They are not: a
        # contraction may be spelled `linalg.generic`, and a named matmul may be tagged with the
        # convolution it came from, so 19 capsules here disagree on the count and every demand after the
        # first disagreement carried another layer's shape (see `_tag_lines`).
        #
        # Selection is by FAMILY, for the same reason `tile_builder_op` matches by family: extents are
        # what make a contraction synthesizable as a tile at all, and withholding them from the ones
        # spelled `convolution_im2col_matmul` / `int_matmul` / `batch_matmul` left real contractions
        # counted `n_unsynthesizable` for a spelling.
        extent = extents[i] if canonical_family == "contraction" else None
        batch, m, k, n = extent if extent is not None else (None, None, None, None)
        demands.append(
            OpDemand(
                op=op,
                in_fmt=in_fmt,
                weight_fmt=(wf if canonical_family == "contraction" else None),
                site=op,
                m=m,
                k=k,
                n=n,
                # The capture states the family; routing asks the unit whether it does
                # this KIND of work, instead of whether it recognises this SPELLING.
                family=canonical_family,
                source_family=fam or None,
                # The format THIS op is captured in, beside the one the compile declared. Unknown
                # (unreadable / mixed / not a registry format) stays None and never widens.
                elem_fmt=formats[i],
                captured_input_formats=input_formats[i],
                batch=batch,
                region_id=region_ids[i],
                carrier_op=carrier or None,
                form=sf.operation_form(op, carrier_op=carrier),
            )
        )
    return demands


class ModelDemandIncomplete(ValueError):
    """A captured contraction was not represented exactly once by the routing demands."""


def model_op_demands_checked(
    linalg_mlir: str, in_fmt: str, weight_fmt: str | None = None, *, source_kind: str = "captured_provenance_v1"
) -> list:
    """Verify complete ordered contraction identity and declared source formats."""
    from merlin.targetgen.source_demand_inventory import model_op_demands_checked as checked

    return checked(linalg_mlir, in_fmt, weight_fmt, source_kind=source_kind)


def linalg_to_iface(linalg_mlir: str, entry: dict, binding):
    """Derive-and-verify a mapped op's ``merlin_iface`` interface from the CAPTURED linalg rather than
    assuming the PyTorch capture matches the profile entry. Structurally confirm the lowering actually
    contains the op family the interface claims AND that its operand shapes are the ones the builder
    declares; fail closed (``M2MUnavailable``) on any mismatch — never emit an interface the lowering does
    not support. Returns ``(cap, mlir)`` from the merlin_iface builder (byte-identical to the handwritten
    form when they agree)."""
    from merlin.targetgen import corpus_spec as CS

    op = entry.get("op", "matmul")
    summ = linalg_summary(linalg_mlir)
    markers = _OP_MARKER.get(op)
    if markers is not None and not (set(markers) & set(summ["prov_ops"])) and op not in summ["prov_ops"]:
        raise M2MUnavailable(
            f"captured linalg for {entry['name']!r} lacks any {op!r} marker op {list(markers)} "
            f"(prov.op={summ['prov_ops']}) — refusing to emit a merlin_iface interface the lowering "
            f"does not support"
        )
    cap, mlir = CS.build(entry, binding)
    # every operand shape the builder declares must appear among the captured linalg's operand shapes.
    lin_shapes = [tuple(sh) for sh, _ in summ["inputs"]]
    for inp in cap.get("inputs", []):
        want = tuple(inp["shape"])
        if lin_shapes and want not in lin_shapes:
            raise M2MUnavailable(
                f"interface operand {inp['name']!r} shape {want} for {entry['name']!r} not found in the "
                f"captured linalg operands {lin_shapes} — interface/lowering shape mismatch"
            )
    return cap, mlir


def verify_exact_application_int_mm(linalg_mlir: str, match: dict) -> dict:
    """Require the emitted PyTorch slice to compute the selected application's exact integer op.

    The source capture's full inventory digest binds the expected row. This check independently parses
    the newly emitted slice and compares its ordered ABI, indexing maps, iterator kinds, arithmetic
    body and geometry. It deliberately does not claim that a whole model compiles or executes.
    """
    from merlin.common import mlir_query as mq
    from merlin.targetgen.application_inventory import (
        exact_int_mm_geometry,
        int_mm_source_is_qualified,
        operation_structure,
    )

    if match.get("status") != "candidate_unverified" or not int_mm_source_is_qualified(match):
        raise M2MUnavailable("exact application slice lacks a digest-bound W8A8 source declaration")
    if not isinstance(match.get("full_inventory_sha256"), str) or len(match["full_inventory_sha256"]) != 64:
        raise M2MUnavailable("exact application slice lacks a full inventory digest")
    expected = match.get("expected_signature")
    if not isinstance(expected, dict) or exact_int_mm_geometry(expected, require_quant_origin=False) is None:
        raise M2MUnavailable("selected application integer-matmul signature is incomplete or unsupported")
    try:
        module = mq.parse(linalg_mlir)
        observed = []
        for op in mq.walk(module, "linalg.generic"):
            if mq.attr_str(op, "prov.aten") != "aten._int_mm.default":
                continue
            structure = operation_structure(op)
            lhs, rhs, _out = (
                structure["ordered_operand_types"] if len(structure["ordered_operand_types"]) == 3 else ({}, {}, {})
            )
            m, k = (lhs.get("shape") or [None, None])[:2]
            n = (rhs.get("shape") or [None, None])[-1]
            row = {
                **structure,
                "operation": "aten._int_mm.default",
                "mlir_operation": mq.op_name(op),
                "frontend_op": mq.attr_str(op, "prov.aten"),
                "provenance_op": mq.attr_str(op, "prov.op"),
                "semantic_family": "contraction",
                "operand_format": "int8" if lhs.get("dtype") == "i8" else None,
                "accumulator_dtypes": sorted(
                    {
                        mq.type_shape_dtype(value.type)[1]
                        for child in mq.walk(op)
                        if mq.op_name(child) == "arith.addi"
                        for value in child.results
                    }
                ),
                "result_shapes": [item["shape"] for item in structure["ordered_result_types"]],
                "contraction_shape": {"M": m, "K": k, "N": n, "rank": len(structure["iterator_types"] or ())},
            }
            if exact_int_mm_geometry(row, require_quant_origin=False) is None:
                raise M2MUnavailable("emitted aten._int_mm has unsupported rank, dtype, maps, body or shape")
            observed.append(row)
    except M2MUnavailable:
        raise
    except Exception as exc:
        raise M2MUnavailable(f"cannot inspect emitted integer-matmul MLIR: {exc}") from exc
    if len(observed) != 1 or observed[0] != expected:
        raise M2MUnavailable(
            "emitted integer-matmul MLIR does not match the selected application signature "
            f"(found {len(observed)} candidates)"
        )
    return {
        "status": "verified_capture_match",
        "full_inventory_sha256": match["full_inventory_sha256"],
        "source_quantization": match["source_quantization"],
        "signature_sha256": hashlib.sha256(
            json.dumps(observed[0], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "emitted_mlir_sha256": hashlib.sha256(linalg_mlir.encode()).hexdigest(),
        "sources": match.get("sources") or [],
        "scope": "isolated PyTorch integer operation and emitted linalg; whole-model compile unverified",
    }
