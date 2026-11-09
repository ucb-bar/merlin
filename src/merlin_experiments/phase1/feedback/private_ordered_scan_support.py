"""Withheld source and linked-build witness for an ordered f32 CPU prefix scan.

This proves a closed prepared-MLIR algorithm and binds its exact reviewed host
admission, not frontend or numerical equivalence of the linked image. The width is a selected
compiler premise only after the existing post-build index observation joins it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin_experiments.phase1.feedback import private_control_support as control

FIELD = "ordered_f32_scan_support"
PENDING = "source_ordered_f32_scan_pending_build"
LINKED = "source_ordered_f32_scan_linked"
SCOPE = (
    "prepared ordered f32/f64 scan source, reviewed root admission, and linked-byte identity; "
    "no frontend or numerical equivalence"
)
_TOP = {
    "status",
    "scope",
    "raw_source_sha256",
    "normalized_source_sha256",
    "capture_receipt_sha256",
    "n_source_operations",
    "selected_index_observation",
    "count",
    "occurrences",
    "occurrences_sha256",
    "source_verified",
    "admission_verified",
    "admissions",
    "admissions_sha256",
    "linked_build",
}
_BUILD = {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
_ADMISSION = {
    "root_ordinal",
    "all_ordinals",
    "software_declaration",
    "host_profile",
    "host_declaration",
    "host_package_sha256",
    "host_spec_sha256",
    "host_dtype_strategy",
    "software_record_sha256",
    "capability_record_sha256",
    "semantic_signature",
    "hardware_refusals",
}
_PROPERTIES = {
    "tensor.collapse_shape": {"reassociation"},
    "arith.constant": {"value"},
    "tensor.empty": set(),
    "scf.for": set(),
    "tensor.extract": set(),
    "arith.extf": set(),
    "arith.divui": set(),
    "arith.remui": set(),
    "arith.cmpi": {"predicate"},
    "scf.if": set(),
    "scf.yield": set(),
    "arith.subi": {"overflowFlags"},
    "arith.addf": {"fastmath"},
    "arith.truncf": set(),
    "tensor.insert": set(),
    "tensor.expand_shape": {"reassociation", "static_output_shape"},
}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(f"ordered f32 scan: {reason}")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _semantic_signature(item: Mapping[str, Any]) -> dict[str, Any]:
    """The source-proved tensor contract, not the flattened scf.for ABI."""
    return {
        "family": "reduction",
        "operand_dtype": item["input_type"],
        "readout_dtype": item["output_type"],
        "accum_dtype": item["carry_type"],
        "ordered_operand_dtypes": [item["input_type"]],
        "ordered_result_dtypes": [item["output_type"]],
        "compute_dtypes": [item["carry_type"]],
        "rank": len(item["input_shape"]),
        "dimensions": {},
    }


def _shape(value, elem: str, *, rank: int | None = None) -> tuple[int, ...]:
    from xdsl.dialects.builtin import NoneAttr, TensorType

    typ = value.type
    _need(isinstance(typ, TensorType) and str(typ.element_type) == elem, "tensor element type differs")
    _need(isinstance(typ.encoding, NoneAttr), "tensor encoding is not default")
    dims = tuple(typ.get_shape())
    _need(rank is None or len(dims) == rank, "tensor rank differs")
    _need(bool(dims) and all(type(n) is int and n > 0 for n in dims), "scan needs positive static extents")
    return dims


def _clean(op, name: str) -> None:
    _need(
        op.name == name and not op.successors and set(op.properties) == _PROPERTIES[name],
        f"expected closed registered {name}",
    )
    _need(all(key.startswith("prov.") for key in op.attributes), f"{name} has non-provenance attributes")
    if name not in {"scf.for", "scf.if"}:
        _need(not op.regions, f"{name} has an unexpected region")
    for key in ("fastmath", "overflowFlags"):
        flag = op.properties.get(key)
        _need(flag is None or (hasattr(flag, "data") and not flag.data), f"{name} has numeric flags")


def _integer_constant(op, value: int) -> None:
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import IndexType, IntegerAttr

    _need(type(op) is ConstantOp, "index bound is not a registered constant")
    _clean(op, "arith.constant")
    attr = op.value
    _need(
        len(op.results) == 1
        and isinstance(op.result.type, IndexType)
        and isinstance(attr, IntegerAttr)
        and isinstance(attr.type, IndexType)
        and type(attr.value.data) is int
        and attr.value.data == value,
        "index constant differs from the closed scan geometry",
    )


def _float_zero(op) -> None:
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import Float64Type, FloatAttr

    _need(type(op) is ConstantOp, "scan seed is not a registered constant")
    _clean(op, "arith.constant")
    attr = op.value
    _need(
        len(op.results) == 1
        and isinstance(op.result.type, Float64Type)
        and isinstance(attr, FloatAttr)
        and isinstance(attr.type, Float64Type)
        and attr.value.data == 0.0
        and math.copysign(1.0, attr.value.data) == 1.0,
        "scan seed is not positive f64 zero",
    )


def _index_constant_value(op) -> int:
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import IndexType, IntegerAttr

    _need(type(op) is ConstantOp, "lane bound is not a registered constant")
    attr = op.value
    _need(
        isinstance(op.result.type, IndexType)
        and isinstance(attr, IntegerAttr)
        and isinstance(attr.type, IndexType)
        and type(attr.value.data) is int,
        "lane bound is not an exact index integer",
    )
    return attr.value.data


def _uses(value, expected: tuple[tuple[Any, int], ...]) -> None:
    actual = sorted((id(use.operation), use.index) for use in value.uses)
    want = sorted((id(op), index) for op, index in expected)
    _need(actual == want, "scan SSA value escapes or has an unproved consumer")


def _ancestry(op, root) -> None:
    for key in ("prov.source_node_ids", "prov.origin_node_ids"):
        selected = root.attributes.get(key)
        if selected is not None:
            _need(op.attributes.get(key) == selected, "scan operation ancestry differs from its root")


def _reassociation(op, rank: int) -> bool:
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

    groups = getattr(op, "reassociation", None)
    return bool(
        isinstance(groups, ArrayAttr)
        and len(groups.data) == 1
        and isinstance(groups.data[0], ArrayAttr)
        and tuple(value.value.data if isinstance(value, IntegerAttr) else None for value in groups.data[0].data)
        == tuple(range(rank))
    )


def _recognize(root, parsed: tuple[Any, ...], ordinal: int, index_bits: int) -> dict:
    from xdsl.dialects import arith, scf, tensor
    from xdsl.dialects.builtin import IndexType

    _need(
        type(root) is scf.ForOp and mq.attr_str(root, "prov.aten") == "aten.cumsum.default",
        "root is not a captured ordered cumsum",
    )
    _clean(root, "scf.for")
    _need(type(index_bits) is int and 2 <= index_bits <= 128, "selected index width is malformed")
    region = mq.attr_str(root, "prov.region_id")
    _need(
        isinstance(region, str) and region and mq.attr_str(root, "prov.family") == "scan",
        "scan region provenance is absent",
    )
    top = [(i, op) for i, op in enumerate(parsed) if mq.attr_str(op, "prov.region_id") == region]
    _need(
        all(mq.attr_str(op, "prov.aten") == "aten.cumsum.default" for _, op in top),
        "scan region has a mixed frontend owner",
    )
    _need(sum(op is root for _, op in top) == 1, "scan region has no unique ordered loop")
    for _, op in top:
        _ancestry(op, root)

    _need(len(root.regions) == 1 and len(root.body.blocks) == 1, "scan loop has an unproved body")
    args = tuple(root.body.block.args)
    _need(len(root.iter_args) == 2 and len(root.results) == 2 and len(args) == 3, "scan loop carries extra state")
    _need(isinstance(args[0].type, IndexType), "scan induction is not index typed")
    out_shape = _shape(root.results[0], "f32", rank=1)
    acc_shape = _shape(root.results[1], "f64", rank=1)
    _need(out_shape == acc_shape, "output and carry geometries differ")
    total = out_shape[0]
    _need(
        total <= min((1 << (index_bits - 1)) - 1, (1 << 63) - 1),
        "loop bound exceeds source or selected signed index width",
    )
    _need(
        args[1].type == root.results[0].type and args[2].type == root.results[1].type, "loop block state types differ"
    )

    loop_at = next(i for i, op in top if op is root)
    before = [(i, op) for i, op in top if i < loop_at]
    after = [(i, op) for i, op in top if i > loop_at]
    _need(len(before) in {8, 9} and len(after) in {0, 1}, "scan predecessor/successor roster is not closed")
    collapsed = before[0][1] if len(before) == 9 else None
    if collapsed is not None:
        _need(type(collapsed) is tensor.CollapseShapeOp, "scan input is not a canonical flatten")
        _clean(collapsed, "tensor.collapse_shape")
        original_shape = _shape(collapsed.src, "f32")
        _need(len(original_shape) > 1 and math.prod(original_shape) == total, "flattened input extent differs")
        _need(_shape(collapsed.result, "f32", rank=1) == (total,), "flattened input type differs")
        _need(_reassociation(collapsed, len(original_shape)), "flattened input reassociation differs")
        flat = collapsed.result
    else:
        original_shape = (total,)
        flat = None
    c0, c1, cend, cstride, cextent, zero, out_empty, acc_empty = [op for _, op in before[-8:]]
    _integer_constant(c0, 0)
    _integer_constant(c1, 1)
    _integer_constant(cend, total)
    _float_zero(zero)
    for empty, elem in ((out_empty, "f32"), (acc_empty, "f64")):
        _need(type(empty) is tensor.EmptyOp, "scan initial tensor is not empty")
        _clean(empty, "tensor.empty")
        _need(
            not empty.operands and _shape(empty.results[0], elem, rank=1) == (total,),
            "scan initial tensor is not static",
        )
    _need(
        tuple(root.operands) == (c0.result, cend.result, c1.result, out_empty.results[0], acc_empty.results[0]),
        "loop bounds or iter_args differ",
    )
    body = tuple(root.body.block.ops)
    names = (
        "tensor.extract",
        "arith.extf",
        "arith.divui",
        "arith.remui",
        "arith.cmpi",
        "scf.if",
        "arith.addf",
        "arith.truncf",
        "tensor.insert",
        "tensor.insert",
        "scf.yield",
    )
    _need(tuple(op.name for op in body) == names, "ordered scan scalar body has extra or missing operations")
    extract, extend, div, rem, cmp, branch, add, truncate, out_insert, acc_insert, yielded = body
    for op, name in zip(body, names, strict=True):
        _clean(op, name)
        _ancestry(op, root)
    if flat is None:
        flat = extract.tensor
        _need(_shape(flat, "f32", rank=1) == (total,), "rank-one input differs")
    _need(
        type(extract) is tensor.ExtractOp
        and extract.tensor is flat
        and tuple(extract.indices) == (args[0],)
        and str(extract.result.type) == "f32",
        "current source element is not the induction element",
    )
    _need(
        type(extend) is arith.ExtFOp
        and tuple(extend.operands) == (extract.result,)
        and str(extend.result.type) == "f64",
        "source element is not widened before addition",
    )
    _need(type(div) is arith.DivUIOp and tuple(div.operands) == (args[0], cstride.result), "lane quotient differs")
    _need(type(rem) is arith.RemUIOp and tuple(rem.operands) == (div.result, cextent.result), "lane remainder differs")
    _need(
        type(cmp) is arith.CmpiOp and tuple(cmp.operands) == (rem.result, c0.result) and cmp.predicate.value.data == 0,
        "lane reset predicate is not equality with zero",
    )
    _need(
        type(branch) is scf.IfOp
        and tuple(branch.operands) == (cmp.result,)
        and len(branch.results) == 1
        and str(branch.results[0].type) == "f64",
        "lane reset branch differs",
    )
    _need(
        len(branch.true_region.blocks) == 1 and len(branch.false_region.blocks) == 1, "lane reset lacks both branches"
    )
    _need(
        not branch.true_region.block.args and not branch.false_region.block.args,
        "lane reset branches carry unproved block arguments",
    )
    true_body = tuple(branch.true_region.block.ops)
    false_body = tuple(branch.false_region.block.ops)
    _need(
        len(true_body) == 1 and type(true_body[0]) is scf.YieldOp and tuple(true_body[0].operands) == (zero.result,),
        "first lane does not reset to positive zero",
    )
    _need(
        tuple(op.name for op in false_body) == ("arith.subi", "tensor.extract", "scf.yield"),
        "nonfirst lane has extra effects",
    )
    subtract, prior, prior_yield = false_body
    for op in (*true_body, *false_body):
        _clean(op, op.name)
        _ancestry(op, root)
    _need(
        type(subtract) is arith.SubiOp and tuple(subtract.operands) == (args[0], cstride.result), "prior offset differs"
    )
    _need(
        type(prior) is tensor.ExtractOp
        and prior.tensor is args[2]
        and tuple(prior.indices) == (subtract.result,)
        and str(prior.result.type) == "f64",
        "prior carry is not read from the preceding lane",
    )
    _need(
        type(prior_yield) is scf.YieldOp and tuple(prior_yield.operands) == (prior.result,),
        "prior carry does not feed reset branch",
    )
    _need(
        type(add) is arith.AddfOp
        and tuple(add.operands) == (branch.results[0], extend.result)
        and str(add.result.type) == "f64",
        "ordered wide addition differs",
    )
    _need(
        type(truncate) is arith.TruncFOp
        and tuple(truncate.operands) == (add.result,)
        and str(truncate.result.type) == "f32",
        "wide prefix is not rounded for output",
    )
    _need(
        type(out_insert) is tensor.InsertOp and tuple(out_insert.operands) == (truncate.result, args[1], args[0]),
        "output insertion differs",
    )
    _need(
        type(acc_insert) is tensor.InsertOp and tuple(acc_insert.operands) == (add.result, args[2], args[0]),
        "wide carry insertion differs",
    )
    _need(
        type(yielded) is scf.YieldOp and tuple(yielded.operands) == (out_insert.result, acc_insert.result),
        "loop output/carry order differs",
    )
    _need(not root.results[1].uses, "wide carry escapes the ordered loop")

    # The listed operations alone do not prove closure if an SSA result has an
    # additional consumer elsewhere in the module. Check every internal edge.
    for value, consumers in (
        (c0.result, ((root, 0), (cmp, 1))),
        (c1.result, ((root, 2),)),
        (cend.result, ((root, 1),)),
        (cstride.result, ((div, 1), (subtract, 1))),
        (cextent.result, ((rem, 1),)),
        (zero.result, ((true_body[0], 0),)),
        (out_empty.results[0], ((root, 3),)),
        (acc_empty.results[0], ((root, 4),)),
        (args[0], ((extract, 1), (div, 0), (subtract, 0), (out_insert, 2), (acc_insert, 2))),
        (args[1], ((out_insert, 1),)),
        (args[2], ((prior, 0), (acc_insert, 1))),
        (extract.result, ((extend, 0),)),
        (extend.result, ((add, 1),)),
        (div.result, ((rem, 0),)),
        (rem.result, ((cmp, 0),)),
        (cmp.result, ((branch, 0),)),
        (branch.results[0], ((add, 0),)),
        (subtract.result, ((prior, 1),)),
        (prior.result, ((prior_yield, 0),)),
        (add.result, ((truncate, 0), (acc_insert, 0))),
        (truncate.result, ((out_insert, 0),)),
        (out_insert.result, ((yielded, 0),)),
        (acc_insert.result, ((yielded, 1),)),
    ):
        _uses(value, consumers)

    stride_value, extent_value = _index_constant_value(cstride), _index_constant_value(cextent)
    axes = [
        axis
        for axis in range(len(original_shape))
        if math.prod(original_shape[axis + 1 :]) == stride_value and original_shape[axis] == extent_value
    ]
    _need(bool(axes), "lane stride/extent do not describe an input axis")
    stride = math.prod(original_shape[axes[0] + 1 :])
    extent = original_shape[axes[0]]
    _need(
        stride <= min((1 << (index_bits - 1)) - 1, (1 << 63) - 1)
        and extent <= min((1 << (index_bits - 1)) - 1, (1 << 63) - 1),
        "lane constants exceed source or selected index width",
    )
    _integer_constant(cstride, stride)
    _integer_constant(cextent, extent)
    if collapsed is not None:
        _uses(collapsed.result, ((extract, 0),))
    _uses(root.results[1], ())
    if len(original_shape) > 1:
        _need(len(after) == 1 and type(after[0][1]) is tensor.ExpandShapeOp, "multidimensional output is not restored")
        expanded = after[0][1]
        _clean(expanded, "tensor.expand_shape")
        _need(
            expanded.src is root.results[0] and _shape(expanded.result, "f32") == original_shape,
            "restored output differs",
        )
        _need(
            not expanded.dynamic_output_shape and tuple(expanded.static_output_shape.get_values()) == original_shape,
            "restored output shape operands or declared extents differ",
        )
        _need(_reassociation(expanded, len(original_shape)), "restored output reassociation differs")
        _uses(root.results[0], ((expanded, 0),))
    else:
        _need(not after, "rank-one scan has an extra successor")

    nested = {id(op) for op in (*body, *true_body, *false_body)}
    ordinals = [i for i, op in enumerate(parsed) if op is root or id(op) in nested or (i, op) in top]
    _need(len(ordinals) == len(set(ordinals)), "source ordinal roster repeats an operation")
    return {
        "root_ordinal": ordinal,
        "all_ordinals": ordinals,
        "input_shape": list(original_shape),
        "output_shape": list(original_shape),
        "axis_candidates": axes,
        "total": total,
        "stride": stride,
        "axis_extent": extent,
        "input_type": "f32",
        "output_type": "f32",
        "carry_type": "f64",
        "index_bits_premise": index_bits,
        "region_id": region,
    }


def begin(
    source: Path, raw_sha: str, normalized_sha: str, receipt_sha: str, selected: Mapping[str, Any] | None
) -> dict:
    """Reparse complete normalized source and prove every tagged ordered scan root."""
    _need(all(is_sha256(value) for value in (raw_sha, normalized_sha, receipt_sha)), "source identity is malformed")
    raw = source.read_bytes()
    _need(sha256(raw).hexdigest() == raw_sha, "raw source bytes changed")
    try:
        text, normalization = normalize_capture_mlir(raw.decode("utf-8"))
        module = mq.parse(text)
        module.verify()
    except Exception as exc:  # noqa: BLE001 - parser/verifier are a fail-closed trust boundary
        raise ValueError(f"ordered f32 scan: complete source failed verification: {exc}") from exc
    _need(
        normalization["input_sha256"] == raw_sha
        and sha256(text.encode()).hexdigest() == normalized_sha == normalization["output_sha256"],
        "normalized source identity differs",
    )
    parsed = tuple(mq.walk(module))
    # A provenance-tagged f64 carry initialization cannot become an empty
    # witness merely because the loop itself lost its frontend/region tag.
    carry_regions = {
        mq.attr_str(op, "prov.region_id")
        for op in parsed
        if mq.attr_str(op, "prov.aten") == "aten.cumsum.default"
        and mq.op_name(op) == "tensor.empty"
        and any(str(result.type).endswith("xf64>") for result in op.results)
    }
    for region in carry_regions:
        _need(
            isinstance(region, str)
            and sum(
                mq.op_name(op) == "scf.for"
                and mq.attr_str(op, "prov.region_id") == region
                and mq.attr_str(op, "prov.aten") == "aten.cumsum.default"
                for op in parsed
            )
            == 1,
            "tagged wide scan carry has no unique ordered loop",
        )
    roots = [
        (i, op)
        for i, op in enumerate(parsed)
        if mq.op_name(op) == "scf.for" and mq.attr_str(op, "prov.aten") == "aten.cumsum.default"
    ]
    if roots:
        _need(
            isinstance(selected, Mapping) and selected.get("schema") == "merlin.selected-index-lowering.v1",
            "selected index observation is missing",
        )
        bits = selected.get("index_bits")
        _need(type(bits) is int and 2 <= bits <= 128, "selected index width is malformed")
    else:
        bits = None
    occurrences = [_recognize(op, parsed, i, bits) for i, op in roots]
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha,
        "normalized_source_sha256": normalized_sha,
        "capture_receipt_sha256": receipt_sha,
        "n_source_operations": len(parsed),
        "selected_index_observation": dict(selected) if selected is not None else None,
        "count": len(occurrences),
        "occurrences": occurrences,
        "occurrences_sha256": _digest(occurrences),
        "source_verified": True,
        "admission_verified": False,
        "admissions": [],
        "admissions_sha256": _digest([]),
    }


def admit(
    witness: dict,
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
    software: Mapping[str, Any],
    capability: Mapping[str, Any],
    cap_map: Mapping[str, Any],
    host: Mapping[str, Any],
) -> set[int]:
    """Admit only reviewed algorithm roots, then account for their closed body.

    The inventory marks an scf.for as support, so its ordinary hardware screen
    is not applicable. Here the independent oracle sees the proved *tensor*
    f32 reduction, not the loop's index operands. Unknown or eligible hardware
    is never converted into host support by this accounting step.
    """
    from merlin.common import mlir_query as mq
    from merlin.targetgen import operation_accounting as oa
    from merlin.targetgen import semantic_families as sf
    from merlin.targetgen.eligibility import (
        RegionDescriptor,
        is_eligible,
        providers_from_contract,
        undetermined_families_from_contract,
    )

    _need(witness.get("status") == PENDING and witness.get("source_verified") is True, "source proof is absent")
    _need(witness.get("admission_verified") is False, "source admission was already recorded")
    occ = witness["occurrences"]
    roots = {item["root_ordinal"] for item in occ}
    all_ordinals = {ordinal for item in occ for ordinal in item["all_ordinals"]}
    _need(len(parsed) == witness["n_source_operations"], "parsed source roster differs")
    _need(
        len(roots) == len(occ) and len(all_ordinals) == sum(len(item["all_ordinals"]) for item in occ),
        "ordered scan source proofs overlap",
    )
    admissions = []
    for item in occ:
        root = item["root_ordinal"]
        row = source_rows.get(root)
        _need(
            isinstance(row, Mapping)
            and row.get("mlir_operation") == "scf.for"
            and row.get("operation") == "aten.cumsum.default"
            and row.get("disposition") == "support_required"
            and root in (row.get("ordinals") or [])
            and set(row.get("ordinals") or []) <= roots,
            "ordered scan root inventory is absent or mixed with unproved loops",
        )
        family = sf.from_prov(mq.attr_str(parsed[root], "prov.family"), row["operation"])
        _need(family == "reduction", "scan family has no canonical reduction meaning")
        _need(item["input_type"] == item["output_type"] == "f32", "scan tensor precision differs")
        refusals = []
        for rank in sorted({1, len(item["input_shape"])}):
            verdict = is_eligible(
                RegionDescriptor(
                    op=row["operation"],
                    family=family,
                    in_dtype=item["input_type"],
                    out_dtype=item["output_type"],
                    rank=rank,
                ),
                dict(cap_map),
                undetermined=undetermined_families_from_contract(dict(capability)),
                providers=providers_from_contract(dict(capability)),
            )
            _need(
                not verdict.eligible
                and not verdict.undetermined
                and verdict.refusal in {"undeclared_family", "input_dtype", "result_dtype"},
                "scan tensor computation is hardware-eligible or eligibility is unresolved",
            )
            refusals.append({"rank": rank, "family": verdict.family, "refusal": verdict.refusal})
        screened = oa.admit_operation_row(
            dict(row),
            software_spec=dict(software),
            capability_contract=dict(capability),
            capability_map=dict(cap_map),
            host_capabilities=dict(host),
            observed=_semantic_signature(item),
            source_operations=tuple(parsed[i] for i in row["ordinals"]),
        )
        sw = [
            decision
            for decision in screened["software_admissions"]
            if decision["status"] == "admitted"
            and decision.get("review_status") == "reviewed"
            and decision.get("placement") == "host"
        ]
        profiles = [
            profile
            for profile in screened["host_admission"]["profiles"]
            if profile["status"] == "admitted" and profile.get("reviewed") is True
        ]
        # The exported accelerator software specification need not duplicate
        # pure-host operations. The exact selected host profile owns this
        # admission; any independently declared host software rule must still
        # be unique and reviewed, never a second source of invented capability.
        _need(software.get("status") == "reviewed", "selected software specification is unreviewed")
        _need(len(sw) <= 1 and len(profiles) == len(host) == 1, "scan has no unique reviewed host admission")
        profile = profiles[0]
        decisions = [
            decision
            for decision in profile["decisions"]
            if decision["status"] == "admitted" and decision.get("review_status") == "reviewed"
        ]
        _need(len(decisions) == 1, "scan host declaration is ambiguous or unreviewed")
        sw_id, host_id = (sw[0]["declaration"] if sw else None), decisions[0]["declaration"]
        sw_rules = [rule for rule in software["operations"] if rule.get("id") == sw_id]
        host_rules = [
            rule for rule in host[profile["profile"]]["capability_spec"]["operations"] if rule.get("id") == host_id
        ]
        _need(
            len(host_rules) == 1
            and (sw_id is None or len(sw_rules) == 1 and row["operation"] in sw_rules[0].get("ops", []))
            and row["operation"] in host_rules[0].get("ops", []),
            "scan requires exact reviewed root declarations, not family-only admission",
        )
        _need(
            is_sha256(profile.get("package_sha256")) and is_sha256(profile.get("capability_spec_sha256")),
            "selected host profile has no exact bytes",
        )
        admissions.append(
            {
                "root_ordinal": root,
                "all_ordinals": item["all_ordinals"],
                "software_declaration": sw_id,
                "host_profile": profile["profile"],
                "host_declaration": host_id,
                "host_package_sha256": profile["package_sha256"],
                "host_spec_sha256": profile["capability_spec_sha256"],
                "host_dtype_strategy": profile["dtype_strategy"],
                "software_record_sha256": _digest(software),
                "capability_record_sha256": _digest(capability),
                "semantic_signature": _semantic_signature(item),
                "hardware_refusals": refusals,
            }
        )
    witness["admissions"] = admissions
    witness["admissions_sha256"] = _digest(admissions)
    witness["admission_verified"] = True
    return all_ordinals


def link(source: Mapping[str, Any], actual_index: Mapping[str, Any], linked_build: Mapping[str, Any]) -> None:
    witness = source.get(FIELD)
    _need(
        isinstance(witness, dict)
        and witness.get("status") == PENDING
        and witness.get("source_verified") is True
        and witness.get("admission_verified") is True,
        "source witness or reviewed admission is absent or not pending",
    )
    _need(
        witness.get("selected_index_observation") == source.get("selected_index_observation")
        and control._record_matches_selected(actual_index, witness.get("selected_index_observation")),
        "actual linked index lowering differs from selected premise",
    )  # noqa: PLC2701
    _need(_valid_admissions(witness), "reviewed source algorithm admission is incomplete")
    _need(
        isinstance(linked_build, Mapping)
        and set(linked_build) == _BUILD
        and all(is_sha256(linked_build[key]) for key in _BUILD),
        "linked build tuple is incomplete",
    )
    witness["status"] = LINKED
    witness["linked_build"] = dict(linked_build)


def _valid_admissions(witness: Mapping[str, Any]) -> bool:
    occ, admissions = witness.get("occurrences"), witness.get("admissions")
    if not isinstance(occ, list) or not isinstance(admissions, list):
        return False
    try:
        digest = _digest(admissions)
        occurrence_digest = _digest(occ)
    except (TypeError, ValueError):
        return False
    if (
        witness.get("admission_verified") is not True
        or type(witness.get("count")) is not int
        or witness["count"] != len(occ)
        or occurrence_digest != witness.get("occurrences_sha256")
        or len(admissions) != len(occ)
        or not is_sha256(witness.get("admissions_sha256"))
        or digest != witness["admissions_sha256"]
    ):
        return False
    for proof, record in zip(occ, admissions, strict=True):
        if not isinstance(proof, Mapping) or not isinstance(record, Mapping) or set(record) != _ADMISSION:
            return False
        shape = proof.get("input_shape")
        if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            return False
        if (proof.get("input_type"), proof.get("output_type"), proof.get("carry_type")) != ("f32", "f32", "f64"):
            return False
        if (
            type(record["root_ordinal"]) is not int
            or record["root_ordinal"] != proof.get("root_ordinal")
            or not isinstance(record["all_ordinals"], list)
            or any(type(ordinal) is not int for ordinal in record["all_ordinals"])
            or record["all_ordinals"] != proof.get("all_ordinals")
            or _digest(record["semantic_signature"]) != _digest(_semantic_signature(proof))
        ):
            return False
        if any(
            not isinstance(record[key], str) or not record[key].strip()
            for key in ("host_profile", "host_declaration", "host_dtype_strategy")
        ):
            return False
        if record["software_declaration"] is not None and (
            not isinstance(record["software_declaration"], str) or not record["software_declaration"].strip()
        ):
            return False
        if any(
            not is_sha256(record[key])
            for key in ("host_package_sha256", "host_spec_sha256", "software_record_sha256", "capability_record_sha256")
        ):
            return False
        refusals = record["hardware_refusals"]
        if (
            not isinstance(refusals, list)
            or [item.get("rank") for item in refusals if isinstance(item, Mapping)] != sorted({1, len(shape)})
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"rank", "family", "refusal"}
                or type(item["rank"]) is not int
                or item["family"] != "reduction"
                or item["refusal"] not in {"undeclared_family", "input_dtype", "result_dtype"}
                for item in refusals
            )
        ):
            return False
    return True


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha: str) -> bool:
    """Mandatory empty or populated roster joined to the same whole-program ELF."""
    if not isinstance(source, Mapping) or not isinstance(entry, Mapping) or not is_sha256(candidate_sha):
        return False
    w = source.get(FIELD)
    if (
        not isinstance(w, Mapping)
        or set(w) != _TOP
        or w.get("status") != LINKED
        or w.get("scope") != SCOPE
        or w.get("source_verified") is not True
        or not _valid_admissions(w)
    ):
        return False
    if any(
        not is_sha256(w.get(key)) for key in ("raw_source_sha256", "normalized_source_sha256", "capture_receipt_sha256")
    ):
        return False
    if (w["raw_source_sha256"], w["normalized_source_sha256"], w["capture_receipt_sha256"]) != (
        source.get("source_sha256"),
        source.get("normalized_source_sha256"),
        source.get("capture_receipt_sha256"),
    ):
        return False
    if (
        type(w.get("n_source_operations")) is not int
        or w["n_source_operations"] < 1
        or w["n_source_operations"] != source.get("n_source_operations")
        or entry.get("source_sha256") != source.get("source_sha256")
    ):
        return False
    occ = w.get("occurrences")
    if (
        not isinstance(occ, list)
        or type(w.get("count")) is not int
        or w["count"] != len(occ)
        or not is_sha256(w.get("occurrences_sha256"))
        or _digest(occ) != w["occurrences_sha256"]
    ):
        return False
    selected = w.get("selected_index_observation")
    if selected != source.get("selected_index_observation") or not control._record_matches_selected(
        entry.get("index_lowering"), selected
    ):  # noqa: PLC2701
        return False
    bits = selected.get("index_bits") if isinstance(selected, Mapping) else None
    if type(bits) is not int or not 2 <= bits <= 128:
        return False
    seen_roots, seen_all = set(), set()
    for item in occ:
        if not isinstance(item, Mapping) or set(item) != {
            "root_ordinal",
            "all_ordinals",
            "input_shape",
            "output_shape",
            "axis_candidates",
            "total",
            "stride",
            "axis_extent",
            "input_type",
            "output_type",
            "carry_type",
            "index_bits_premise",
            "region_id",
        }:
            return False
        root, all_ordinals = item["root_ordinal"], item["all_ordinals"]
        if (
            type(root) is not int
            or root < 0
            or root >= w["n_source_operations"]
            or root in seen_roots
            or not isinstance(all_ordinals, list)
            or root not in all_ordinals
            or not all(type(i) is int and 0 <= i < w["n_source_operations"] for i in all_ordinals)
            or all_ordinals != sorted(set(all_ordinals))
            or seen_all.intersection(all_ordinals)
        ):
            return False
        seen_roots.add(root)
        seen_all.update(all_ordinals)
        shape, axes = item["input_shape"], item["axis_candidates"]
        maximum = min((1 << (bits - 1)) - 1, (1 << 63) - 1)
        if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            return False
        expected_axes = [
            axis
            for axis in range(len(shape))
            if math.prod(shape[axis + 1 :]) == item.get("stride") and shape[axis] == item.get("axis_extent")
        ]
        if (
            type(item.get("total")) is not int
            or math.prod(shape) != item["total"]
            or not 0 < item["total"] <= maximum
            or item["output_shape"] != shape
            or not isinstance(axes, list)
            or not axes
            or any(type(axis) is not int or axis < 0 or axis >= len(shape) for axis in axes)
            or axes != expected_axes
            or type(item.get("stride")) is not int
            or type(item.get("axis_extent")) is not int
            or not 0 < item["stride"] <= maximum
            or not 0 < item["axis_extent"] <= maximum
        ):
            return False
        if (
            item["index_bits_premise"] != bits
            or item["input_type"] != "f32"
            or item["output_type"] != "f32"
            or item["carry_type"] != "f64"
            or not isinstance(item["region_id"], str)
            or not item["region_id"]
        ):
            return False
    linked = w.get("linked_build")
    return bool(
        isinstance(linked, Mapping)
        and set(linked) == _BUILD
        and all(is_sha256(linked[key]) for key in _BUILD)
        and linked["candidate_tree_sha256"] == candidate_sha
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    )
