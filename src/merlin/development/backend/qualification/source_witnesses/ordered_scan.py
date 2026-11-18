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

from merlin.common.digest import is_sha256
from merlin.compiler.frontend.patterns.capture_normalization import normalize_capture_mlir
from merlin.compiler.ir import mlir_query as mq
from merlin.development.backend.qualification.source_witnesses import control_guards as control

# ---- private_ordered_scan_support --------------------------------------------------------------

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
