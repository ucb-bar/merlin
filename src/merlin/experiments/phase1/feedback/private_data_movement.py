"""Source-structural data movement proofs, bound to exact linked model builds.

These are typed source observations and lowering obligations, not numerical
equivalence or a host/accelerator compute admission.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256

SCOPE = "typed source data movement and whole-program build obligation only; no value equivalence or execution"
PENDING = "source_structural_data_support_pending_build"
LINKED = "source_structural_data_support_linked"


def verify_transpose(op: Any, *, ordinal: int) -> None:
    """Verify one exact named transpose is static data movement, not computation."""
    from xdsl.dialects.builtin import DenseArrayBase, TensorType
    from xdsl.dialects.linalg.ops import TransposeOp, YieldOp

    if not isinstance(op, TransposeOp) or len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError(f"source operation {ordinal} is not a single-result linalg.transpose")
    if (
        set(op.properties) != {"permutation"}
        or any(not key.startswith("prov.") for key in op.attributes)
        or op.successors
    ):
        raise ValueError(f"source transpose {ordinal} carries an unrecognized semantic property or attribute")
    types = [value.type for value in (*op.operands, *op.results)]
    if any(not isinstance(value, TensorType) or not value.has_static_shape() for value in types):
        raise ValueError(f"source transpose {ordinal} has a dynamic or non-tensor type")
    input_type, init_type, output_type = types
    input_shape, init_shape, output_shape = (tuple(value.get_shape()) for value in types)
    rank = len(input_shape)
    if (
        rank == 0
        or any(
            type(extent) is not int or extent < 0
            for shape in (input_shape, init_shape, output_shape)
            for extent in shape
        )
        or input_type.element_type != init_type.element_type
        or input_type.element_type != output_type.element_type
        or init_shape != output_shape
    ):
        raise ValueError(f"source transpose {ordinal} changes element type or output storage")
    permutation_attr = op.properties["permutation"]
    if not isinstance(permutation_attr, DenseArrayBase) or str(permutation_attr.elt_type) != "i64":
        raise ValueError(f"source transpose {ordinal} has no typed static permutation")
    permutation = tuple(permutation_attr.iter_values())
    if (
        len(permutation) != rank
        or any(type(axis) is not int for axis in permutation)
        or sorted(permutation) != list(range(rank))
        or output_shape != tuple(input_shape[axis] for axis in permutation)
    ):
        raise ValueError(f"source transpose {ordinal} has no bijective shape-preserving permutation")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError(f"source transpose {ordinal} has no single data-movement body")
    block = op.regions[0].blocks[0]
    body = list(block.ops)
    yielded = body[0] if len(body) == 1 else None
    if (
        len(block.args) != 2
        or any(argument.type != input_type.element_type for argument in block.args)
        or not isinstance(yielded, YieldOp)
        or yielded.results
        or yielded.regions
        or yielded.successors
        or yielded.properties
        or any(not key.startswith("prov.") for key in yielded.attributes)
        or len(yielded.operands) != 1
        or yielded.operands[0] is not block.args[0]
    ):
        raise ValueError(f"source transpose {ordinal} computes or yields something other than its input")


def prove_transpose_source(
    module: Any, inventory: Mapping[str, Any], *, raw_sha256: str, normalized_sha256: str
) -> dict[str, Any]:
    """Join every inventory ordinal to the same parsed normalized transpose."""
    parsed = tuple(mq.walk(module))
    _check_source_inventory(parsed, inventory, raw_sha256, normalized_sha256)
    actual = {ordinal for ordinal, op in enumerate(parsed) if mq.op_name(op) == "linalg.transpose"}
    listed = []
    for row in inventory.get("signatures") or []:
        if row.get("mlir_operation") != "linalg.transpose":
            continue
        ordinals = row.get("ordinals")
        if (
            row.get("disposition") != "support_required"
            or not isinstance(ordinals, list)
            or type(row.get("count")) is not int
            or row["count"] != len(ordinals)
        ):
            raise ValueError("source transpose inventory has no exact support-required occurrences")
        for ordinal in ordinals:
            if type(ordinal) is not int or ordinal < 0 or ordinal >= len(parsed):
                raise ValueError("source transpose inventory has an invalid occurrence ordinal")
            verify_transpose(parsed[ordinal], ordinal=ordinal)
            listed.append(ordinal)
    if len(listed) != len(set(listed)) or set(listed) != actual:
        raise ValueError("source transpose inventory omits or repeats a parsed occurrence")
    return _pending(raw_sha256, normalized_sha256, listed)


def _check_source_inventory(
    parsed: tuple[Any, ...], inventory: Mapping[str, Any], raw_sha256: str, normalized_sha256: str
) -> None:
    normalization = inventory.get("capture_normalization")
    if (
        inventory.get("n_operations") != len(parsed)
        or inventory.get("capture_sha256") != raw_sha256
        or not isinstance(normalization, Mapping)
        or normalization.get("output_sha256") != normalized_sha256
    ):
        raise ValueError("source inventory differs from normalized parsed source")


def source_inventory_by_ordinal(
    parsed: tuple[Any, ...], inventory: Mapping[str, Any], raw_sha256: str, normalized_sha256: str
) -> dict[int, Mapping[str, Any]]:
    """Join every typed inventory signature to its exact parsed source operation."""
    _check_source_inventory(parsed, inventory, raw_sha256, normalized_sha256)
    rows = inventory.get("signatures")
    if not isinstance(rows, list):
        raise ValueError("source inventory has no signature list")
    joined: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source inventory signature is malformed")
        ordinals = row.get("ordinals")
        if type(row.get("count")) is not int or not isinstance(ordinals, list) or row["count"] != len(ordinals):
            raise ValueError("source inventory signature has no exact occurrence count")
        for ordinal in ordinals:
            if (
                type(ordinal) is not int
                or ordinal < 0
                or ordinal >= len(parsed)
                or ordinal in joined
                or row.get("mlir_operation") != mq.op_name(parsed[ordinal])
            ):
                raise ValueError("source inventory occurrence differs from parsed source")
            joined[ordinal] = row
    if len(joined) != len(parsed):
        raise ValueError("source inventory omits parsed source operations")
    return joined


def _pending(raw_sha256: str, normalized_sha256: str, listed: list[int]) -> dict[str, Any]:
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "count": len(listed),
        "ordinals": sorted(listed),
    }


def _copy_affine_range(expr: Any, shape: tuple[int, ...]) -> tuple[int, int, set[int]]:
    """Conservative integer bounds for symbol-free linear copy addresses."""
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr

    if isinstance(expr, AffineConstantExpr) and type(expr.value) is int:
        return expr.value, expr.value, set()
    if isinstance(expr, AffineDimExpr) and type(expr.position) is int and 0 <= expr.position < len(shape):
        return 0, shape[expr.position] - 1, {expr.position}
    if isinstance(expr, AffineBinaryOpExpr):
        lo, hi, dims = _copy_affine_range(expr.lhs, shape)
        other_lo, other_hi, other_dims = _copy_affine_range(expr.rhs, shape)
        if expr.kind == AffineBinaryOpKind.Add:
            return lo + other_lo, hi + other_hi, dims | other_dims
        if expr.kind == AffineBinaryOpKind.Mul and (
            isinstance(expr.lhs, AffineConstantExpr) or isinstance(expr.rhs, AffineConstantExpr)
        ):
            corners = [left * right for left in (lo, hi) for right in (other_lo, other_hi)]
            return min(corners), max(corners), dims | other_dims
    raise ValueError("unsupported copy affine address")


def verify_generic_copy(op: Any, *, ordinal: int) -> None:
    """Prove pure bounded linear movement/projection/singleton broadcast from typed IR only."""
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    if mq.op_name(op) != "linalg.generic" or len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError(f"source generic copy {ordinal} has incorrect segment arity")
    if set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"} or any(
        not key.startswith("prov.") for key in op.attributes
    ):
        raise ValueError(f"source generic copy {ordinal} has unrecognized semantic properties or attributes")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (1, 1)
    ):
        raise ValueError(f"source generic copy {ordinal} has incorrect operand segments")
    types = [value.type for value in (*op.operands, *op.results)]
    if any(not isinstance(value, TensorType) or not value.has_static_shape() for value in types):
        raise ValueError(f"source generic copy {ordinal} has a dynamic or non-tensor type")
    input_type, init_type, result_type = types
    input_shape, init_shape, result_shape = (tuple(value.get_shape()) for value in types)
    rank = len(result_shape)
    if (
        rank == 0
        or any(
            type(extent) is not int or extent <= 0
            for shape in (input_shape, init_shape, result_shape)
            for extent in shape
        )
        or input_type.element_type != init_type.element_type
        or input_type.element_type != result_type.element_type
        or init_shape != result_shape
    ):
        raise ValueError(f"source generic copy {ordinal} changes element type or output storage")
    maps = op.properties["indexing_maps"]
    iterators = op.properties["iterator_types"]
    if (
        not isinstance(maps, ArrayAttr)
        or len(maps) != 2
        or any(not isinstance(item, AffineMapAttr) for item in maps)
        or not isinstance(iterators, ArrayAttr)
        or len(iterators) != rank
        or any(not isinstance(item, IteratorTypeAttr) or item.data != IteratorType.PARALLEL for item in iterators)
    ):
        raise ValueError(f"source generic copy {ordinal} has unknown maps or non-parallel iterators")
    input_map, output_map = (item.data for item in maps)
    if (
        input_map.num_dims != rank
        or output_map.num_dims != rank
        or input_map.num_symbols != 0
        or output_map.num_symbols != 0
        or len(input_map.results) != len(input_shape)
        or len(output_map.results) != rank
        or any(
            not isinstance(expr, AffineDimExpr) or expr.position != axis for axis, expr in enumerate(output_map.results)
        )
    ):
        raise ValueError(f"source generic copy {ordinal} has no identity destination map")
    seen_dims: set[int] = set()
    for extent, expr in zip(input_shape, input_map.results, strict=True):
        if isinstance(expr, AffineConstantExpr) and type(expr.value) is int and expr.value == 0 and extent == 1:
            continue
        try:
            lower, upper, dims = _copy_affine_range(expr, result_shape)
        except ValueError as exc:
            raise ValueError(f"source generic copy {ordinal} has an unsupported input map") from exc
        if not dims or seen_dims & dims or lower < 0 or upper >= extent:
            raise ValueError(f"source generic copy {ordinal} has a non-projection or out-of-bounds input map")
        seen_dims.update(dims)
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError(f"source generic copy {ordinal} has no single pure body")
    block = op.regions[0].blocks[0]
    body = list(block.ops)
    if (
        len(block.args) != 2
        or any(argument.type != input_type.element_type for argument in block.args)
        or len(body) != 1
        or mq.op_name(body[0]) != "linalg.yield"
        or body[0].results
        or body[0].regions
        or body[0].properties
        or any(not key.startswith("prov.") for key in body[0].attributes)
        or len(body[0].operands) != 1
        or body[0].operands[0] is not block.args[0]
    ):
        raise ValueError(f"source generic copy {ordinal} computes or yields something other than its input")


def prove_generic_copy_source(
    module: Any, inventory: Mapping[str, Any], *, raw_sha256: str, normalized_sha256: str
) -> dict[str, Any]:
    """Join every movement-family generic copy to exactly one source inventory ordinal."""
    from merlin.xdsl_dialects.lowering.contraction_coverage import classify_generic

    parsed = tuple(mq.walk(module))
    _check_source_inventory(parsed, inventory, raw_sha256, normalized_sha256)
    actual = {
        ordinal
        for ordinal, op in enumerate(parsed)
        if mq.op_name(op) == "linalg.generic" and classify_generic(op) == "movement"
    }
    listed: list[int] = []
    for row in inventory.get("signatures") or []:
        if row.get("mlir_operation") != "linalg.generic" or row.get("semantic_family") != "movement":
            continue
        ordinals = row.get("ordinals")
        if (
            row.get("disposition") != "support_required"
            or not isinstance(ordinals, list)
            or type(row.get("count")) is not int
            or row["count"] != len(ordinals)
        ):
            raise ValueError("source generic copy inventory has no exact support-required occurrences")
        for ordinal in ordinals:
            if type(ordinal) is not int or ordinal < 0 or ordinal >= len(parsed):
                raise ValueError("source generic copy inventory has an invalid occurrence ordinal")
            verify_generic_copy(parsed[ordinal], ordinal=ordinal)
            listed.append(ordinal)
    if len(listed) != len(set(listed)) or set(listed) != actual:
        raise ValueError("source generic copy inventory omits or repeats a parsed occurrence")
    return _pending(raw_sha256, normalized_sha256, listed)


def linked_movement_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Require both data-movement proofs to bind the exact source and linked ELF."""
    for key in ("transpose_data_support", "generic_copy_data_support"):
        proof = source.get(key)
        if not isinstance(proof, Mapping):
            return False
        ordinals = proof.get("ordinals")
        if (
            proof.get("status") != LINKED
            or proof.get("scope") != SCOPE
            or type(proof.get("count")) is not int
            or proof["count"] < 0
            or not isinstance(ordinals, list)
            or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
            or ordinals != sorted(set(ordinals))
            or proof["count"] != len(ordinals)
            or not is_sha256(source.get("source_sha256"))
            or proof.get("raw_source_sha256") != source["source_sha256"]
            or not is_sha256(proof.get("normalized_source_sha256"))
            or proof.get("normalized_source_sha256") != source.get("normalized_source_sha256")
            or entry.get("source_sha256") != source["source_sha256"]
        ):
            return False
        linked = proof.get("linked_build")
        if (
            not isinstance(linked, Mapping)
            or set(linked) != {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
            or linked.get("candidate_tree_sha256") != candidate_sha256
            or linked.get("capture_tree_sha256") != entry.get("capture_tree_sha256")
            or linked.get("elf_sha256") != entry.get("elf_sha256")
            or any(not is_sha256(linked[part]) for part in linked)
        ):
            return False
    return True
