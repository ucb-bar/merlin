"""Source-only recognition of one closed floating bucketize counting reduction.

The proof describes the selected MLIR algorithm, not boundary sortedness,
PyTorch equivalence, selected index-conversion width, host placement, or
correctness of a linked executable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from merlin.frontends.linalg_patterns import (
    InvalidLinalgPattern,
    _checked_static_indexed_linalg_shell,
    _screen_static_linalg_source,
)


@dataclass(frozen=True)
class BucketizeSourcePattern:
    """Exact source algorithm, conditional on nondecreasing non-NaN boundaries.

    ``count_range`` bounds i64 arithmetic only. It does not prove that the
    selected compiler's index type can represent every static loop extent.
    """

    input_shape: tuple[int, ...]
    boundary_count: int
    right: bool
    comparison: str
    count_range: tuple[int, int]
    input_dtype: str = "f32"


@dataclass(frozen=True)
class BucketizeTraceBinding:
    """Three recorded right flags match; intervening ancestry is not proved."""

    prepared_node_id: str
    original_node_id: str
    quantized_node_id: str
    unresolved_ancestry_ids: tuple[str, ...]
    ancestry_status: str = "original_quantized_prepared_flag_match_only"


@dataclass(frozen=True)
class BucketizeSource:
    raw_sha256: str
    normalized_sha256: str
    trace_sha256: str
    ordinals: tuple[tuple[int, BucketizeSourcePattern], ...]
    trace_bindings: tuple[tuple[int, BucketizeTraceBinding], ...]


STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA = "merlin.static_bucketize_source_body.v1"
BUCKETIZE_COUNT_OPERATION = "f32_literal_boundary_count_i64"
BF16_BUCKETIZE_COUNT_OPERATION = "bf16_extended_f32_literal_boundary_count_i64"


@dataclass(frozen=True)
class StaticBucketizeHostPattern:
    """A literal-boundary prepared source form, not a frontend/numeric proof."""

    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    boundary_count: int
    right: bool
    comparison: str
    count_range: tuple[int, int]
    literal_sha256: str


def validate_static_bucketize_source_body(declaration: object) -> dict[str, str]:
    """Accept only the explicit source-body contract for the closed count form."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "operation"}
        or declaration.get("schema") != STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA
        or declaration.get("operation") not in {BUCKETIZE_COUNT_OPERATION, BF16_BUCKETIZE_COUNT_OPERATION}
    ):
        raise InvalidLinalgPattern("source_body requires the closed literal bucketize v1 declaration")
    return declaration


def literal_bucketize_boundary_fact(op, count: int) -> dict[str, str | None]:
    """Prove the exact boundary SSA operand is sorted, non-NaN f32 literal data."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, f32

    pending = {"status": "not_proved", "required_precondition": "nondecreasing_non_nan_f32_boundaries", "literal_sha256": None}
    boundary = op.inputs[1]
    owner = boundary.owner
    if type(owner) is not arith.ConstantOp:
        return pending
    _owned_operation(owner, arith.ConstantOp, {"value"})
    if owner.operands or len(owner.results) != 1 or owner.results[0] is not boundary:
        raise InvalidLinalgPattern("bucketize boundary literal has an invalid SSA owner")
    value = owner.properties["value"]
    if (
        not isinstance(value, DenseIntOrFPElementsAttr)
        or not isinstance(value.type, TensorType)
        or value.type != boundary.type
        or value.type.get_element_type() != f32
        or tuple(value.type.get_shape()) != (count,)
    ):
        raise InvalidLinalgPattern("bucketize boundary literal type or extent differs")
    values = tuple(value.get_values())
    if len(values) != count or any(type(item) is not float for item in values):
        raise InvalidLinalgPattern("bucketize boundary literal is incomplete")
    if any(math.isnan(item) for item in values) or any(left > right for left, right in zip(values, values[1:])):
        return pending
    return {
        "status": "source_dense_non_decreasing_non_nan_proved",
        "required_precondition": "nondecreasing_non_nan_f32_boundaries",
        "literal_sha256": sha256(value.data.data).hexdigest(),
    }


def bucketize_pattern_fits_index(
    pattern: BucketizeSourcePattern | StaticBucketizeHostPattern, index_bits: int
) -> bool:
    """Bound static extents and tensor byte spans by a caller-selected index width."""
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        return False
    maximum = (1 << (index_bits - 1)) - 1
    count = pattern.boundary_count
    if type(count) is not int or not 0 < count <= maximum or count > maximum // 4:
        return False
    shape = pattern.input_shape if isinstance(pattern, BucketizeSourcePattern) else pattern.shape
    elements = 1
    for extent in shape:
        if type(extent) is not int or not 0 < extent <= maximum or elements > maximum // extent:
            return False
        elements *= extent
    return bool(shape) and elements <= maximum // 8


def recognize_static_bucketize_body(op) -> StaticBucketizeHostPattern:
    """Recognize a closed literal count; the trace still owns the right flag."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr

    shell = _checked_static_indexed_linalg_shell(op)
    comparison_index = 1 if shell.ordered_types == ("bf16", "f32", "i64", "i64") else 0
    if len(shell.body) != 4 + comparison_index or type(shell.body[comparison_index]) is not arith.CmpfOp:
        raise InvalidLinalgPattern("bucketize source has no closed comparison body")
    predicate = shell.body[comparison_index].properties.get("predicate")
    if (
        not isinstance(predicate, IntegerAttr)
        or str(predicate.type) != "i64"
        or type(predicate.value.data) is not int
        or predicate.value.data not in (11, 12)
    ):
        raise InvalidLinalgPattern("bucketize source needs unordered lower/upper comparison")
    pattern = recognize_bucketize_source(op, right=predicate.value.data == 12)
    fact = literal_bucketize_boundary_fact(op, pattern.boundary_count)
    if fact["status"] != "source_dense_non_decreasing_non_nan_proved":
        raise InvalidLinalgPattern("bucketize boundary ordering/non-NaN premise is not proved")
    return StaticBucketizeHostPattern(
        BF16_BUCKETIZE_COUNT_OPERATION if pattern.input_dtype == "bf16" else BUCKETIZE_COUNT_OPERATION,
        pattern.input_shape,
        (pattern.input_dtype, "f32", "i64", "i64"),
        pattern.boundary_count,
        pattern.right,
        pattern.comparison,
        pattern.count_range,
        fact["literal_sha256"],
    )


def _owned_operation(op, expected_class, properties: set[str]) -> None:
    from xdsl.traits import Pure

    if type(op) is not expected_class or not type(op).has_trait(Pure):
        raise InvalidLinalgPattern("bucketize source has an unregistered or impure operation")
    if op.regions or op.successors or any(not name.startswith("prov.") for name in op.attributes):
        raise InvalidLinalgPattern("bucketize source has an effect, region, or unknown attribute")
    if set(op.properties) != properties:
        raise InvalidLinalgPattern("bucketize source operation properties differ from the closed form")


def _integer_constant(op, value: int) -> None:
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr

    _owned_operation(op, arith.ConstantOp, {"value"})
    attribute = op.properties["value"]
    if (
        not isinstance(attribute, IntegerAttr)
        or str(attribute.type) != "i64"
        or type(attribute.value.data) is not int
        or attribute.value.data != value
        or op.operands
        or len(op.results) != 1
        or str(op.results[0].type) != "i64"
    ):
        raise InvalidLinalgPattern("bucketize source requires exact i64 zero and one constants")


def recognize_bucketize_source(op, *, right: bool) -> BucketizeSourcePattern:
    """Prove one static f32 lower/upper-bound *count*, with an explicit right flag.

    For nondecreasing non-NaN boundaries and non-NaN input, counting unordered
    boundary<input (or boundary<=input) equals lower_bound (or upper_bound).
    A NaN input instead increments every boundary, as the actual ULT/ULE
    predicate specifies. Sortedness, input provenance, and the selected
    index-conversion width are not proved here.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(right) is not bool:
        raise InvalidLinalgPattern("bucketize right must be an explicit Boolean trace value")
    shell = _checked_static_indexed_linalg_shell(op)
    if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("bucketize requires two inputs and one initialized result")
    input_shape, boundary_shape = shell.input_shapes
    if (
        len(boundary_shape) != 1
        or shell.output_shapes != (input_shape,)
        or shell.ordered_types not in {("f32", "f32", "i64", "i64"), ("bf16", "f32", "i64", "i64")}
    ):
        raise InvalidLinalgPattern("bucketize input, boundary, and output tensor types differ")
    boundary_count = boundary_shape[0]
    if boundary_count > (1 << 63) - 1:
        raise InvalidLinalgPattern("bucketize count can overflow signed i64")
    rank = len(input_shape)
    if shell.shape != (*input_shape, boundary_count):
        raise InvalidLinalgPattern("bucketize loop extents differ from its tensor extents")
    parallel_map = AffineMap(rank + 1, 0, tuple(AffineDimExpr(i) for i in range(rank)))
    boundary_map = AffineMap(rank + 1, 0, (AffineDimExpr(rank),))
    if shell.indexing_maps != (parallel_map, boundary_map, parallel_map):
        raise InvalidLinalgPattern("bucketize input, boundary, and output maps differ")
    if shell.iterator_types != (IteratorType.PARALLEL,) * rank + (IteratorType.REDUCTION,):
        raise InvalidLinalgPattern("bucketize requires one final reduction axis")

    seed = op.outputs[0].owner
    if type(seed) is not tensor.SplatOp or seed.parent is not op.parent:
        raise InvalidLinalgPattern("bucketize init is not a local tensor.splat")
    _owned_operation(seed, tensor.SplatOp, set())
    if len(seed.operands) != 1 or len(seed.results) != 1 or seed.results[0] != op.outputs[0]:
        raise InvalidLinalgPattern("bucketize init does not splat its zero seed")
    zero = seed.operands[0].owner
    if type(zero) is not arith.ConstantOp or zero.parent is not op.parent:
        raise InvalidLinalgPattern("bucketize zero seed is not in the source block")
    _integer_constant(zero, 0)
    if seed.operands[0] != zero.results[0]:
        raise InvalidLinalgPattern("bucketize init does not use the checked zero")

    args, body = shell.args, shell.body
    input_value = args[0]
    if shell.ordered_types[0] == "bf16":
        if len(body) != 5:
            raise InvalidLinalgPattern("bf16 bucketize requires one exact extension before its count")
        extension, *body = body
        _owned_operation(extension, arith.ExtFOp, set())
        if tuple(extension.operands) != (args[0],) or len(extension.results) != 1 or str(extension.results[0].type) != "f32":
            raise InvalidLinalgPattern("bf16 bucketize extension is not exact bf16-to-f32")
        input_value = extension.results[0]
    if len(body) != 4:
        raise InvalidLinalgPattern("bucketize body is not compare, select, add, yield")
    compare, select, add, yld = body
    _owned_operation(compare, arith.CmpfOp, {"predicate", "fastmath"})
    predicate = compare.properties["predicate"]
    fastmath = compare.properties["fastmath"]
    expected_predicate = 12 if right else 11  # ULE or ULT, including unordered NaN.
    if (
        not isinstance(predicate, IntegerAttr)
        or str(predicate.type) != "i64"
        or predicate.value.data != expected_predicate
        or not isinstance(fastmath, arith.FastMathFlagsAttr)
        or fastmath.data
        or tuple(compare.operands) != (args[1], input_value)
        or len(compare.results) != 1
        or str(compare.results[0].type) != "i1"
    ):
        raise InvalidLinalgPattern("bucketize requires unflagged unordered boundary-vs-input comparison")
    _owned_operation(select, arith.SelectOp, set())
    if len(select.results) != 1 or str(select.results[0].type) != "i64":
        raise InvalidLinalgPattern("bucketize increment is not i64")
    if len(select.operands) != 3 or select.operands[0] != compare.results[0] or select.operands[2] != zero.results[0]:
        raise InvalidLinalgPattern("bucketize increment is not a one-or-zero select")
    one = select.operands[1].owner
    if type(one) is not arith.ConstantOp or one.parent is not op.parent:
        raise InvalidLinalgPattern("bucketize one is not in the source block")
    _integer_constant(one, 1)
    if select.operands[1] != one.results[0]:
        raise InvalidLinalgPattern("bucketize increment does not use the checked one")
    outer = tuple(op.parent.ops)
    if not (outer.index(zero) < outer.index(seed) < outer.index(op) and outer.index(one) < outer.index(op)):
        raise InvalidLinalgPattern("bucketize constants and zero init do not dominate the reduction")
    _owned_operation(add, arith.AddiOp, {"overflowFlags"})
    flags = add.properties["overflowFlags"]
    if (
        not isinstance(flags, arith.IntegerOverflowAttr)
        or flags.data
        or tuple(add.operands) != (args[2], select.results[0])
        or len(add.results) != 1
        or str(add.results[0].type) != "i64"
    ):
        raise InvalidLinalgPattern("bucketize requires unflagged accumulator-plus-increment")
    if (
        type(yld) is not YieldOp
        or yld.regions
        or yld.successors
        or yld.results
        or yld.properties
        or any(not name.startswith("prov.") for name in yld.attributes)
        or tuple(yld.operands) != (add.results[0],)
    ):
        raise InvalidLinalgPattern("bucketize does not yield exactly the new accumulator")
    return BucketizeSourcePattern(input_shape, boundary_count, right, "ule" if right else "ult", (0, boundary_count), shell.ordered_types[0])


def screen_bucketize_source(model_path: Path, trace_path: Path, ordinals: tuple[int, ...]) -> BucketizeSource:
    """Bind exact parsed MLIR ordinals to complete trace nodes and their right flag.

    This reads and hashes source/trace bytes; it does not authenticate their
    external issuer. The caller must separately verify the capture receipt.
    """
    try:
        trace_bytes = Path(trace_path).read_bytes()
        trace = json.loads(trace_bytes)
        raw = Path(model_path).read_bytes()
    except (OSError, UnicodeError, ValueError) as exc:
        raise InvalidLinalgPattern(f"bucketize source/trace is unreadable: {exc}") from exc
    if (
        not isinstance(trace, dict)
        or trace.get("schema") != "m2m.frontend_trace.v1"
        or trace.get("status") != "complete"
        or not isinstance(trace.get("mlir"), dict)
        or trace["mlir"].get("sha256") != sha256(raw).hexdigest()
        or trace["mlir"].get("bytes") != len(raw)
    ):
        raise InvalidLinalgPattern("bucketize trace does not bind complete source bytes")
    mlir_rows = trace["mlir"].get("operations")
    correspondences = trace["mlir"].get("source_correspondence")
    graphs = trace.get("graphs")
    if not isinstance(mlir_rows, list) or not isinstance(correspondences, list) or not isinstance(graphs, dict):
        raise InvalidLinalgPattern("bucketize trace is missing operation/source rosters")
    prepared = graphs.get("prepared")
    original = graphs.get("original")
    quantized = graphs.get("quantized")
    if any(not isinstance(graph, dict) for graph in (prepared, original, quantized)):
        raise InvalidLinalgPattern("bucketize trace lacks original, quantized, and prepared graphs")
    if any(not isinstance(graph.get("nodes"), list) for graph in (prepared, original, quantized)):
        raise InvalidLinalgPattern("bucketize trace graph nodes are malformed")
    if any(
        not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"]
        for nodes in (prepared["nodes"], original["nodes"], quantized["nodes"])
        for node in nodes
    ):
        raise InvalidLinalgPattern("bucketize trace graph node IDs are malformed")
    prepared_nodes = {node["id"]: node for node in prepared["nodes"]}
    original_nodes = {node["id"]: node for node in original["nodes"]}
    quantized_nodes = {node["id"]: node for node in quantized["nodes"]}
    if any(
        len(nodes) != len(graph["nodes"])
        for nodes, graph in ((prepared_nodes, prepared), (original_nodes, original), (quantized_nodes, quantized))
    ):
        raise InvalidLinalgPattern("bucketize trace graph nodes contain missing or duplicate IDs")
    transformations = trace.get("transformations")
    if not isinstance(transformations, list):
        raise InvalidLinalgPattern("bucketize trace lacks complete graph transformations")

    def require_relation(from_stage: str, to_stage: str, source_id: str, destination_id: str) -> None:
        stages = [
            item
            for item in transformations
            if isinstance(item, dict) and item.get("from_stage") == from_stage and item.get("to_stage") == to_stage
        ]
        if (
            len(stages) != 1
            or stages[0].get("status") != "complete"
            or not isinstance(stages[0].get("relations"), list)
        ):
            raise InvalidLinalgPattern("bucketize trace transformation stage is missing or incomplete")
        relations = [
            row
            for row in stages[0]["relations"]
            if isinstance(row, dict)
            and row.get("source_ids") == [source_id]
            and row.get("destination_ids") == [destination_id]
        ]
        if len(relations) != 1 or relations[0].get("kind") not in {"identity", "introduced"}:
            raise InvalidLinalgPattern("bucketize trace lacks an exact selected transformation relation")

    if (
        not isinstance(ordinals, tuple)
        or not ordinals
        or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
        or len(set(ordinals)) != len(ordinals)
    ):
        raise InvalidLinalgPattern("bucketize source requires distinct nonnegative MLIR ordinals")
    requested = ordinals
    sorted_ordinals = sorted(requested)
    next_ordinal = iter(sorted_ordinals)
    bindings: list[tuple[int, BucketizeTraceBinding]] = []

    def check(op):
        from xdsl.dialects.builtin import ArrayAttr, StringAttr

        ordinal = next(next_ordinal)
        if ordinal >= len(mlir_rows) or not isinstance(mlir_rows[ordinal], dict):
            raise InvalidLinalgPattern("bucketize trace has no matching MLIR ordinal")
        row = mlir_rows[ordinal]
        source_ids = op.attributes.get("prov.source_node_ids")
        if (
            not isinstance(source_ids, ArrayAttr)
            or len(source_ids.data) != 1
            or not isinstance(source_ids.data[0], StringAttr)
        ):
            raise InvalidLinalgPattern("bucketize source lacks one prepared-node identity")
        node_id = source_ids.data[0].data
        if (
            type(row.get("ordinal")) is not int
            or row["ordinal"] != ordinal
            or row.get("operation") != "linalg.generic"
            or row.get("source_node_ids") != [node_id]
            or row.get("operand_types") != [str(value.type) for value in op.operands]
            or row.get("result_types") != [str(value.type) for value in op.results]
        ):
            raise InvalidLinalgPattern("bucketize trace ordinal differs from parsed source")
        node = prepared_nodes.get(node_id)
        if not isinstance(node, dict) or node.get("target") != "aten.bucketize.Tensor":
            raise InvalidLinalgPattern("bucketize MLIR does not join a prepared bucketize node")
        kwargs = node.get("kwargs")
        if not isinstance(kwargs, dict) or set(kwargs) != {"right"} or type(kwargs["right"]) is not bool:
            raise InvalidLinalgPattern("bucketize trace lacks one explicit Boolean right flag")
        origin_ids = node.get("origin_node_ids")
        if (
            not isinstance(origin_ids, list)
            or not origin_ids
            or any(not isinstance(source_id, str) or not source_id for source_id in origin_ids)
            or len(set(origin_ids)) != len(origin_ids)
        ):
            raise InvalidLinalgPattern("bucketize ancestry IDs must be distinct nonempty strings")
        original_ids = [source_id for source_id in origin_ids if source_id.startswith("g:original:")]
        quantized_ids = [source_id for source_id in origin_ids if source_id.startswith("g:quantized:")]
        if len(original_ids) != 1 or len(quantized_ids) != 1:
            raise InvalidLinalgPattern("bucketize original or quantized ancestry is missing or ambiguous")
        original_node = original_nodes.get(original_ids[0])
        if not isinstance(original_node, dict) or original_node.get("target") != "aten.bucketize.Tensor":
            raise InvalidLinalgPattern("bucketize original ancestry does not resolve to a bucketize node")
        quantized_node = quantized_nodes.get(quantized_ids[0])
        if not isinstance(quantized_node, dict) or quantized_node.get("target") != "aten.bucketize.Tensor":
            raise InvalidLinalgPattern("bucketize quantized ancestry does not resolve to a bucketize node")
        for ancestor in (original_node, quantized_node):
            ancestor_kwargs = ancestor.get("kwargs")
            if (
                not isinstance(ancestor_kwargs, dict)
                or set(ancestor_kwargs) != {"right"}
                or type(ancestor_kwargs["right"]) is not bool
                or ancestor_kwargs["right"] is not kwargs["right"]
            ):
                raise InvalidLinalgPattern("bucketize original, quantized, and prepared right flags differ")
        quantized_origins = quantized_node.get("origin_node_ids")
        if (
            not isinstance(quantized_origins, list)
            or any(not isinstance(source_id, str) or not source_id for source_id in quantized_origins)
            or len(set(quantized_origins)) != len(quantized_origins)
            or set(quantized_origins) != set(origin_ids) - {quantized_ids[0]}
        ):
            raise InvalidLinalgPattern("bucketize quantized and prepared recorded ancestry differ")
        require_relation("original", "quantized", original_ids[0], quantized_ids[0])
        require_relation("quantized", "prepared", quantized_ids[0], node_id)
        matches = [entry for entry in correspondences if isinstance(entry, dict) and entry.get("node_id") == node_id]
        if len(matches) != 1 or matches[0].get("status") != "lowered":
            raise InvalidLinalgPattern("bucketize trace does not join the lowered source ordinal")
        linked = matches[0].get("mlir_ordinals")
        if (
            not isinstance(linked, list)
            or not linked
            or any(type(item) is not int or item < 0 or item >= len(mlir_rows) for item in linked)
            or len(set(linked)) != len(linked)
            or ordinal not in linked
        ):
            raise InvalidLinalgPattern("bucketize source correspondence ordinals are malformed or incomplete")
        if any(
            not isinstance(mlir_rows[item], dict)
            or type(mlir_rows[item].get("ordinal")) is not int
            or mlir_rows[item]["ordinal"] != item
            or mlir_rows[item].get("source_node_ids") != [node_id]
            for item in linked
        ):
            raise InvalidLinalgPattern("bucketize source correspondence contains an unrelated MLIR row")
        bindings.append(
            (
                ordinal,
                BucketizeTraceBinding(
                    node_id,
                    original_ids[0],
                    quantized_ids[0],
                    tuple(
                        source_id for source_id in origin_ids if source_id not in (original_ids[0], quantized_ids[0])
                    ),
                ),
            )
        )
        return recognize_bucketize_source(op, right=kwargs["right"])

    raw_sha256, normalized_sha256, evidence = _screen_static_linalg_source(model_path, requested, check)
    if raw_sha256 != sha256(raw).hexdigest():
        raise InvalidLinalgPattern("bucketize model source changed between trace check and parsing")
    return BucketizeSource(raw_sha256, normalized_sha256, sha256(trace_bytes).hexdigest(), evidence, tuple(bindings))
