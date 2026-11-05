"""Shared structural recognition of admitted Linalg scalar regions.

The reader and independent source verifier use this one checked pattern so
provenance labels or operation names cannot silently stand in for the body.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class InvalidLinalgPattern(ValueError):
    pass


@dataclass(frozen=True)
class StaticPointwisePattern:
    """A source-body description, not a host or accelerator admission."""

    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    predicate: str | None = None


@dataclass(frozen=True)
class StaticPointwiseSource:
    """Exact parsed source ordinals and bytes; not a linked or numerical proof."""

    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticPointwisePattern], ...]


@dataclass(frozen=True)
class StaticProjectedPointwisePattern:
    """A closed static input projection and scalar body, not an admission."""

    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_maps: tuple[str, ...]
    predicate: str | None = None


@dataclass(frozen=True)
class StaticProjectedPointwiseSource:
    """Exact prepared-source identity and ordinal roster; no linked proof."""

    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticProjectedPointwisePattern], ...]


@dataclass(frozen=True)
class _StaticLinalgShell:
    shape: tuple[int, ...]
    args: tuple[Any, ...]
    body: tuple[Any, ...]
    ordered_types: tuple[str, ...]
    input_maps: tuple[Any, ...]


@dataclass(frozen=True)
class _StaticIndexedLinalgShell:
    """Checked iteration geometry; the scalar algorithm remains unproved."""

    shape: tuple[int, ...]
    args: tuple[Any, ...]
    body: tuple[Any, ...]
    ordered_types: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    indexing_maps: tuple[Any, ...]
    iterator_types: tuple[Any, ...]


def _checked_static_indexed_linalg_shell(op) -> _StaticIndexedLinalgShell:
    """Check a static, dimension-projected GenericOp without admitting its body.

    Unlike the pointwise shell, this allows reduction iterators and multiple
    initialized results. Every loop bound must be supplied consistently by a
    tensor dimension; symbols, offsets, repeated dimensions, unusual
    encodings and unknown operation metadata refuse. Callers must separately
    prove the seed, scalar body, reduction semantics and any required bounds.
    """
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, NoneAttr, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineDimExpr

    if type(op) is not GenericOp:
        raise InvalidLinalgPattern("indexed shell requires registered inputs and initialized results")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("indexed shell has unknown properties or successors")
    if any(not name.startswith("prov.") for name in op.attributes):
        raise InvalidLinalgPattern("indexed shell has an unknown attribute")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or len(segment_sizes := tuple(segments.iter_values())) != 2
        or any(type(size) is not int or size < 1 for size in segment_sizes)
        or sum(segment_sizes) != len(op.operands)
        or segment_sizes[1] != len(op.results)
    ):
        raise InvalidLinalgPattern("indexed shell has inconsistent operand segments")
    tensors = (*op.inputs, *op.outputs)
    values = (*tensors, *op.results)
    if any(not isinstance(value.type, TensorType) for value in values):
        raise InvalidLinalgPattern("indexed shell requires tensor operands and results")
    shapes = tuple(tuple(value.type.get_shape()) for value in values)
    if any(type(dim) is not int or dim < 0 for shape in shapes for dim in shape):
        raise InvalidLinalgPattern("indexed shell requires nonnegative static extents")
    if any(not isinstance(value.type.encoding, NoneAttr) for value in values):
        raise InvalidLinalgPattern("indexed shell requires default tensor encodings")
    if any(result.type != output.type for output, result in zip(op.outputs, op.results, strict=True)):
        raise InvalidLinalgPattern("indexed shell result differs from its initialized output")
    if not isinstance(op.iterator_types, ArrayAttr) or any(
        not isinstance(attribute, IteratorTypeAttr) for attribute in op.iterator_types
    ):
        raise InvalidLinalgPattern("indexed shell iterator metadata is malformed")
    iterators = tuple(attribute.data for attribute in op.iterator_types)
    if not iterators or any(iterator not in (IteratorType.PARALLEL, IteratorType.REDUCTION) for iterator in iterators):
        raise InvalidLinalgPattern("indexed shell requires parallel or reduction iterators")
    if not isinstance(op.indexing_maps, ArrayAttr) or any(
        not isinstance(attribute, AffineMapAttr) for attribute in op.indexing_maps
    ):
        raise InvalidLinalgPattern("indexed shell indexing maps are malformed")
    maps = tuple(attribute.data for attribute in op.indexing_maps)
    if len(maps) != len(tensors):
        raise InvalidLinalgPattern("indexed shell indexing-map arity differs from tensor arity")
    bounds: list[int | None] = [None] * len(iterators)
    for mapping, tensor_shape in zip(maps, shapes[: len(tensors)], strict=True):
        if mapping.num_dims != len(iterators) or mapping.num_symbols != 0 or len(mapping.results) != len(tensor_shape):
            raise InvalidLinalgPattern("indexed shell map rank or symbol count is inconsistent")
        seen: set[int] = set()
        for expression, extent in zip(mapping.results, tensor_shape, strict=True):
            if (
                type(expression) is not AffineDimExpr
                or type(expression.position) is not int
                or not 0 <= expression.position < len(iterators)
                or expression.position in seen
            ):
                raise InvalidLinalgPattern("indexed shell map is not a distinct-dimension projection")
            position = expression.position
            seen.add(position)
            if bounds[position] is not None and bounds[position] != extent:
                raise InvalidLinalgPattern("indexed shell loop bounds disagree between tensors")
            bounds[position] = extent
    if any(bound is None for bound in bounds):
        raise InvalidLinalgPattern("indexed shell has an unbounded iteration dimension")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("indexed shell requires a single scalar block")
    block = op.regions[0].block
    args = tuple(block.args)
    ordered_types = tuple(str(value.type.element_type) for value in values)
    if tuple(str(arg.type) for arg in args) != ordered_types[: len(tensors)]:
        raise InvalidLinalgPattern("indexed shell block argument types differ from tensor elements")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL verification is a fail-closed source boundary
        raise InvalidLinalgPattern(f"indexed shell operation failed xDSL verification: {exc}") from exc
    return _StaticIndexedLinalgShell(
        tuple(bound for bound in bounds if bound is not None),
        args,
        tuple(block.ops),
        ordered_types,
        shapes[: len(op.inputs)],
        shapes[len(op.inputs) : len(tensors)],
        maps,
        iterators,
    )


_POINTWISE_TYPES = {
    "arith.addf": (("f32", "f32"), "f32"),
    "arith.subf": (("f32", "f32"), "f32"),
    "arith.mulf": (("f32", "f32"), "f32"),
    "arith.negf": (("f32",), "f32"),
    "arith.select": (("i1", "f32", "f32"), "f32"),
    "arith.cmpf": (("f32", "f32"), "i1"),
    "arith.addi": (("i64", "i64"), "i64"),
    "arith.subi": (("i64", "i64"), "i64"),
    "arith.muli": (("i64", "i64"), "i64"),
    "arith.cmpi": (("i64", "i64"), "i1"),
    "arith.andi": (("i1", "i1"), "i1"),
    "arith.xori": (("i1", "i1"), "i1"),
    "arith.extui": (("i1",), "i64"),
    "arith.uitofp": (("i1",), "f32"),
    "arith.sitofp": (("i64",), "f32"),
}
_FLOAT_PREDICATES = {1: "oeq", 2: "ogt", 3: "oge", 4: "olt", 5: "ole", 6: "one"}
_SIGNED_PREDICATES = {0: "eq", 1: "ne", 2: "slt", 3: "sle", 4: "sgt", 5: "sge"}
STATIC_POINTWISE_SOURCE_BODY_SCHEMA = "merlin.static_pointwise_source_body.v1"
STATIC_PROJECTED_POINTWISE_BODY_SCHEMA = "merlin.static_projected_pointwise_body.v1"


def static_pointwise_ordered_types(declaration: object) -> tuple[str, ...]:
    """Return exact input/init/result types of one validated source body."""
    body = validate_static_pointwise_source_body(declaration)
    inputs, result = _POINTWISE_TYPES[body["operation"]]
    return (*inputs, result, result)


def static_projected_pointwise_ordered_types(declaration: object) -> tuple[str, ...]:
    """Return exact input/init/result types for a projected scalar body."""
    body = validate_static_projected_pointwise_source_body(declaration)
    inputs, result = _POINTWISE_TYPES[body["operation"]]
    return (*inputs, result, result)


def validate_static_pointwise_source_body(declaration: object) -> dict[str, str]:
    """Validate the closed, opt-in declaration for this source-only proof."""
    if not isinstance(declaration, dict) or declaration.get("schema") != STATIC_POINTWISE_SOURCE_BODY_SCHEMA:
        raise InvalidLinalgPattern("source_body requires the static pointwise v1 schema")
    operation = declaration.get("operation")
    if not isinstance(operation, str) or operation not in _POINTWISE_TYPES:
        raise InvalidLinalgPattern("source_body has no reviewed pointwise scalar operation")
    comparison = operation in {"arith.cmpf", "arith.cmpi"}
    expected = {"schema", "operation", "predicate"} if comparison else {"schema", "operation"}
    if set(declaration) != expected:
        raise InvalidLinalgPattern("source_body has missing or extra fields")
    if comparison:
        allowed = _FLOAT_PREDICATES if operation == "arith.cmpf" else _SIGNED_PREDICATES
        if declaration["predicate"] not in allowed.values():
            raise InvalidLinalgPattern("source_body comparison predicate is not ordered/signed")
    return dict(declaration)


def validate_static_projected_pointwise_source_body(declaration: object) -> dict[str, str]:
    """Validate a separate opt-in projection schema, never the strict v1 schema."""
    if not isinstance(declaration, dict) or declaration.get("schema") != STATIC_PROJECTED_POINTWISE_BODY_SCHEMA:
        raise InvalidLinalgPattern("source_body requires the static projected pointwise v1 schema")
    strict = {**declaration, "schema": STATIC_POINTWISE_SOURCE_BODY_SCHEMA}
    validate_static_pointwise_source_body(strict)
    return dict(declaration)


def validate_serialized_static_projected_pointwise_pattern(pattern: object) -> None:
    """Independently recheck recorded types, static shapes, and exact input maps."""
    from collections.abc import Mapping

    from xdsl.context import Context
    from xdsl.parser import Parser

    if not isinstance(pattern, Mapping) or set(pattern) != {
        "operation",
        "shape",
        "ordered_types",
        "input_shapes",
        "input_maps",
        "predicate",
    }:
        raise InvalidLinalgPattern("serialized projected pointwise pattern is not closed")
    declaration = {"schema": STATIC_PROJECTED_POINTWISE_BODY_SCHEMA, "operation": pattern["operation"]}
    if pattern["predicate"] is not None:
        declaration["predicate"] = pattern["predicate"]
    expected_types = static_projected_pointwise_ordered_types(declaration)
    shape, shapes, maps = pattern["shape"], pattern["input_shapes"], pattern["input_maps"]
    if (
        not isinstance(shape, (tuple, list))
        or not shape
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        or not isinstance(shapes, (tuple, list))
        or len(shapes) != len(expected_types) - 2
        or not isinstance(maps, (tuple, list))
        or len(maps) != len(shapes)
        or not isinstance(pattern["ordered_types"], (tuple, list))
        or tuple(pattern["ordered_types"]) != expected_types
    ):
        raise InvalidLinalgPattern("serialized projected pointwise tensor roster is invalid")
    for input_shape, text in zip(shapes, maps, strict=True):
        if (
            not isinstance(input_shape, (tuple, list))
            or any(type(dim) is not int or dim <= 0 for dim in input_shape)
            or not isinstance(text, str)
        ):
            raise InvalidLinalgPattern("serialized projected pointwise input is malformed")
        try:
            mapping = Parser(Context(), text).parse_affine_map()
        except Exception as exc:  # noqa: BLE001 - untrusted serialized map must refuse
            raise InvalidLinalgPattern("serialized projected pointwise input map does not parse") from exc
        if str(mapping) != text:
            raise InvalidLinalgPattern("serialized projected pointwise input map is not canonical")
        validate_singleton_projection_map(mapping, tuple(shape), tuple(input_shape))


def validate_singleton_projection_map(mapping, output_shape: tuple[int, ...], input_shape: tuple[int, ...]) -> None:
    """Require an ascending dimension projection or zero at a singleton input axis."""
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    if mapping.num_dims != len(output_shape) or mapping.num_symbols != 0 or len(mapping.results) != len(input_shape):
        raise InvalidLinalgPattern("pointwise input map is not a static singleton projection")
    last_dim = -1
    for extent, expr in zip(input_shape, mapping.results, strict=True):
        if isinstance(expr, AffineConstantExpr) and type(expr.value) is int and expr.value == 0 and extent == 1:
            continue
        if (
            not isinstance(expr, AffineDimExpr)
            or type(expr.position) is not int
            or expr.position <= last_dim
            or expr.position >= len(output_shape)
            or extent != output_shape[expr.position]
        ):
            raise InvalidLinalgPattern("pointwise input map is not a static singleton projection")
        last_dim = expr.position


def _checked_static_linalg_shell(op, *, singleton_projection_inputs: bool = False) -> _StaticLinalgShell:
    """Check the common typed tensor/map shell; never admit a scalar body here."""

    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseArrayBase, NoneAttr, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineMap

    if type(op) is not GenericOp or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("expected registered one-result linalg.GenericOp")
    if op.successors:
        raise InvalidLinalgPattern("linalg.generic has an unexpected successor")
    if set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("unrecognized linalg.generic property")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (len(op.inputs), 1)
    ):
        raise InvalidLinalgPattern("pointwise operand segments differ from input/output arity")
    if any(not name.startswith("prov.") for name in op.attributes):
        raise InvalidLinalgPattern("unrecognized linalg.generic attribute")
    values = (*op.inputs, *op.outputs, *op.results)
    if not all(isinstance(value.type, TensorType) for value in values):
        raise InvalidLinalgPattern("pointwise operands and result must be tensors")
    shape = tuple(op.results[0].type.get_shape())
    if not shape or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise InvalidLinalgPattern("pointwise shape is not positive-rank static")
    if not singleton_projection_inputs and any(tuple(value.type.get_shape()) != shape for value in values):
        raise InvalidLinalgPattern("pointwise tensor shapes differ")
    if singleton_projection_inputs and (
        any(dim <= 0 for dim in shape)
        or any(not value.type.has_static_shape() or any(dim <= 0 for dim in value.type.get_shape()) for value in values)
    ):
        raise InvalidLinalgPattern("projected pointwise tensors require positive static extents")
    if any(not isinstance(value.type.encoding, NoneAttr) for value in values):
        raise InvalidLinalgPattern("pointwise tensor encoding is not default")
    if op.results[0].type != op.outputs[0].type:
        raise InvalidLinalgPattern("pointwise result type differs from output init")
    if not isinstance(op.indexing_maps, ArrayAttr) or any(
        not isinstance(attribute, AffineMapAttr) for attribute in op.indexing_maps
    ):
        raise InvalidLinalgPattern("pointwise indexing maps are malformed")
    maps = tuple(attribute.data for attribute in op.indexing_maps)
    if len(maps) != len(op.inputs) + 1 or maps[-1] != AffineMap.identity(len(shape)):
        raise InvalidLinalgPattern("pointwise destination indexing map is not identity")
    if not singleton_projection_inputs:
        if maps != (AffineMap.identity(len(shape)),) * len(maps):
            raise InvalidLinalgPattern("pointwise indexing maps are not identity")
    else:
        for input_value, input_map in zip(op.inputs, maps[:-1], strict=True):
            input_shape = tuple(input_value.type.get_shape())
            validate_singleton_projection_map(input_map, shape, input_shape)
    if not isinstance(op.iterator_types, ArrayAttr) or any(
        not isinstance(attribute, IteratorTypeAttr) for attribute in op.iterator_types
    ):
        raise InvalidLinalgPattern("pointwise iterator metadata is malformed")
    if tuple(attribute.data for attribute in op.iterator_types) != (IteratorType.PARALLEL,) * len(shape):
        raise InvalidLinalgPattern("pointwise iterators are not all parallel")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("pointwise region must have one block")
    block = op.regions[0].block
    args = tuple(block.args)
    ordered_types = tuple(str(value.type.element_type) for value in values)
    if tuple(str(arg.type) for arg in args) != ordered_types[:-1]:
        raise InvalidLinalgPattern("pointwise block argument types differ from tensor elements")
    body = tuple(block.ops)
    return _StaticLinalgShell(shape, args, body, ordered_types, maps[:-1])


def recognize_static_pointwise(op) -> StaticPointwisePattern:
    """Prove an exact input-only positive-rank static pointwise body; make no placement claim.

    This is intentionally one scalar instruction and one yield. Algebraically
    similar regions, broadcasts, reductions, init reads and flagged arithmetic
    need different proofs and must not enter by an operation-name match.
    """
    return _recognize_static_scalar_pointwise(op, _checked_static_linalg_shell(op))


def recognize_static_projected_pointwise(op) -> StaticProjectedPointwisePattern:
    """Prove static singleton-projected inputs with one exact scalar/yield body."""
    shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    scalar = _recognize_static_scalar_pointwise(op, shell)
    return StaticProjectedPointwisePattern(
        scalar.operation,
        scalar.shape,
        scalar.ordered_types,
        tuple(tuple(value.type.get_shape()) for value in op.inputs),
        tuple(str(mapping) for mapping in shell.input_maps),
        scalar.predicate,
    )


def recognize_static_f32_division(op) -> StaticProjectedPointwisePattern:
    """Check projected, unflagged f32 division without admitting it to host schemas."""
    shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    scalar = _recognize_static_scalar_pointwise(
        op, shell, additional_signatures={"arith.divf": (("f32", "f32"), "f32")},
    )
    if scalar.operation != "arith.divf":
        raise InvalidLinalgPattern("f32 division needs exactly one arith.divf and yield")
    return StaticProjectedPointwisePattern(
        scalar.operation,
        scalar.shape,
        scalar.ordered_types,
        tuple(tuple(value.type.get_shape()) for value in op.inputs),
        tuple(str(mapping) for mapping in shell.input_maps),
        scalar.predicate,
    )


def _recognize_static_scalar_pointwise(
    op, shell: _StaticLinalgShell, *,
    additional_signatures: Mapping[str, tuple[tuple[str, ...], str]] | None = None,
) -> StaticPointwisePattern:
    """Common closed scalar body for strict identity and opt-in projections."""
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.traits import Pure

    shape, args, body, ordered_types = shell.shape, shell.args, shell.body, shell.ordered_types
    if len(body) != 2:
        raise InvalidLinalgPattern("pointwise body is not one scalar operation and yield")
    if type(body[1]) is not YieldOp:
        raise InvalidLinalgPattern("pointwise body does not end in registered linalg.yield")
    scalar, yld = body
    signature = _POINTWISE_TYPES.get(scalar.name)
    if signature is None and additional_signatures is not None:
        signature = additional_signatures.get(scalar.name)
    if signature is None or type(scalar) not in tuple(arith.Arith.operations) or not type(scalar).has_trait(Pure):
        raise InvalidLinalgPattern("scalar operation is not a registered pure arith pointwise form")
    if scalar.regions or scalar.successors:
        raise InvalidLinalgPattern("scalar operation has an unexpected region or successor")
    if yld.regions or yld.successors:
        raise InvalidLinalgPattern("linalg.yield has an unexpected region or successor")
    if yld.properties or any(not name.startswith("prov.") for name in yld.attributes):
        raise InvalidLinalgPattern("linalg.yield has an unrecognized attribute or property")
    inputs, result = signature
    if len(op.inputs) != len(inputs) or tuple(str(arg.type) for arg in args[:-1]) != inputs:
        raise InvalidLinalgPattern("scalar input types differ from proved pointwise signature")
    if str(args[-1].type) != result or ordered_types[-1] != result:
        raise InvalidLinalgPattern("scalar output type differs from proved pointwise signature")
    if tuple(scalar.operands) != args[:-1] or len(scalar.results) != 1 or str(scalar.results[0].type) != result:
        raise InvalidLinalgPattern("scalar operation does not consume exactly input block arguments")
    if tuple(yld.operands) != (scalar.results[0],):
        raise InvalidLinalgPattern("pointwise body does not yield only the scalar result")
    if any(not name.startswith("prov.") for name in scalar.attributes):
        raise InvalidLinalgPattern("scalar operation has unrecognized attributes")
    expected_properties = {
        "arith.cmpf": {"predicate", "fastmath"},
        "arith.cmpi": {"predicate"},
        "arith.addf": {"fastmath"},
        "arith.subf": {"fastmath"},
        "arith.mulf": {"fastmath"},
        "arith.divf": {"fastmath"},
        "arith.negf": {"fastmath"},
        "arith.addi": {"overflowFlags"},
        "arith.subi": {"overflowFlags"},
        "arith.muli": {"overflowFlags"},
    }.get(scalar.name, set())
    if not set(scalar.properties) <= expected_properties:
        raise InvalidLinalgPattern("scalar operation has unrecognized properties")
    for flag in ("fastmath", "overflowFlags"):
        value = scalar.properties.get(flag)
        flag_type = arith.FastMathFlagsAttr if flag == "fastmath" else arith.IntegerOverflowAttr
        if value is not None and (not isinstance(value, flag_type) or value.data):
            raise InvalidLinalgPattern("scalar operation has fast-math or integer overflow flags")
    predicate = None
    if scalar.name in ("arith.cmpf", "arith.cmpi"):
        from xdsl.dialects.builtin import IntegerAttr

        predicate_attribute = scalar.properties.get("predicate")
        if not isinstance(predicate_attribute, IntegerAttr):
            raise InvalidLinalgPattern("comparison predicate property is missing or malformed")
        allowed = _FLOAT_PREDICATES if scalar.name == "arith.cmpf" else _SIGNED_PREDICATES
        predicate = allowed.get(predicate_attribute.value.data)
        if predicate is None:
            raise InvalidLinalgPattern("comparison predicate is not ordered/signed")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL verification is a fail-closed source boundary
        raise InvalidLinalgPattern(f"pointwise operation failed xDSL verification: {exc}") from exc
    return StaticPointwisePattern(scalar.name, shape, ordered_types, predicate)


def _screen_static_linalg_source(
    path: Path, ordinals: tuple[int, ...], recognizer: Callable[[Any], Any]
) -> tuple[str, str, tuple[tuple[int, Any], ...]]:
    """Reparse and independently hash every requested normalized source ordinal."""
    from hashlib import sha256

    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir

    requested = tuple(ordinals)
    if not requested or any(type(ordinal) is not int or ordinal < 0 for ordinal in requested):
        raise InvalidLinalgPattern("source ordinals must be nonnegative integers")
    if len(set(requested)) != len(requested):
        raise InvalidLinalgPattern("source ordinals contain a duplicate")
    raw = Path(path).read_bytes()
    raw_sha256 = sha256(raw).hexdigest()
    try:
        normalized, receipt = normalize_capture_mlir(raw.decode("utf-8"))
        module = mq.parse(normalized)
        module.verify()
    except Exception as exc:  # noqa: BLE001 - parser and xDSL verifier have distinct exception bases
        raise InvalidLinalgPattern(f"normalized source module failed verification: {exc}") from exc
    normalized_sha256 = sha256(normalized.encode("utf-8")).hexdigest()
    if receipt["input_sha256"] != raw_sha256:
        raise InvalidLinalgPattern("normalization input hash differs from read source bytes")
    if receipt["output_sha256"] != normalized_sha256:
        raise InvalidLinalgPattern("normalization output hash differs from normalized source bytes")
    operations = tuple(mq.walk(module))
    if any(ordinal >= len(operations) for ordinal in requested):
        raise InvalidLinalgPattern("source ordinal is outside parsed normalized program")
    evidence = tuple((ordinal, recognizer(operations[ordinal])) for ordinal in sorted(requested))
    return raw_sha256, normalized_sha256, evidence


def screen_static_pointwise_source(path: Path, ordinals: tuple[int, ...]) -> StaticPointwiseSource:
    """Bind each proved pointwise body to exact raw/normalized source bytes and ordinal."""
    raw_sha256, normalized_sha256, evidence = _screen_static_linalg_source(path, ordinals, recognize_static_pointwise)
    return StaticPointwiseSource(raw_sha256, normalized_sha256, evidence)


def screen_static_projected_pointwise_source(path: Path, ordinals: tuple[int, ...]) -> StaticProjectedPointwiseSource:
    """Bind proved projected bodies to exact raw/normalized source ordinals."""
    raw, normalized, evidence = _screen_static_linalg_source(path, ordinals, recognize_static_projected_pointwise)
    return StaticProjectedPointwiseSource(raw, normalized, evidence)


def recognize_f32_relu_ordered_select(op) -> tuple[object, object]:
    """Recognize the closed f32 `x < +0 ? +0 : x` scalar region.

    This is a source arithmetic form, not a target ReLU admission. Its compare
    is ordered, so NaNs take the original input arm and negative zero survives.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.ir.affine import AffineMap
    from xdsl.traits import Pure

    shell = _checked_static_indexed_linalg_shell(op)
    rank = len(shell.shape)
    if (
        len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1
        or shell.ordered_types != ("f32", "f32", "f32")
        or shell.input_shapes != (shell.shape,)
        or shell.output_shapes != (shell.shape,)
        or shell.indexing_maps != (AffineMap.identity(rank),) * 2
        or shell.iterator_types != (IteratorType.PARALLEL,) * rank
    ):
        raise InvalidLinalgPattern("f32 zero select needs identity maps and one static f32 input/output")
    body = shell.body
    if tuple(item.name for item in body) != (
        "arith.constant", "arith.sitofp", "arith.cmpf", "arith.select", "linalg.yield",
    ):
        raise InvalidLinalgPattern("f32 zero select needs integer zero, cast, ordered compare and select")
    integer, cast, compare, select, yld = body
    for item, allowed in zip(body[:-1], ({"value"}, set(), {"predicate", "fastmath"}, set()), strict=True):
        if (
            not type(item).has_trait(Pure) or item.regions or item.successors
            or any(not name.startswith("prov.") for name in item.attributes)
            or set(item.properties) - allowed
        ):
            raise InvalidLinalgPattern("f32 zero select has an effect or unknown scalar property")
    zero = integer.properties.get("value")
    predicate = compare.properties.get("predicate")
    flags = compare.properties.get("fastmath")
    if (
        not isinstance(zero, IntegerAttr) or str(zero.type) != "i64" or zero.value.data != 0
        or not isinstance(predicate, IntegerAttr) or str(predicate.type) != "i64"
        or predicate.value.data != 4  # ordered less than
        or flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
        or any(len(item.results) != 1 for item in body[:-1])
        or str(integer.results[0].type) != "i64"
        or str(cast.results[0].type) != "f32" or str(compare.results[0].type) != "i1"
        or str(select.results[0].type) != "f32"
        or tuple(cast.operands) != (integer.results[0],)
        or tuple(compare.operands) != (shell.args[0], cast.results[0])
        or tuple(select.operands) != (compare.results[0], cast.results[0], shell.args[0])
        or type(yld) is not YieldOp or yld.regions or yld.successors or yld.properties
        or any(not name.startswith("prov.") for name in yld.attributes)
        or tuple(yld.operands) != (select.results[0],)
    ):
        raise InvalidLinalgPattern("f32 zero select scalar body differs from ordered less-than and select")
    return op.inputs[0], op.outputs[0]


def recognize_signed_i8_i32_matmul(op) -> tuple:
    """Recognize a rank-two signed i8 matmul with ordered i32 wrap accumulation."""
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("linalg.generic is not a two-input, one-init, one-result contraction")
    lhs, rhs = op.inputs
    init = op.outputs[0]
    values = (lhs, rhs, init, op.results[0])
    if not all(isinstance(value.type, TensorType) for value in values) or [
        str(value.type.element_type) for value in values
    ] != ["i8", "i8", "i32", "i32"]:
        raise InvalidLinalgPattern("linalg.generic is not signed i8 x i8 -> i32 matmul")
    if op.results[0].type != init.type:
        raise InvalidLinalgPattern("linalg.generic result type differs from its initialized output")

    maps = tuple(attribute.data for attribute in op.indexing_maps)
    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    expected = (
        AffineMap(3, 0, (d0, d2)),
        AffineMap(3, 0, (d2, d1)),
        AffineMap(3, 0, (d0, d1)),
    )
    if maps != expected:
        raise InvalidLinalgPattern("linalg.generic indexing maps are not rank-2 matmul maps")
    if tuple(attribute.data for attribute in op.iterator_types) != (
        IteratorType.PARALLEL,
        IteratorType.PARALLEL,
        IteratorType.REDUCTION,
    ):
        raise InvalidLinalgPattern("linalg.generic iterators are not two parallel and one reduction")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("linalg.generic requires a single scalar body block")
    block = op.regions[0].block
    args = list(block.args)
    if len(args) != 3 or [str(arg.type) for arg in args] != ["i8", "i8", "i32"]:
        raise InvalidLinalgPattern("linalg.generic body argument types do not match signed matmul")
    body_ops = list(block.ops)
    if [inner.name for inner in body_ops] != [
        "arith.extsi",
        "arith.extsi",
        "arith.muli",
        "arith.addi",
        "linalg.yield",
    ]:
        raise InvalidLinalgPattern("linalg.generic body is not signed widen, multiply, add, yield")
    ex_lhs, ex_rhs, mul, add, yld = body_ops
    if (list(ex_lhs.operands), list(ex_rhs.operands)) != ([args[0]], [args[1]]):
        raise InvalidLinalgPattern("linalg.generic body does not widen both input elements")
    if [str(ex_lhs.results[0].type), str(ex_rhs.results[0].type)] != ["i32", "i32"]:
        raise InvalidLinalgPattern("linalg.generic body widens to a different integer width")
    if set(mul.operands) != {ex_lhs.results[0], ex_rhs.results[0]} or len(mul.operands) != 2:
        raise InvalidLinalgPattern("linalg.generic body product does not use both widened inputs")
    if set(add.operands) != {mul.results[0], args[2]} or len(add.operands) != 2:
        raise InvalidLinalgPattern("linalg.generic body sum does not use product and accumulator")
    if list(yld.operands) != [add.results[0]]:
        raise InvalidLinalgPattern("linalg.generic body yields a different value")
    for arithmetic in (mul, add):
        if str(arithmetic.results[0].type) != "i32" or arithmetic.overflow_flags.data:
            raise InvalidLinalgPattern("linalg.generic body has an unsupported arithmetic width or overflow flag")
    return lhs, rhs, init
