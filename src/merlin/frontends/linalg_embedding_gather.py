"""Closed typed embedding-copy source proof, conditional on valid table indices.

This identifies the exact captured table argument, maps and extraction. It does
not prove run-time index values, frontend equivalence, or compiled numerics.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from merlin.frontends.linalg_patterns import InvalidLinalgPattern, _checked_static_linalg_shell

SCHEMA = "merlin.static_embedding_gather_body.v1"
OPERATION = "bf16_embedding_gather"
TYPES = ("i64", "bf16", "bf16")


def validate_source_body(declaration):
    if not isinstance(declaration, dict) or declaration != {"schema": SCHEMA, "operation": OPERATION}:
        raise InvalidLinalgPattern("embedding gather requires its exact typed source declaration")
    return dict(declaration)


@dataclass(frozen=True)
class EmbeddingGatherPattern:
    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_maps: tuple[str, ...]
    table_shape: tuple[int, ...]
    table_argument: int


def _plain(op, *, properties=frozenset()):
    if op.regions or op.successors or set(op.properties) != properties or any(
        not name.startswith("prov.") for name in op.attributes
    ):
        raise InvalidLinalgPattern("embedding gather has unrecognized scalar metadata")


def recognize(op):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import NoneAttr, TensorType
    from xdsl.dialects.linalg.ops import IndexOp, YieldOp
    from xdsl.ir import BlockArgument
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    if len(op.inputs) != 1 or len(shell.shape) != 3 or shell.ordered_types != TYPES or len(shell.body) != 4:
        raise InvalidLinalgPattern("embedding gather requires rank-three BF16 output and rank-two i64 indices")
    input_shape = tuple(op.inputs[0].type.get_shape())
    expected_map = AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(1)))
    if input_shape != shell.shape[:2] or shell.input_maps != (expected_map,):
        raise InvalidLinalgPattern("embedding gather index map differs from the output prefix")
    cast, column, extract, yld = shell.body
    if tuple(type(v) for v in shell.body) != (arith.IndexCastOp, IndexOp, tensor.ExtractOp, YieldOp):
        raise InvalidLinalgPattern("embedding gather is not one exact cast/index/extract/yield body")
    _plain(cast)
    _plain(column, properties=frozenset({"dim"}))
    _plain(extract)
    _plain(yld)
    if (
        tuple(cast.operands) != (shell.args[0],)
        or len(cast.results) != 1 or str(cast.results[0].type) != "index"
        or column.operands or len(column.results) != 1 or str(column.results[0].type) != "index"
        or getattr(getattr(column.properties["dim"], "value", None), "data", None) != 2
        or len(extract.operands) != 3 or tuple(extract.operands[1:]) != (cast.results[0], column.results[0])
        or len(extract.results) != 1 or str(extract.results[0].type) != "bf16"
        or tuple(yld.operands) != (extract.results[0],) or yld.results
    ):
        raise InvalidLinalgPattern("embedding gather scalar types, coordinates or selected value differ")
    table = extract.operands[0]
    function = op.parent_op()
    while function is not None and function.name != "func.func":
        function = function.parent_op()
    if (
        function is None or len(function.regions) != 1 or len(function.regions[0].blocks) != 1
        or not isinstance(table, BlockArgument) or table.owner is not function.regions[0].block
        or not isinstance(table.type, TensorType) or str(table.type.element_type) != "bf16"
        or not isinstance(table.type.encoding, NoneAttr)
    ):
        raise InvalidLinalgPattern("embedding gather table is not an original typed function argument")
    table_shape = tuple(table.type.get_shape())
    pattern = EmbeddingGatherPattern(
        OPERATION, shell.shape, TYPES, (input_shape,), tuple(str(v) for v in shell.input_maps),
        table_shape, table.index,
    )
    from dataclasses import asdict
    validate_pattern(asdict(pattern))
    return pattern


def validate_pattern(pattern):
    from xdsl.context import Context
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.parser import Parser

    if not isinstance(pattern, Mapping) or set(pattern) != {
        "operation", "shape", "ordered_types", "input_shapes", "input_maps", "table_shape", "table_argument"
    }:
        raise InvalidLinalgPattern("embedding gather serialized source fields differ")
    shape, inputs, table = pattern["shape"], pattern["input_shapes"], pattern["table_shape"]
    if (
        pattern["operation"] != OPERATION or tuple(pattern["ordered_types"]) != TYPES
        or not isinstance(shape, (tuple, list)) or len(shape) != 3
        or not isinstance(inputs, (tuple, list)) or len(inputs) != 1
        or not isinstance(inputs[0], (tuple, list)) or tuple(inputs[0]) != tuple(shape[:2])
        or not isinstance(table, (tuple, list)) or len(table) != 2 or table[1] != shape[2]
        or any(type(v) is not int or v <= 0 for v in (*shape, *table))
        or type(pattern["table_argument"]) is not int or pattern["table_argument"] < 0
        or not isinstance(pattern["input_maps"], (tuple, list)) or len(pattern["input_maps"]) != 1
    ):
        raise InvalidLinalgPattern("embedding gather serialized types, shapes or table identity differ")
    try:
        mapping = Parser(Context(), pattern["input_maps"][0]).parse_affine_map()
        expected = AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(1)))
        if mapping != expected or str(mapping) != pattern["input_maps"][0]:
            raise InvalidLinalgPattern("embedding gather serialized index map differs")
    except Exception as exc:
        raise InvalidLinalgPattern("embedding gather serialized index map is malformed") from exc
