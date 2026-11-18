"""Materialize admitted Interface arithmetic as ordinary structured compiler IR.

This is lowering after the canonical boundary, not source matching or placement.
Scalar/vector providers can then use their existing structured-IR passes. No
target operation, resource choice, rounding change or physical ABI is selected.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, func, tensor
from xdsl.dialects.builtin import AffineMapAttr, DictionaryAttr, ModuleOp, NoneAttr, StringAttr, TensorType, i8, i32
from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
from xdsl.dialects.linalg.ops import FillOp, GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.printer import Printer

from merlin.compiler.dialects import interface
from merlin.compiler.ir.interface_contract import NUMERICAL_ATTRIBUTE, InterfaceConversionError, validate_interface


def _raw_matmul(module, function):
    validate_interface(module, function=function)
    roots = tuple(module.body.block.ops)
    if len(roots) != 1 or not isinstance(roots[0], func.FuncOp) or len(roots[0].body.blocks) != 1:
        raise InterfaceConversionError("structured Interface lowering requires one complete single-block function")
    entry = roots[0]
    operations = tuple(entry.body.block.ops)
    if len(operations) != 3:
        raise InterfaceConversionError("structured Interface lowering implements the complete matmul/commit/return chain")
    matmul, commit, returned = operations
    if not isinstance(matmul, interface.MatmulOp) or not isinstance(commit, interface.CommitOp) or not isinstance(returned, func.ReturnOp):
        raise InterfaceConversionError("structured Interface lowering has no implementation for this operation chain")
    if (
        len(entry.body.block.args) != 2
        or {matmul.lhs, matmul.rhs} != set(entry.body.block.args)
        or commit.acc is not matmul.acc
        or tuple(returned.operands) != (commit.out,)
        or tuple(entry.function_type.outputs.data) != (commit.out.type,)
        or matmul.properties
        or set(commit.properties) - {"epilogue", "output_dtype"}
        or tuple(commit.epilogue)
        or commit.output_dtype is None
        or commit.output_dtype.data != "i32"
    ):
        raise InterfaceConversionError("structured Interface route requires a raw i32 commit and unchanged complete ports")
    types = (matmul.lhs.type, matmul.rhs.type, commit.out.type)
    if any(
        not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr) or any(d <= 0 for d in ty.get_shape())
        for ty in types
    ) or tuple(ty.element_type for ty in types) != (i8, i8, i32):
        raise InterfaceConversionError("structured Interface route requires ordinary positive static i8/i8/i32 tensors")
    lhs, rhs, output = (tuple(ty.get_shape()) for ty in types)
    if len(lhs) == len(rhs) == len(output) == 2:
        shape_ok = lhs[1] == rhs[0] and output == (lhs[0], rhs[1])
        maps = ((0, 2), (2, 1), (0, 1))
    elif len(lhs) == len(rhs) == len(output) == 3:
        shape_ok = lhs[0] == rhs[0] and lhs[2] == rhs[1] and output == (lhs[0], lhs[1], rhs[2])
        maps = ((0, 1, 3), (0, 3, 2), (0, 1, 2))
    else:
        raise InterfaceConversionError("structured Interface matmul supports dense rank two or equal-batch rank three")
    expected = {
        "lhs_signedness": "signed", "rhs_signedness": "signed", "accumulator_initialization": "zero",
        "accumulator_overflow": "wrap", "layout": "dense_row_major",
        "contraction": "matmul" if len(lhs) == 2 else "batch_matmul",
    }
    numerical = matmul.attributes.get(NUMERICAL_ATTRIBUTE)
    if (
        not shape_ok
        or matmul.acc.type.element != commit.out.type
        or not isinstance(numerical, DictionaryAttr)
        or numerical.data != {key: StringAttr(value) for key, value in expected.items()}
    ):
        raise InterfaceConversionError("structured lowering cannot preserve the declared Interface matmul semantics")
    return entry, matmul, commit, returned, maps


def materialize_interface(module: ModuleOp, *, function: str | None = None) -> ModuleOp:
    """Return a fresh structured module with exactly the same tensor signature."""
    _raw_matmul(module, function)
    converted = module.clone()
    entry, matmul, commit, _, maps = _raw_matmul(converted, function)
    body = entry.body.block
    zero = arith.ConstantOp.from_int_and_width(0, i32)
    empty = tensor.EmptyOp((), commit.out.type)
    fill = FillOp(inputs=(zero.result,), outputs=(empty.tensor,), res=(commit.out.type,))
    scalar = Block(arg_types=[i8, i8, i32])
    left = arith.ExtSIOp(scalar.args[0], i32)
    right = arith.ExtSIOp(scalar.args[1], i32)
    product = arith.MuliOp(left.result, right.result)
    total = arith.AddiOp(product.result, scalar.args[2])
    scalar.add_ops([left, right, product, total, YieldOp(total.result)])
    loop_rank = len(commit.out.type.get_shape()) + 1
    contraction = GenericOp(
        inputs=(matmul.lhs, matmul.rhs), outputs=(fill.results[0],), body=Region(scalar),
        indexing_maps=[AffineMapAttr(AffineMap(loop_rank, 0, tuple(AffineDimExpr(axis) for axis in row))) for row in maps],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * (loop_rank - 1) + [IteratorTypeAttr(IteratorType.REDUCTION)],
        result_types=(commit.out.type,),
    )
    contraction.attributes.update(matmul.attributes)
    contraction.location = matmul.location
    zero.location = empty.location = fill.location = matmul.location
    body.insert_ops_before([zero, empty, fill, contraction], matmul)
    commit.out.replace_all_uses_with(contraction.results[0])
    body.erase_op(commit)
    body.erase_op(matmul)
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    converted.attributes["merlin.interface_materialization"] = StringAttr(json.dumps({
        "schema": "merlin.interface_materialization.v1",
        "input_sha256": hashlib.sha256(stream.getvalue().encode()).hexdigest(),
        "implementation": "signed i8 extension, wrapping i32 product/sum, exact zero fill, raw i32 commit",
        "physical_abi": "unchanged tensor signature; target lowering remains required",
        "execution": "not_run",
    }, sort_keys=True))
    converted.verify()
    return converted


@dataclass(frozen=True)
class InterfaceMaterializationPass(ModulePass):
    name = "lower-interface-to-structured"
    function: str | None = None

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        converted = materialize_interface(op, function=self.function)
        for operation in reversed(list(op.body.block.ops)):
            op.body.block.detach_op(operation)
            operation.erase()
        for operation in list(converted.body.block.ops):
            converted.body.block.detach_op(operation)
            op.body.block.add_op(operation)
        op.attributes.clear()
        op.attributes.update(converted.attributes)
