"""Exact modular normalization of uint8×int8 dots into signed byte arithmetic.

For every byte u, signed(u xor 128) = unsigned(u) - 128. Consequently
dot_u8s8(u,b) = dot_s8s8(u xor 128,b) + 128*column_sum_s8(b), modulo 2**32.
The source must express unsigned extension explicitly. This pass never guesses
unsigned semantics for a signed dot or repairs a contradictory source capture.
"""

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, StringAttr, TensorType, i8, i32
from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.rewriter import Rewriter

from merlin.kernels.shapes import zero_initialised


def byte_dot_kind(op) -> str | None:
    """Prove a closed canonical byte dot's scalar body, maps, types and wiring."""
    if not isinstance(op, GenericOp) or len(op.operands) != 3 or len(op.results) != 1:
        return None
    if set(op.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        return None
    maps = [str(value) for value in op.indexing_maps]
    if maps != [
        "affine_map<(d0, d1, d2) -> (d0, d2)>",
        "affine_map<(d0, d1, d2) -> (d2, d1)>",
        "affine_map<(d0, d1, d2) -> (d0, d1)>",
    ]:
        return None
    if [str(value) for value in op.iterator_types] != [
        "#linalg.iterator_type<parallel>",
        "#linalg.iterator_type<parallel>",
        "#linalg.iterator_type<reduction>",
    ]:
        return None
    types = [value.type for value in [*op.operands, *op.results]]
    if any(not isinstance(t, TensorType) or t.get_num_dims() != 2 for t in types):
        return None
    if [t.element_type for t in types] != [i8, i8, i32, i32]:
        return None
    a, b, c, out = [tuple(t.get_shape()) for t in types]
    if any(dim <= 0 for shape in (a, b, c, out) for dim in shape) or a[1] != b[0] or c != out or c != (a[0], b[1]):
        return None
    block = op.body.block
    body = list(block.ops)
    if len(block.args) != 3 or len(body) != 5:
        return None
    ea, eb, multiply, add, yielded = body
    if (
        ea.name not in ("arith.extsi", "arith.extui")
        or eb.name != "arith.extsi"
        or multiply.name != "arith.muli"
        or add.name != "arith.addi"
        or yielded.name != "linalg.yield"
        or list(ea.operands) != [block.args[0]]
        or list(eb.operands) != [block.args[1]]
        or set(multiply.operands) != {ea.results[0], eb.results[0]}
        or set(add.operands) != {block.args[2], multiply.results[0]}
        or list(yielded.operands) != [add.results[0]]
    ):
        return None
    if [arg.type for arg in block.args] != [i8, i8, i32] or any(
        len(inner.results) != 1 or inner.results[0].type != i32 for inner in (ea, eb, multiply, add)
    ):
        return None
    for inner in body:
        for key, value in {**inner.attributes, **inner.properties}.items():
            if not key.startswith("prov.") and not (
                inner in (multiply, add) and key == "overflowFlags" and str(value) == "#arith.overflow<none>"
            ):
                return None
    return "u8s8" if ea.name == "arith.extui" else "s8s8"


def _generic(inputs, output, block, maps, iterators):
    return GenericOp(
        inputs=inputs,
        outputs=[output],
        body=Region(block),
        indexing_maps=[AffineMapAttr(value) for value in maps],
        iterator_types=[IteratorTypeAttr(value) for value in iterators],
        result_types=[output.type],
    )


def normalize_unsigned_byte_dots(module) -> int:
    """Rewrite proved zero-initialized u8×s8→i32 dots; other arithmetic is untouched."""
    count = 0
    d = AffineExpr.dimension
    par, red = IteratorType.PARALLEL, IteratorType.REDUCTION
    identity = AffineMap(2, 0, (d(0), d(1)))
    for op in list(module.walk()):
        if byte_dot_kind(op) != "u8s8" or not zero_initialised(op):
            continue
        a, b, init = op.operands
        n = b.type.get_shape()[1]
        shifted_empty = tensor.EmptyOp([], a.type)
        block = Block(arg_types=[i8, i8])
        high_bit = arith.ConstantOp(IntegerAttr(-128, i8), i8)
        shifted = arith.XOrIOp(block.args[0], high_bit.result)
        block.add_ops([high_bit, shifted, YieldOp(shifted.result)])
        shift = _generic([a], shifted_empty.tensor, block, [identity, identity], [par, par])

        block = Block(arg_types=[i8, i8, i32])
        ea, eb = arith.ExtSIOp(block.args[0], i32), arith.ExtSIOp(block.args[1], i32)
        product = arith.MuliOp(ea.result, eb.result)
        total = arith.AddiOp(block.args[2], product.result)
        block.add_ops([ea, eb, product, total, YieldOp(total.result)])
        signed = _generic(
            [shift.results[0], b], init, block, [value.data for value in op.indexing_maps], [par, par, red]
        )
        signed.attributes.update(op.attributes)
        signed.attributes["prov.int_mm_source_lhs_dtype"] = StringAttr("uint8")
        signed.attributes["prov.int_mm_lhs_dtype"] = StringAttr("int8")
        signed.attributes["prov.rewrite"] = StringAttr("u8s8_to_s8s8_modular")

        zero = arith.ConstantOp(IntegerAttr(0, i32), i32)
        column_init = tensor.SplatOp(zero.result, [], TensorType(i32, [n]))
        block = Block(arg_types=[i8, i32])
        widened = arith.ExtSIOp(block.args[0], i32)
        column_total = arith.AddiOp(block.args[1], widened.result)
        block.add_ops([widened, column_total, YieldOp(column_total.result)])
        columns = _generic([b], column_init.result, block, [identity, AffineMap(2, 0, (d(1),))], [red, par])

        corrected_empty = tensor.EmptyOp([], op.results[0].type)
        block = Block(arg_types=[i32, i32, i32])
        offset = arith.ConstantOp(IntegerAttr(128, i32), i32)
        compensation = arith.MuliOp(block.args[1], offset.result)
        corrected = arith.AddiOp(block.args[0], compensation.result)
        block.add_ops([offset, compensation, corrected, YieldOp(corrected.result)])
        output = _generic(
            [signed.results[0], columns.results[0]],
            corrected_empty.tensor,
            block,
            [identity, AffineMap(2, 0, (d(1),)), identity],
            [par, par],
        )
        output.attributes.update(
            {
                key: value
                for key, value in op.attributes.items()
                if key.startswith("prov.") and key not in ("prov.op", "prov.aten", "prov.family")
            }
        )
        output.attributes["prov.rewrite"] = StringAttr("u8s8_to_s8s8_modular_compensation")
        Rewriter.replace_op(
            op,
            [shifted_empty, shift, signed, zero, column_init, columns, corrected_empty, output],
            new_results=output.results,
        )
        count += 1
    return count
