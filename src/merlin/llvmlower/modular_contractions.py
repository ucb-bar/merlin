"""Exactly widen wrapping integer contractions, retaining a truncating host readout.

For signless n-bit multiply/add without overflow promises, every operation is in
Z/(2**n). Sign extension into w >= n bits preserves residues modulo 2**n;
truncation is a ring homomorphism. Thus widening the entire from-zero reduction
and truncating its result preserves every original bit, even if the wider sum
itself overflows. This is not saturation, quantization or an approximate rewrite.

The caller supplies eligible datapath triples. No precision or target is assumed.
Only a closed multiply/add/yield body and proven zero initialization are admitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _closed_wrapping_body(op, element) -> bool:
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import YieldOp

    if not hasattr(op, "body") or len(op.body.blocks) != 1:
        return False
    block = op.body.block
    if len(block.args) != 3 or any(arg.type != element for arg in block.args):
        return False
    operations = list(block.ops)
    if len(operations) != 3:
        return False
    mul, add, yielded = operations
    if not isinstance(mul, arith.MuliOp) or not isinstance(add, arith.AddiOp) or not isinstance(yielded, YieldOp):
        return False
    if mul.result.type != element or add.result.type != element:
        return False
    if mul.overflow_flags.data or add.overflow_flags.data:
        return False
    if any(not key.startswith("prov.") for item in (mul, add) for key in item.attributes):
        return False
    if any(key != "overflowFlags" for item in (mul, add) for key in item.properties):
        return False
    lhs, rhs, init = block.args
    return (
        tuple(mul.operands) in ((lhs, rhs), (rhs, lhs))
        and tuple(add.operands) in ((mul.result, init), (init, mul.result))
        and tuple(yielded.operands) == (add.result,)
    )


def widen_module(module, triples) -> int:
    """Mutate only proved modular contractions that a supplied wider datapath can implement."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        IntegerAttr,
        IntegerType,
        NoneAttr,
        Signedness,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.kernels.shapes import observe_contractions, zero_initialised

    count = 0
    for op, shape in list(observe_contractions(module)):
        if len(op.results) != 1 or len(op.operands) != 3 or not zero_initialised(op):
            continue
        if any(
            not isinstance(value.type, TensorType) or not isinstance(value.type.encoding, NoneAttr)
            for value in op.operands
        ):
            continue
        result_type = op.results[0].type
        if not isinstance(result_type, TensorType):
            continue
        element = result_type.get_element_type()
        if not isinstance(element, IntegerType) or element.signedness.data != Signedness.SIGNLESS:
            continue
        token = str(element)
        if tuple(shape.dtypes) != (token, token, token) or not _closed_wrapping_body(op, element):
            continue
        if getattr(op, "library_call", None) is not None:
            continue
        wider = set()
        for lhs, rhs, accum in triples:
            if lhs == rhs == token and accum.startswith("i") and accum[1:].isdecimal():
                width = int(accum[1:])
                if width > element.width.data:
                    wider.add(width)
        if len(wider) != 1:
            continue  # ambiguous precision is a refusal, never a choice of the first datapath
        wide = IntegerType(wider.pop())
        wide_type = TensorType(wide, result_type.get_shape())
        zero = arith.ConstantOp(IntegerAttr(0, wide))
        init = tensor.SplatOp(zero.result, [], wide_type)
        body = Block(arg_types=[element, element, wide])
        lhs = arith.ExtSIOp(body.args[0], wide)
        rhs = arith.ExtSIOp(body.args[1], wide)
        mul = arith.MuliOp(lhs.result, rhs.result)
        add = arith.AddiOp(mul.result, body.args[2])
        body.add_ops([lhs, rhs, mul, add, L.YieldOp(add.result)])
        contraction = L.GenericOp(
            inputs=op.operands[:2],
            outputs=[init.result],
            body=Region(body),
            indexing_maps=op.get_indexing_maps(),
            iterator_types=op.get_iterator_types(),
            result_types=[wide_type],
        )
        contraction.attributes.update(op.attributes)
        contraction.attributes["merlin.modular_widening"] = StringAttr(f"{element}->{wide}")
        empty = tensor.EmptyOp([], result_type)
        readout = Block(arg_types=[wide, element])
        truncate = arith.TruncIOp(readout.args[0], element)
        readout.add_ops([truncate, L.YieldOp(truncate.result)])
        rank = len(result_type.get_shape())
        identity = AffineMapAttr(AffineMap.identity(rank))
        result = L.GenericOp(
            inputs=[contraction.results[0]],
            outputs=[empty.tensor],
            body=Region(readout),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * rank),
            result_types=[result_type],
        )
        result.attributes.update({key: value for key, value in op.attributes.items() if key.startswith("prov.")})
        result.attributes["prov.role"] = StringAttr("modular_readout")
        Rewriter.replace_op(op, [zero, init, contraction, empty, result], [result.results[0]])
        count += 1
    if count:
        module.verify()
    return count


def prepare_bundle(directory: Path, triples) -> dict:
    """Preserve original bundle bytes and write an attributed, equivalent wider source when admitted."""
    from merlin.frontends.linalg_mlir import parse_mlir_file

    path = directory / "model.mlir"
    source = path.read_bytes()
    module = parse_mlir_file(path)
    count = widen_module(module, triples)
    if not count:
        return {"widened_contractions": 0}
    original = directory / "model.before_modular_widening.mlir"
    original.write_bytes(source)
    path.write_text(str(module) + "\n", encoding="utf-8")
    receipt = {
        "schema": "modular_contraction_widening_v1",
        "widened_contractions": count,
        "original": str(original),
        "original_sha256": hashlib.sha256(source).hexdigest(),
        "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "numeric_contract": "exact wrapping multiply/add followed by truncation; no saturation",
    }
    (directory / "modular_widening.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
