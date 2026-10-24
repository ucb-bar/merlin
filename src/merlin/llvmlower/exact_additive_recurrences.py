"""Expose closed resetting additive scans under an exact rational contract.

Index-only resets and a single additive backedge define a static linear map.
IEEE ordered accumulation is deliberately outside this proof.
"""


def expose_exact_additive_recurrences(module, *, quantized_contract=None):
    if quantized_contract is None:
        return 0
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import PARALLEL, REDUCTION, dims, generic
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    changed = 0
    for loop in list(module.walk()):
        if loop.name != "scf.for" or len(loop.results) != 2 or loop.results[1].uses:
            continue
        if len(loop.operands) != 5 or len(loop.body.block.args) != 3:
            continue
        if _constant_of(loop.operands[0]) != 0 or _constant_of(loop.operands[2]) != 1:
            continue
        n = _constant_of(loop.operands[1])
        if not isinstance(n, int) or n <= 0:
            continue
        block = loop.body.block
        iv, output, accumulator = block.args
        if any(not isinstance(v.type, TensorType) for v in loop.results):
            continue
        if loop.results[0].type.get_shape() != (n,) or loop.results[1].type.get_shape() != (n,):
            continue
        element, accum_type = loop.results[0].type.element_type, loop.results[1].type.element_type
        if str(element) not in {"f16", "bf16", "f32"} or str(accum_type) not in {"f32", "f64"}:
            continue
        children = list(block.ops)
        final = children[-1]
        if final.name != "scf.yield" or len(final.operands) != 2:
            continue
        inserts = [v.owner for v in final.operands]
        if any(getattr(op, "name", None) != "tensor.insert" for op in inserts):
            continue
        out_insert, acc_insert = inserts
        if list(out_insert.operands[1:]) != [output, iv] or list(acc_insert.operands[1:]) != [accumulator, iv]:
            continue
        addition = acc_insert.operands[0].owner
        if getattr(addition, "name", None) != "arith.addf" or addition.fastmath.data:
            continue
        result = out_insert.operands[0]
        narrowing = None
        if result is not addition.result:
            narrowing = result.owner
            if getattr(narrowing, "name", None) != "arith.truncf" or list(narrowing.operands) != [addition.result]:
                continue
        branch = next((v.owner for v in addition.operands if getattr(v.owner, "name", None) == "scf.if"), None)
        if branch is None or len(branch.results) != 1 or len(branch.regions) != 2:
            continue
        current = next(v for v in addition.operands if v is not branch.results[0])
        widening = None
        if getattr(current.owner, "name", None) == "arith.extf":
            widening = current.owner
            current = widening.operands[0]
        extract = current.owner
        if getattr(extract, "name", None) != "tensor.extract" or list(extract.operands[1:]) != [iv]:
            continue
        source = extract.operands[0]
        if source.type != loop.results[0].type:
            continue
        predicate = branch.operands[0].owner
        if getattr(predicate, "name", None) != "arith.cmpi" or predicate.predicate.value.data != 0:
            continue
        remainder = predicate.operands[0].owner
        if _constant_of(predicate.operands[1]) != 0 or getattr(remainder, "name", None) != "arith.remui":
            continue
        division = remainder.operands[0].owner
        if getattr(division, "name", None) != "arith.divui" or division.operands[0] is not iv:
            continue
        stride, extent = _constant_of(division.operands[1]), _constant_of(remainder.operands[1])
        if not isinstance(stride, int) or not isinstance(extent, int) or stride <= 0 or extent <= 0:
            continue
        if n % (stride * extent):
            continue
        reset, previous = [list(region.block.ops) for region in branch.regions]
        if len(reset) != 1 or reset[0].name != "scf.yield" or _constant_of(reset[0].operands[0]) != 0:
            continue
        if len(previous) != 3 or [op.name for op in previous] != ["arith.subi", "tensor.extract", "scf.yield"]:
            continue
        decrement, prior, yielded = previous
        flags = decrement.properties.get("overflowFlags")
        if flags is not None and str(flags) != "#arith.overflow<none>":
            continue
        if decrement.operands[0] is not iv or _constant_of(decrement.operands[1]) != stride:
            continue
        if list(prior.operands) != [accumulator, decrement.result] or list(yielded.operands) != [prior.result]:
            continue
        expected = {extract, division, remainder, predicate, branch, addition, out_insert, acc_insert, final}
        expected.update(op for op in (narrowing, widening) if op is not None)
        if set(children) != expected:
            continue
        weights = [
            float(j <= i and (i - j) % stride == 0 and (i - j) // stride <= (i // stride) % extent)
            for j in range(n)
            for i in range(n)
        ]
        coefficient = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(element, [n, n]), weights))
        zero = arith.ConstantOp(FloatAttr(0.0, element))
        seed = tensor.SplatOp(zero.result, [], loop.results[0].type)
        scalar = Block(arg_types=[element, element, element])
        product = arith.MulfOp(scalar.args[0], scalar.args[1])
        total = arith.AddfOp(product.result, scalar.args[2])
        scalar.add_ops([product, total, L.YieldOp(total.result)])
        maps = [dims(2, axes) for axes in [(1,), (1, 0), (0,)]]
        contraction = generic([source, coefficient.result], seed.result, scalar, maps, [PARALLEL, REDUCTION])
        contraction.attributes.update(loop.attributes)
        # The second result is proved dead; its original empty destination is
        # only a type-correct placeholder while replacing the two-result loop.
        Rewriter.replace_op(loop, [coefficient, zero, seed, contraction], [contraction.results[0], loop.operands[4]])
        changed += 1
    return changed
