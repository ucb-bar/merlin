"""Keep uniform linalg inputs rank zero instead of materializing full splats."""

from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.dialects.tensor import SplatOp
from xdsl.ir.affine import AffineMap


def scalarize_splat_inputs(module) -> int:
    """Fold splat inputs when a static identity output establishes the loop domain.

    The output guard matters: removing a shaped operand must never remove the
    only source of an iteration extent. Rank-zero tensors carry an empty indexing map and
    preserve the region argument type and value for every original iteration.
    """
    count = 0
    for op in list(module.walk()):
        if not isinstance(op, GenericOp):
            continue
        maps = list(op.indexing_maps)
        rank = len(op.iterator_types)
        if not all(it.data.value == "parallel" for it in op.iterator_types):
            continue
        identity = AffineMap.identity(rank)
        if not any(
            isinstance(out.type, TensorType)
            and all(d >= 0 for d in out.type.get_shape())
            and maps[len(op.inputs) + i].data == identity
            for i, out in enumerate(op.outputs)
        ):
            continue
        operands = list(op.operands)
        erased = []
        changed = False
        for i, value in enumerate(op.inputs):
            splat = value.owner
            if not isinstance(splat, SplatOp):
                continue
            # Native linalg-specialize-generic-ops in supported toolchains
            # assumes shaped operands and crashes on a bare scalar input.
            # A rank-zero tensor retains one value with the same empty map.
            if isinstance(value.type, TensorType) and len(value.type.get_shape()) == 0:
                continue
            uniform = SplatOp(splat.input, [], TensorType(splat.input.type, []))
            assert op.parent is not None
            op.parent.insert_op_before(uniform, op)
            operands[i] = uniform.result
            maps[i] = AffineMapAttr(AffineMap(rank, 0, ()))
            erased.append(splat)
            count += 1
            changed = True
        if changed:
            op.operands = operands
            op.properties["indexing_maps"] = ArrayAttr(maps)
            for splat in erased:
                if not splat.result.uses and splat.parent is not None:
                    splat.parent.erase_op(splat)
    return count
