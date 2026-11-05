"""Keep channel-last storage for exact single-column integer contractions.

Recognize ``reshape(transpose(X)) @ w`` when the transpose merely places
the original trailing channel axis first. Replace it by ``w.T @ reshape(X)``
and a unit-axis output view. The signed modular scalar recurrence and reduction
order are unchanged. This is a source transformation, not a target schedule or
a claim about physical aliasing, execution costs or hardware support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod

from xdsl.dialects import builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.ir import Operation

from merlin.transforms.dense_ir import recognize_signed_integer_matmul


@dataclass
class ReductionLayoutReport:
    rewritten: int = 0
    eliminated_transpose_bytes: int = 0
    preferred_layouts: int = 0

    def to_dict(self):
        return asdict(self)


def _static(value):
    ty = value.type
    if (not isinstance(ty, builtin.TensorType) or ty.encoding != builtin.NoneAttr()
            or not ty.get_shape() or any(d <= 0 for d in ty.get_shape())):
        return None
    return ty


def _pure(op):
    """Do not move reads through opaque effects or coordinate-observing bodies."""
    from merlin.llvmlower.layout_conformance import per_tensor_quantization

    if per_tensor_quantization(op):
        return True
    if op.name in {"tensor.empty", "tensor.splat", "tensor.collapse_shape", "tensor.expand_shape",
                   "arith.constant", "linalg.transpose"}:
        return all(inner.name == "linalg.yield" for region in op.regions for inner in region.walk())
    if op.name != "linalg.generic":
        return False
    if set(op.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        return False
    return all(inner.name.startswith(("arith.", "math.")) or inner.name == "linalg.yield"
               for region in op.regions for inner in region.walk())


def _reshape(value, shape, anchor):
    old = value.type
    if old.get_shape() == tuple(shape):
        return value
    rank = len(old.get_shape())
    flat_type = builtin.TensorType(old.element_type, [prod(shape)])
    collapse = tensor.CollapseShapeOp.create(
        operands=[value], result_types=[flat_type],
        properties={"reassociation": builtin.ArrayAttr([
            builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(rank)])])},
    )
    anchor.parent.insert_op_before(collapse, anchor)
    reassociation = builtin.ArrayAttr([
        builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(len(shape))])])
    result_type = builtin.TensorType(old.element_type, shape)
    expand = tensor.ExpandShapeOp(collapse.results[0], [], reassociation, shape, result_type)
    anchor.parent.insert_op_before(expand, anchor)
    return expand.results[0]


def rewrite_module(module) -> ReductionLayoutReport:
    """Apply only exact, sole-use, static channel-last reduction alternatives.

    Other consumers retain the original transpose. Unknown scalar recurrences,
    nonunit output columns, changed reduction ordering and opaque intervening
    effects remain unchanged. No model, operation provenance or target name
    participates in selection.
    """
    module.verify()
    report = ReductionLayoutReport()
    for op in tuple(module.walk()):
        try:
            lhs, rhs, init = recognize_signed_integer_matmul(op)
        except ValueError:
            continue
        m, k = lhs.type.get_shape()
        if rhs.type.get_shape() != (k, 1) or m <= 1 or k <= 1:
            continue
        chain, value = [], lhs
        while isinstance(value.owner, tensor.CollapseShapeOp | tensor.ExpandShapeOp):
            view = value.owner
            if (view.parent is not op.parent or not value.has_one_use() or len(view.operands) != 1
                    or _static(view.operands[0]) is None or not _pure(view)):
                break
            chain.append(view)
            value = view.operands[0]
        trans = value.owner
        direct = type(trans) is TransposeOp
        if (getattr(trans, 'parent', None) is not op.parent
                or not value.has_one_use() or not _pure(trans)):
            continue
        if direct:
            source = trans.inputs[0]
            ty = _static(source)
            if ty is None:
                continue
            shape = ty.get_shape()
            permutation = tuple(trans.permutation.get_values())
            axes = tuple(i for i, extent in enumerate(shape) if extent != 1)
            expected = (len(shape) - 1, *axes[:-1])
            actual = tuple(i for i in permutation if shape[i] != 1)
            if (len(shape) < 2 or shape[-1] != m or prod(shape[:-1]) != k
                    or actual != expected):
                continue
        else:
            from merlin.llvmlower.layout_propagation import _input_layout

            ty = _static(value)
            permutation = _input_layout(value, op.parent)
            if ty is None or permutation is None:
                continue
            shape = ty.get_shape()
            axes = tuple(i for i, extent in enumerate(shape) if extent != 1)
            channel = permutation[-1]
            if (not axes or channel != axes[0] or shape[channel] != m
                    or prod(shape[i] for i in range(len(shape)) if i != channel) != k
                    or tuple(i for i in permutation if shape[i] != 1) != (*axes[1:], channel)):
                continue
        current = trans.next_op
        while current is not None and current is not op and _pure(current):
            current = current.next_op
        if current is not op:
            continue
        if not direct:
            from merlin.llvmlower.layout_propagation import realize_input_layout

            source, _ = realize_input_layout(module, value, permutation, chain[-1] if chain else op)
            report.preferred_layouts += 1
        left = _reshape(rhs, (1, k), op)
        right = _reshape(source, (k, m), op)
        seed = _reshape(init, (1, m), op)
        replacement = GenericOp(
            [left, right], [seed], op.body.clone(), list(op.indexing_maps), list(op.iterator_types),
            [builtin.TensorType(init.type.element_type, [1, m])],
        )
        replacement.attributes.update(op.attributes)
        # Preserve the alternative selected by this source transformation when
        # stating a device command. Immutable ownership does not prescribe
        # which operand occupies the stationary execution slot.
        replacement.attributes["merlin.preserve_contraction_orientation"] = builtin.UnitAttr()
        op.parent.insert_op_before(replacement, op)
        result = _reshape(replacement.results[0], (m, 1), op)
        op.results[0].replace_all_uses_with(result)
        op.parent.erase_op(op)
        pending = [*chain, trans]
        while pending:
            dead = pending.pop(0)
            if (not isinstance(dead, Operation) or dead.parent is None or not _pure(dead)
                    or any(result.uses for result in dead.results)):
                continue
            pending.extend(operand.owner for operand in dead.operands)
            dead.parent.erase_op(dead)
        report.rewritten += 1
        if direct:
            report.eliminated_transpose_bytes += prod(shape)  # signed i8 input
    module.verify()
    return report
