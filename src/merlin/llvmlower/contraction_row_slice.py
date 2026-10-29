"""Push a sole static row observer through a closed integer-product epilogue.

Only parallel rows disappear. The complete ordered integer dot and every
retained scalar instruction are cloned unchanged, at their original positions.
Static contiguous views preserve row-major coordinates. Immutable coefficients
are sliced, never recomputed or substituted. This supplies no target schedule,
physical alias, floating tolerance or execution qualification.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.ops import FillOp, GenericOp
from xdsl.ir import BlockArgument, Operation
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from merlin.common.mlir_query import op_name

from .quant_round_check import pure_scalar_body

PASS_NAME = "contraction-row-slice"


def selected():
    from .optional_passes import switched

    return switched(PASS_NAME, os.environ.get("MERLIN_CONTRACTION_ROW_SLICE", "0") == "1")


def _shape(value):
    typ = value.type
    if not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr):
        return None
    shape = tuple(typ.get_shape())
    return shape if all(d > 0 for d in shape) else None


def _attribution(op):
    return all(k.startswith("prov.") or k == "op_name__" for k in op.attributes)


def _container(op):
    fn = op.parent_op()
    if (
        not isinstance(fn, func.FuncOp)
        or len(fn.body.blocks) != 1
        or any(k != "llvm.emit_c_interface" and not k.startswith("prov.") for k in fn.attributes)
        or set(fn.properties) - {"sym_name", "sym_visibility", "function_type"}
    ):
        return False
    module = fn.parent_op()
    return (
        isinstance(module, builtin.ModuleOp)
        and not module.properties
        and all(k in {"llvm.data_layout", "llvm.target_triple"} or k.startswith("prov.") for k in module.attributes)
    )


def _interval(shape, offsets, sizes):
    if (
        len(shape) != len(offsets)
        or len(shape) != len(sizes)
        or any(o < 0 or n <= 0 or o + n > d for d, o, n in zip(shape, offsets, sizes))
    ):
        return None
    strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    count = math.prod(sizes)
    if count != 1 + sum((n - 1) * s for n, s in zip(sizes, strides)):
        return None
    return sum(o * s for o, s in zip(offsets, strides)), count


def _rectangle(shape, start, count):
    if not shape or start < 0 or count <= 0 or start + count > math.prod(shape):
        return None
    strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    offsets = tuple((start // s) % d for d, s in zip(shape, strides))
    last = start + count - 1
    sizes = tuple((last // s) % d - o + 1 for d, s, o in zip(shape, strides, offsets))
    return (offsets, sizes) if _interval(shape, offsets, sizes) == (start, count) else None


def _closed_body(op):
    if not pure_scalar_body(op.body.block):
        return False
    # Keep arithmetic value semantics; opaque calls and index-dependent bodies
    # cannot change their domain. Strict/fast floating attributes refuse above.
    for child in op.body.block.ops:
        for operand in child.operands:
            if operand.owner is not op.body.block and getattr(operand.owner, "parent", None) is not op.body.block:
                if not isinstance(operand.owner, arith.ConstantOp) or not _attribution(operand.owner):
                    return False
    return True


def _pointwise(op):
    if (
        not isinstance(op, GenericOp)
        or len(op.inputs) < 1
        or len(op.outputs) != 1
        or len(op.results) != 1
        or not isinstance(op.outputs[0].owner, tensor.EmptyOp)
        or not _attribution(op.outputs[0].owner)
        or op.outputs[0].owner.operands
        or op.body.block.args[-1].uses
        or not _closed_body(op)
        or set(op.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    ):
        return False
    shape = _shape(op.results[0])
    if (
        shape is None
        or op.outputs[0].type != op.results[0].type
        or len(op.iterator_types) != len(shape)
        or any(it.data.value != "parallel" for it in op.iterator_types)
        or op.indexing_maps.data[-1].data != AffineMap.identity(len(shape))
    ):
        return False
    for value, attribute in zip(op.inputs, op.indexing_maps, strict=False):
        source, mapping = _shape(value), attribute.data
        if (
            source is None
            or mapping.num_symbols
            or mapping.num_dims != len(shape)
            or len(mapping.results) != len(source)
        ):
            return False
        seen = set()
        for extent, expression in zip(source, mapping.results):
            if isinstance(expression, AffineDimExpr):
                if (
                    not 0 <= expression.position < len(shape)
                    or expression.position in seen
                    or extent != shape[expression.position]
                ):
                    return False
                seen.add(expression.position)
            elif not isinstance(expression, AffineConstantExpr) or expression.value != 0 or extent != 1:
                return False
    return True


def _fixed(value, loader, depth=0):
    if depth > 32:
        return False
    owner = value.owner
    if isinstance(value, BlockArgument):
        if loader is None or _shape(value) is None:
            return False
        width = getattr(value.type.element_type, "bitwidth", 0)
        size = math.prod(_shape(value)) * width // 8
        return 0 < size <= 1 << 20 and isinstance(raw := loader(value), bytes) and len(raw) == size
    if not isinstance(owner, Operation) or not _attribution(owner):
        return False
    if isinstance(owner, arith.ConstantOp):
        return set(owner.properties) == {"value"} and not owner.operands
    if isinstance(owner, tensor.SplatOp):
        return len(owner.operands) == 1 and not owner.properties and _fixed(owner.input, loader, depth + 1)
    if isinstance(owner, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
        return len(owner.operands) == 1 and _fixed(owner.operands[0], loader, depth + 1)
    return _pointwise(owner) and all(_fixed(v, loader, depth + 1) for v in owner.inputs)


@dataclass
class _Plan:
    op: Operation
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    child: _Plan | None = None
    input_index: int = 0
    geometry: tuple[int, int, int] | None = None


def _plan(value, start, count, consumer, loader, depth=0):
    op, shape = value.owner, _shape(value)
    if (
        depth > 64
        or shape is None
        or not isinstance(op, Operation)
        or not _container(op)
        or not _attribution(op)
        or op.parent is not consumer.parent
        or len(op.results) != 1
        or tuple(use.operation for use in value.uses) != (consumer,)
    ):
        return None
    rectangle = _rectangle(shape, start, count)
    if rectangle is None:
        return None
    offsets, sizes = rectangle
    if isinstance(op, GenericOp) and any(it.data.value == "reduction" for it in op.iterator_types):
        from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul

        try:
            geometry = read_zero_seeded_integer_matmul(op, max_source_elements=1 << 31)
        except ValueError:
            return None
        m, _, n = geometry
        if offsets[1] or sizes[1] != n or sizes[0] >= m:
            return None
        return _Plan(op, offsets, sizes, geometry=geometry)
    if isinstance(op, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
        allowed = {"reassociation"} | ({"static_output_shape"} if isinstance(op, tensor.ExpandShapeOp) else set())
        if len(op.operands) != 1 or set(op.properties) - allowed or _shape(op.operands[0]) is None:
            return None
        child = _plan(op.operands[0], start, count, op, loader, depth + 1)
        return _Plan(op, offsets, sizes, child) if child else None
    if _pointwise(op):
        dynamic = [i for i, v in enumerate(op.inputs) if not _fixed(v, loader)]
        if len(dynamic) != 1:
            return None
        index = dynamic[0]
        if op.indexing_maps.data[index].data != AffineMap.identity(len(shape)):
            return None
        child = _plan(op.inputs[index], start, count, op, loader, depth + 1)
        return _Plan(op, offsets, sizes, child, index) if child else None
    if op_name(op) in {"quant_ext.quantize_per_tensor", "quant_ext.dequantize_per_tensor"}:
        if (
            len(op.operands) != 3
            or op.regions
            or op.successors
            or set(op.properties) - {"quant_min", "quant_max", "output_dtype"}
            or _shape(op.operands[0]) != shape
            or any(_shape(v) != () or not _fixed(v, loader) for v in op.operands[1:])
        ):
            return None
        child = _plan(op.operands[0], start, count, op, loader, depth + 1)
        return _Plan(op, offsets, sizes, child) if child else None
    return None


def _insert(op, before):
    before.parent.insert_op_before(op, before)
    return op.results[0]


def _slice(value, offsets, sizes, before):
    if not offsets or (not any(offsets) and tuple(sizes) == _shape(value)):
        return value
    if isinstance(value.owner, tensor.SplatOp):
        result = tensor.SplatOp(value.owner.input, [], builtin.TensorType(value.type.element_type, sizes))
    elif isinstance(value.owner, FillOp):
        typ = builtin.TensorType(value.type.element_type, sizes)
        empty = _insert(tensor.EmptyOp([], typ), before)
        result = FillOp(list(value.owner.inputs), [empty], [typ])
    else:
        result = tensor.ExtractSliceOp.from_static_parameters(value, offsets, sizes, (1,) * len(sizes))
    return _insert(result, before)


def _reshape(value, shape, before):
    if tuple(value.type.get_shape()) == shape:
        return value

    def groups(rank):
        return builtin.ArrayAttr([builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(rank)])])

    if len(value.type.get_shape()) != 1:
        collapsed = tensor.CollapseShapeOp(
            operands=[value],
            result_types=[builtin.TensorType(value.type.element_type, (math.prod(shape),))],
            properties={"reassociation": groups(len(value.type.get_shape()))},
        )
        collapsed.attributes.update(before.attributes)
        value = _insert(collapsed, before)
    if len(shape) != 1:
        expanded = tensor.ExpandShapeOp(
            value, [], groups(len(shape)), shape, builtin.TensorType(value.type.element_type, shape)
        )
        expanded.attributes.update(before.attributes)
        value = _insert(expanded, before)
    return value


def _apply(plan):
    op = plan.op
    typ = builtin.TensorType(op.results[0].type.element_type, plan.sizes)
    if plan.geometry is not None:
        _, k, _ = plan.geometry
        lhs = _slice(op.inputs[0], (plan.offsets[0], 0), (plan.sizes[0], k), op)
        zero = _slice(op.outputs[0], plan.offsets, plan.sizes, op)
        inputs, outputs = [lhs, op.inputs[1]], [zero]
    else:
        child = _apply(plan.child)
        if isinstance(op, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            return _reshape(child, plan.sizes, op)
        if not isinstance(op, GenericOp):
            return _insert(
                type(op).create(
                    operands=[child, *op.operands[1:]],
                    result_types=[typ],
                    properties=dict(op.properties),
                    attributes=dict(op.attributes),
                ),
                op,
            )
        inputs = []
        for i, (value, attribute) in enumerate(zip(op.inputs, op.indexing_maps, strict=False)):
            if i == plan.input_index:
                inputs.append(child)
                continue
            offsets, sizes = [], []
            for expression in attribute.data.results:
                axis = expression.position if isinstance(expression, AffineDimExpr) else None
                offsets.append(0 if axis is None else plan.offsets[axis])
                sizes.append(1 if axis is None else plan.sizes[axis])
            inputs.append(_slice(value, offsets, sizes, op))
        outputs = [_insert(tensor.EmptyOp([], typ), op)]
    new = GenericOp(inputs, outputs, op.body.clone(), op.indexing_maps, op.iterator_types, result_types=[typ])
    new.attributes.update(op.attributes)
    return _insert(new, op)


def pushback_contraction_row_slices(module, *, immutable_argument_loader=None, max_rewrites=64):
    """Return a private transformed clone and source-only observations.

    The optional loader supplies exact original immutable argument bytes; it
    establishes no ownership by itself. No coefficient words are specialized.
    Unknown, dynamic, shared, noncontiguous or column-truncating observers retain
    their complete original computation. This pass runs before device binding.
    """
    if type(max_rewrites) is not int or max_rewrites < 0:
        raise ValueError("row slicing requires a nonnegative rewrite budget")
    module.verify()
    result, rows = module.clone(), []
    for observer in tuple(result.walk()):
        if len(rows) >= max_rewrites:
            break
        if (
            not isinstance(observer, tensor.ExtractSliceOp)
            or not _container(observer)
            or not _attribution(observer)
            or len(observer.operands) != 1
            or len(observer.results) != 1
            or _shape(observer.source) is None
            or set(observer.properties) - {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
        ):
            continue
        offsets = tuple(observer.static_offsets.get_values())
        sizes = tuple(observer.static_sizes.get_values())
        if tuple(observer.static_strides.get_values()) != (1,) * len(sizes) or _shape(observer.result) != sizes:
            continue
        interval = _interval(_shape(observer.source), offsets, sizes)
        if interval is None:
            continue
        plan = _plan(observer.source, *interval, observer, immutable_argument_loader)
        if plan is None:
            continue
        value, chain = _apply(plan), []
        value.owner.attributes["prov.row_slice_observer"] = builtin.DictionaryAttr(dict(observer.attributes))
        cursor = plan
        while cursor is not None:
            chain.append(cursor)
            cursor = cursor.child
        root = chain[-1]
        observer.result.replace_all_uses_with(value)
        observer.parent.erase_op(observer)
        for item in chain:
            item.op.parent.erase_op(item.op)
        rows.append(
            {
                "original_m_k_n": list(root.geometry),
                "selected_m_k_n": [root.sizes[0], root.geometry[1], root.geometry[2]],
                "row_offset": root.offsets[0],
                "epilogue_operations": len(chain) - 1,
                "source_scalar_bodies_unchanged": True,
                "ordered_reduction_unchanged": True,
            }
        )
    result.verify()
    return result, {"schema": "merlin.contraction_row_slice.v1", "rewrites": len(rows), "regions": rows}


def prepare_contraction_row_slices(module, *, bundle_dir=None):
    """Ordinary explicit registry selection before complete-group routing.

    Generated-caller ownership distinguishes fixed stored coefficients from
    session state. Source code and coefficient values are never substituted.
    Source-projected routes must select this API before constructing their new
    source owner; a late transform cannot change an already bound device ABI.
    """
    if not selected():
        return module, {"schema": "merlin.contraction_row_slice.v1", "rewrites": 0, "regions": []}
    loader, bound = None, []
    if bundle_dir is not None:
        from merlin.capture.safetensors import load_header

        from .weight_prepack_runtime import fixed_arguments
        from .weights_pack import weight_files_beside

        manifest, weights = weight_files_beside(bundle_dir)
        if manifest is not None and weights is not None:
            records = json.loads(manifest.read_text())
            fixed = fixed_arguments(bundle_dir, records)
            header, payload = load_header(weights)

            def loader(argument):
                if argument.index not in fixed or argument.owner.parent_op().sym_name.data != "forward":
                    return None
                record = records[str(argument.index)]
                metadata = header.get(record["weight"])
                if metadata is None or tuple(metadata["shape"]) != _shape(argument):
                    return None
                begin, end = metadata["data_offsets"]
                if not 0 <= begin <= end or end - begin > 1 << 20:
                    return None
                with weights.open("rb") as handle:
                    handle.seek(payload + begin)
                    raw = handle.read(end - begin)
                bound.append(argument.index)
                return raw

    result, report = pushback_contraction_row_slices(module, immutable_argument_loader=loader)
    report["immutable_generated_argument_indices"] = sorted(set(bound))
    return result, report
