"""Host-owned extraction and independent evaluation of a small actual source chain.

This is a numerical mechanism witness, not a timing capsule or whole-layer surrogate.
Scalar regions and indexing maps are cloned from source; only statically related tensor
extents are reduced. Unsupported operations abstain instead of borrowing candidate semantics.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from itertools import product
from typing import Any, Sequence

# ---- host_source_witness -----------------------------------------------------------------------


def extract_pointwise_reduction(
    source_text: str, source_indices: Sequence[int], *, max_extent: int = 3
) -> tuple[str, dict[str, Any]]:
    """Clone a sole-use pointwise producer and its full-domain scalar reduction.

    Actual affine maps, typed scalar bodies, reduction dimensions and constant initializers are
    retained. Each input-domain element is visited once in source dimension order.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.analysis.optimization.witnesses.gathers import _info, _props
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if not 1 <= max_extent <= 8:
        raise ValueError("reduction witness extent must lie in [1,8]")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1:
        raise ValueError("reduction witness requires one source function")
    ops = [op for op in functions[0].body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("reduction source indices outside source function")
    options = []
    supported = {
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "linalg.yield",
    }
    for index in sorted(permitted):
        reduction = ops[index]
        if reduction.name != "linalg.reduce" or len(reduction.operands) != 2 or len(reduction.results) != 1:
            continue
        producer = reduction.operands[0].owner
        info = _info(producer) if producer in positions else None
        if (
            not info
            or positions[producer] not in permitted
            or len(list(producer.results[0].uses)) != 1
            or any(op.name not in supported or op.regions for op in reduction.regions[0].blocks[0].ops)
        ):
            continue
        dims = tuple(_props(reduction, "dimensions").get_values())
        shape = info[0]
        output_shape = tuple(extent for axis, extent in enumerate(shape) if axis not in dims)
        if (
            not dims
            or len(set(dims)) != len(dims)
            or any(not 0 <= dim < len(shape) for dim in dims)
            or tuple(reduction.results[0].type.get_shape()) != output_shape
            or tuple(reduction.operands[1].type.get_shape()) != output_shape
            or str(reduction.results[0].type.get_element_type()) not in {"f32", "i8", "i16", "i32", "i64"}
        ):
            continue
        options.append((math.prod(shape), positions[producer], index, dims))
    if not options:
        raise ValueError("no sole-use pointwise-to-full-domain reduction source pair")
    _, producer_index, reduction_index, dims = max(options)
    producer, reduction = ops[producer_index], ops[reduction_index]
    shape, maps, _ = _info(producer)
    reduced = tuple(min(extent, max_extent) for extent in shape)
    permutation = [expr.position for expr in maps[-1].results]
    bounds = [reduced[permutation.index(dim)] for dim in range(len(shape))]
    shapes = {
        producer.results[0]: reduced,
        reduction.results[0]: tuple(extent for axis, extent in enumerate(reduced) if axis not in dims),
    }
    for value, amap in zip(producer.operands, maps):
        needed = tuple(
            bounds[expr.position] if isinstance(expr, AffineDimExpr) else expr.value + 1 for expr in amap.results
        )
        if any(size <= 0 or size > old for size, old in zip(needed, value.type.get_shape())):
            raise ValueError("pointwise map does not admit reduced extents")
        if value in shapes and shapes[value] != needed:
            raise ValueError("aliased pointwise inputs have inconsistent reduced shape constraints")
        shapes[value] = needed
    shapes[reduction.operands[1]] = shapes[reduction.results[0]]
    keep = {producer, reduction}
    pending = list(keep)
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                if (
                    owner in positions
                    and owner not in keep
                    and owner.name in {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"}
                ):
                    keep.add(owner)
                    pending.append(owner)
    kept = [op for op in ops if op in keep]
    for op in reversed(kept):
        if op.name == "linalg.fill" and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
    produced = {value for op in kept for value in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if (
                    (value.owner is functions[0].body.block or value.owner in positions)
                    and value not in produced
                    and value not in boundary
                ):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) for value in boundary):
        raise ValueError("non-tensor reduction boundary is unsupported")

    def value_type(value):
        if not isinstance(value.type, TensorType):
            return value.type
        if value not in shapes:
            raise ValueError("reduction source tensor has unconstrained witness shape")
        return TensorType(value.type.get_element_type(), shapes[value])

    block = Block(arg_types=[value_type(value) for value in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        cloned = op.clone(value_mapper=mapping)
        block.add_op(cloned)
        for old, new in zip(op.results, list(cloned.results)):
            mapping[old] = Rewriter.replace_value_with_new_type(new, value_type(old))
    result = mapping[reduction.results[0]]
    block.add_op(ReturnOp(result))
    witness = ModuleOp(
        [FuncOp("forward", FunctionType.from_lists([value.type for value in block.args], [result.type]), Region(block))]
    )
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    printer = Printer(stream=scalar_stream, print_generic_format=True)
    for op in (producer, reduction):
        for scalar_op in op.regions[0].blocks[0].ops:
            printer.print_op(scalar_op)
    dtype = str(producer.results[0].type.get_element_type())
    width = int(dtype[1:])
    output = {"shape": list(shapes[reduction.results[0]]), "dtype": str(reduction.results[0].type.get_element_type())}
    return text, {
        "schema": "actual_source_pointwise_reduction_witness_v1",
        "mechanism": "pointwise_reduction",
        "source_indices": [producer_index, reduction_index],
        "source_shape": list(shape),
        "probe_shape": list(reduced),
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "producer_result_scalar_operation": getattr(
            producer.regions[0].blocks[0].last_op.operands[0].owner, "name", None
        ),
        "reduction_result_scalar_operation": getattr(
            reduction.regions[0].blocks[0].last_op.operands[0].owner, "name", None
        ),
        "inputs": [{"shape": list(shapes[value]), "dtype": str(value.type.get_element_type())} for value in boundary],
        "outputs": [output],
        "output": output,
        "reduction_dimensions": list(dims),
        "source_intermediate_payload_bytes": math.prod(shape) * ((width + 7) // 8),
        "probe_intermediate_payload_bytes": math.prod(reduced) * ((width + 7) // 8),
        "all_source_producer_uses_preserved": True,
        "recomputation_factor": 1,
        "scope": "cloned scalar regions/maps/initializers; lexicographic full input domain at reduced extents",
    }


def extract_dequant_contraction(
    source_text: str, source_indices: Sequence[int], *, max_extent: int = 3
) -> tuple[str, dict[str, Any]]:
    """Clone a sole-use signed per-channel dequantization feeding a named f32 matmul.

    Preserve the original no-recomputation geometry: RHS fusion needs M=1, LHS needs N=1.
    Ranking by actual source intermediate size is only selection, not full-backend attribution.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.analysis.optimization.witnesses.gathers import _props
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if not 1 <= max_extent <= 8:
        raise ValueError("contraction witness extent must lie in [1,8]")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1:
        raise ValueError("contraction witness requires one source function")
    ops = [op for op in functions[0].body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("contraction source indices are outside the source function")
    options = []
    for index in sorted(permitted):
        mm = ops[index]
        if mm.name != "linalg.matmul" or len(mm.operands) != 3 or len(mm.results) != 1:
            continue
        explicit_maps = _props(mm, "indexing_maps")
        if explicit_maps is not None:
            from xdsl.ir.affine import AffineDimExpr

            maps = [attribute.data for attribute in explicit_maps]
            if (
                len(maps) != 3
                or any(amap.num_dims != 3 or amap.num_symbols for amap in maps)
                or any(not all(isinstance(expr, AffineDimExpr) for expr in amap.results) for amap in maps)
                or [[expr.position for expr in amap.results] for amap in maps] != [[0, 2], [2, 1], [0, 1]]
            ):
                continue
        types = [value.type for value in (*mm.operands, *mm.results)]
        if any(
            not isinstance(ty, TensorType) or len(ty.get_shape()) != 2 or str(ty.get_element_type()) != "f32"
            for ty in types
        ):
            continue
        (m, k), (kk, n), output, result = [tuple(ty.get_shape()) for ty in types]
        if k != kk or output != (m, n) or result != output or min(m, k, n) <= 0:
            continue
        for side in (0, 1):
            producer = mm.operands[side].owner
            name = getattr(getattr(producer, "op_name", None), "data", getattr(producer, "name", None))
            if (
                name != "quant_ext.dequantize_per_channel"
                or positions.get(producer) not in permitted
                or len(producer.operands) != 3
                or len(producer.results) != 1
                or len(list(producer.results[0].uses)) != 1
                or (n if side == 0 else m) != 1
            ):
                continue
            axis = getattr(getattr(_props(producer, "axis"), "value", None), "data", None)
            source_type, scale_type, zero_type = [value.type for value in producer.operands]
            shape = tuple(producer.results[0].type.get_shape())
            if (
                axis not in (0, 1)
                or str(producer.results[0].type.get_element_type()) != "f32"
                or str(source_type.get_element_type()) not in {"i8", "i16", "i32"}
                or getattr(_props(producer, "input_dtype"), "data", str(source_type.get_element_type()))
                != str(source_type.get_element_type())
                or str(scale_type.get_element_type()) != "f32"
                or str(zero_type.get_element_type()) != "i32"
                or tuple(source_type.get_shape()) != shape
                or tuple(scale_type.get_shape()) != (shape[axis],)
                or tuple(zero_type.get_shape()) != (shape[axis],)
            ):
                continue
            options.append((math.prod(shape), positions[producer], index, side, axis, (m, k, n)))
    if not options:
        raise ValueError("no exact sole-use dequant-to-contraction source pair with non-amplifying geometry")
    _, producer_index, consumer_index, side, axis, geometry = max(options)
    producer, mm = ops[producer_index], ops[consumer_index]
    m, k, n = [min(extent, max_extent) for extent in geometry]
    operand_shapes = [(m, k), (k, n)]
    dq_shape = operand_shapes[side]
    boundary = [*producer.operands, mm.operands[1 - side], mm.operands[2]]
    if len(set(boundary)) != len(boundary):
        raise ValueError("contraction witness shared boundary aliases require an explicit alias-preserving extractor")
    shapes = [dq_shape, (dq_shape[axis],), (dq_shape[axis],), operand_shapes[1 - side], (m, n)]
    types = [TensorType(value.type.get_element_type(), shape) for value, shape in zip(boundary, shapes)]
    block = Block(arg_types=types)
    mapping = dict(zip(boundary, block.args))
    dq = producer.clone(value_mapper=mapping)
    block.add_op(dq)
    mapping[producer.results[0]] = Rewriter.replace_value_with_new_type(
        dq.results[0], TensorType(producer.results[0].type.get_element_type(), dq_shape)
    )
    contraction = mm.clone(value_mapper=mapping)
    block.add_op(contraction)
    result = Rewriter.replace_value_with_new_type(contraction.results[0], types[-1])
    block.add_op(ReturnOp(result))
    witness = ModuleOp([FuncOp("forward", FunctionType.from_lists(types, [result.type]), Region(block))])
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    output = {"shape": [m, n], "dtype": "f32"}
    return text, {
        "schema": "actual_source_dequant_contraction_witness_v1",
        "mechanism": "dequant_contraction",
        "source_indices": [producer_index, consumer_index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "inputs": [{"shape": list(shape), "dtype": str(ty.get_element_type())} for shape, ty in zip(shapes, types)],
        "output": output,
        "outputs": [output],
        "source_geometry_mkn": list(geometry),
        "probe_geometry_mkn": [m, k, n],
        "producer_operand": side,
        "channel_axis": axis,
        "source_intermediate_payload_bytes": math.prod(producer.results[0].type.get_shape()) * 4,
        "probe_intermediate_payload_bytes": math.prod(dq_shape) * 4,
        "scalar_semantics": "signed convert to f32, f32 subtract zero point, f32 multiply scale; "
        "matmul i,j,k order with distinct f32 multiply then initialized f32 add",
        "indexing_semantics": "actual cloned per-channel axis and named matmul indexing",
        "all_source_producer_uses_preserved": True,
        "recomputation_factor": 1,
        "scope": "exact selected source operations at reduced extents, not full-backend attribution",
    }


def evaluate_pointwise_source(
    source_text: str, inputs: Sequence[Any], *, allow_quantized_epilogue: bool = False
) -> list[Any]:
    """Independent concrete source semantics, rounding after each typed scalar operation.

    Index maps use xDSL's affine evaluator. Numeric values use NumPy scalar containers and
    Python libm, not candidate builders or the saturating command-buffer golden.
    """
    import numpy as np

    from merlin.analysis.optimization.witnesses.gathers import (
        _generic_reduction_info,
        _info,
        _props,
        bf16_bits,
        bf16_values,
    )
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source_text)
    function = next(o for o in module.body.block.ops if o.name == "func.func")
    env = {}
    for argument, values in zip(function.body.block.args, inputs, strict=True):
        dtype = str(argument.type.get_element_type())
        env[argument] = bf16_values(bf16_bits(values)) if dtype == "bf16" else values

    def round_value(value, ty):
        dtype = str(ty)
        if dtype == "f32":
            return float(np.float32(value))
        if dtype == "bf16":
            return float(bf16_values(bf16_bits(np.asarray([value], dtype=np.float32)))[0])
        if dtype == "index":
            return int(value)
        if dtype.startswith("i") and dtype[1:].isdigit():
            width = int(dtype[1:])
            unsigned = int(value) % (1 << width)
            return unsigned - (1 << width) if unsigned >= (1 << (width - 1)) else unsigned
        raise ValueError(f"unsupported source scalar dtype {dtype}")

    def scalar(op, values, indices=()):
        name = op.name
        if name in {"arith.addi", "arith.subi", "arith.muli"}:
            flags = _props(op, "overflowFlags")
            if flags is not None and getattr(flags, "data", None) != frozenset():
                raise ValueError("independent modular arithmetic does not model poison overflow flags")
        if name == "arith.constant":
            value = _props(op, "value").value.data
        elif name in {"arith.addf", "arith.addi"}:
            value = values[0] + values[1]
        elif name in {"arith.subf", "arith.subi"}:
            value = values[0] - values[1]
        elif name in {"arith.mulf", "arith.muli"}:
            value = values[0] * values[1]
        elif name == "arith.negf":
            value = -values[0]
        elif name in {"arith.maxsi", "arith.minsi"}:
            value = (max if name == "arith.maxsi" else min)(values)
        elif name in {"arith.maxui", "arith.minui"}:
            width = int(str(op.operands[0].type)[1:])
            unsigned = [int(item) % (1 << width) for item in values]
            value = (max if name == "arith.maxui" else min)(unsigned)
        elif name == "arith.divf":
            value = values[0] / values[1]
        elif name in {"arith.extsi", "arith.trunci", "arith.sitofp"}:
            value = values[0]
        elif name == "arith.index_cast":
            value = int(values[0])
        elif name == "linalg.index":
            value = indices[_props(op, "dim").value.data]
        elif name == "tensor.extract":
            array, *index = values
            if len(index) != array.ndim or any(i < 0 or i >= n for i, n in zip(index, array.shape)):
                raise ValueError("independent gather index is out of bounds")
            value = array[tuple(index)].item()
        elif name in {"arith.extui", "arith.uitofp"}:
            width = int(str(op.operands[0].type)[1:])
            value = int(values[0]) % (1 << width)
        elif name in {"arith.fptosi", "arith.fptoui"}:
            width = int(str(op.results[0].type)[1:])
            value = math.trunc(values[0])
            lo, hi = (0, (1 << width) - 1) if name.endswith("ui") else (-(1 << (width - 1)), (1 << (width - 1)) - 1)
            if not lo <= value <= hi:
                raise ValueError("source floating cast has undefined out-of-range input")
        elif name in {"math.tanh", "math.erf", "math.sqrt", "math.exp"}:
            value = getattr(math, name.split(".")[1])(values[0])
        elif name == "math.rsqrt":
            value = 1.0 / math.sqrt(values[0])
        elif allow_quantized_epilogue and name == "math.roundeven":
            value = float(np.rint(np.float32(values[0])))
        elif allow_quantized_epilogue and name in {"arith.maximumf", "arith.minimumf"}:
            left, right = values
            maximum = name == "arith.maximumf"
            if math.isnan(left) or math.isnan(right):
                value = float("nan")
            elif left == right == 0:
                signs = (math.copysign(1, left), math.copysign(1, right))
                negative = all(sign < 0 for sign in signs) if maximum else any(sign < 0 for sign in signs)
                value = -0.0 if negative else 0.0
            else:
                value = max(left, right) if maximum else min(left, right)
        else:
            raise ValueError(f"unsupported independent scalar operation {name}")
        return round_value(value, op.results[0].type)

    for op in function.body.block.ops:
        if op.name == "func.return":
            return [env[v] for v in op.operands]
        actual_name = getattr(getattr(op, "op_name", None), "data", op.name)
        if actual_name == "quant_ext.dequantize_per_channel":
            values, scales, zeros = [env[value] for value in op.operands]
            axis = _props(op, "axis").value.data
            if axis not in range(values.ndim) or str(op.results[0].type.get_element_type()) != "f32":
                raise ValueError("unsupported independent per-channel dequantization")
            result = np.empty(values.shape, dtype=np.float32)
            for index in product(*(range(n) for n in values.shape)):
                channel = index[axis]
                result[index] = np.float32(
                    np.float32(np.float32(values[index]) - np.float32(zeros[channel])) * np.float32(scales[channel])
                )
            env[op.results[0]] = result
        elif op.name == "linalg.reduce":
            dims = tuple(_props(op, "dimensions").get_values())
            if len(op.operands) != 2 or len(op.results) != 1:
                raise ValueError("independent reduction supports one input and initialized output")
            values = env[op.operands[0]]
            result = env[op.operands[1]].copy()
            body = op.regions[0].blocks[0]
            for index in product(*(range(extent) for extent in values.shape)):
                out_index = tuple(value for axis, value in enumerate(index) if axis not in dims)
                local = dict(zip(body.args, [values[index].item(), result[out_index].item()], strict=True))
                for operation in body.ops:
                    if operation.name == "linalg.yield":
                        result[out_index] = local[operation.operands[0]]
                        break
                    local[operation.results[0]] = scalar(
                        operation, [local[value] if value in local else env[value] for value in operation.operands]
                    )
            env[op.results[0]] = result
        elif op.name == "linalg.matmul":
            lhs, rhs, initial = [env[value] for value in op.operands]
            types = [str(value.type.get_element_type()) for value in op.operands]
            if all(dtype in {"i8", "i16", "i32", "i64"} for dtype in types):
                from xdsl.ir.affine import AffineDimExpr

                op.verify()
                maps = [item.data for item in op.get_indexing_maps()]
                if (
                    len(maps) != 3
                    or any(amap.num_symbols or amap.num_dims != 3 for amap in maps)
                    or any(not all(isinstance(expr, AffineDimExpr) for expr in amap.results) for amap in maps)
                    or [[expr.position for expr in amap.results] for amap in maps] != [[0, 2], [2, 1], [0, 1]]
                ):
                    raise ValueError("integer matmul reference requires canonical source indexing")
                result = initial.copy()
                body = op.regions[0].block
                for i, j, k in product(range(result.shape[0]), range(result.shape[1]), range(lhs.shape[1])):
                    local = dict(zip(body.args, [lhs[i, k].item(), rhs[k, j].item(), result[i, j].item()], strict=True))
                    for operation in body.ops:
                        if operation.name == "linalg.yield":
                            result[i, j] = local[operation.operands[0]]
                            break
                        local[operation.results[0]] = scalar(operation, [local[value] for value in operation.operands])
                env[op.results[0]] = result
                continue
            if any(dtype != "f32" for dtype in types):
                raise ValueError("independent contraction reference requires supported uniform arithmetic")
            result = initial.copy().astype(np.float32)
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    for k in range(lhs.shape[1]):
                        result[i, j] = np.float32(result[i, j] + np.float32(lhs[i, k] * rhs[k, j]))
            env[op.results[0]] = result
        elif op.name == "arith.constant":
            env[op.results[0]] = scalar(op, [])
        elif op.name in {"tensor.empty", "tensor.splat", "linalg.fill"}:
            ty = op.results[0].type
            shape, dtype = ty.get_shape(), str(ty.get_element_type())
            fill = 0 if op.name == "tensor.empty" else env[op.operands[0]]
            env[op.results[0]] = np.full(
                shape, fill, dtype="float32" if dtype in {"f32", "bf16"} else f"int{dtype[1:]}"
            )
        elif op.name in {"tensor.expand_shape", "tensor.collapse_shape"}:
            env[op.results[0]] = env[op.operands[0]].reshape(op.results[0].type.get_shape()).copy()
        elif op.name == "linalg.transpose":
            env[op.results[0]] = env[op.operands[0]].transpose(tuple(_props(op, "permutation").get_values())).copy()
        elif op.name == "tensor.insert_slice":
            if len(op.operands) != 2:
                raise ValueError("independent insert needs static same-rank offsets/sizes/strides")
            values, destination = [env[v] for v in op.operands]
            offsets, sizes, strides = [
                tuple(_props(op, key).get_values()) for key in ("static_offsets", "static_sizes", "static_strides")
            ]
            if (
                len(offsets) != destination.ndim
                or sizes != values.shape
                or len(strides) != destination.ndim
                or len(sizes) != destination.ndim
                or any(
                    off < 0 or stride <= 0 or off + (size - 1) * stride >= extent
                    for off, size, stride, extent in zip(offsets, sizes, strides, destination.shape)
                )
            ):
                raise ValueError("independent insert is not a bounded same-rank slice")
            result = destination.copy()
            result[
                tuple(slice(off, off + size * stride, stride) for off, size, stride in zip(offsets, sizes, strides))
            ] = values
            env[op.results[0]] = result
        elif op.name == "tensor.concat":
            arrays = [env[value] for value in op.operands]
            result_shape = tuple(op.results[0].type.get_shape())
            axis = _props(op, "dim").value.data
            if (
                not arrays
                or len(result_shape) == 0
                or not 0 <= axis < len(result_shape)
                or any(array.ndim != len(result_shape) for array in arrays)
                or any(
                    array.shape[dimension] != result_shape[dimension]
                    for array in arrays
                    for dimension in range(len(result_shape))
                    if dimension != axis
                )
                or sum(array.shape[axis] for array in arrays) != result_shape[axis]
                or any(array.dtype != arrays[0].dtype for array in arrays)
            ):
                raise ValueError("independent concat requires exact static segment geometry")
            env[op.results[0]] = np.concatenate(arrays, axis=axis)
        elif op.name == "linalg.generic":
            info = _info(op, allow_gather=True, allow_quantized_epilogue=allow_quantized_epilogue)
            reduction_info = _generic_reduction_info(op) if info is None else None
            if info is None and reduction_info is None:
                raise ValueError("reference encountered unsupported generic source")
            if reduction_info is not None:
                maps, n_in = reduction_info["maps"], reduction_info["n_in"]
                bounds = reduction_info["iteration_shape"]
                out = env[op.operands[-1]].copy()
            else:
                shape, maps, n_in = info
                ty = op.results[0].type.get_element_type()
                out = np.zeros(shape, dtype="float32" if str(ty) in {"f32", "bf16"} else f"int{str(ty)[1:]}")
                perm = [e.position for e in maps[-1].results]
                bounds = [shape[perm.index(d)] for d in range(len(shape))]
            body = op.regions[0].blocks[0]
            for index in product(*(range(n) for n in bounds)):
                args = [env[v][tuple(amap.eval(index, ()))].item() for v, amap in zip(op.operands[:n_in], maps[:n_in])]
                output_index = tuple(maps[-1].eval(index, ()))
                output_value = out[output_index].item() if reduction_info is not None else 0
                local = dict(zip(body.args, [*args, output_value], strict=True))
                for operation in body.ops:
                    if operation.name == "linalg.yield":
                        out[output_index] = local[operation.operands[0]]
                        break
                    local[operation.results[0]] = scalar(
                        operation, [local[v] if v in local else env[v] for v in operation.operands], index
                    )
            env[op.results[0]] = out
        else:
            raise ValueError(f"unsupported independent source operation {op.name}")
    raise ValueError("source witness has no return")
