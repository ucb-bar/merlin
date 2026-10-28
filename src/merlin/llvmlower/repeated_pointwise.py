"""Prepare exact pure pointwise values before an explicit repeated broadcast.

Tensor views carry a checked row-major coordinate relation. A scalar body is
cloned unchanged onto the smaller domain only when every input has that same
relation, omits the repeated dimension, or is an explicitly bound immutable
constant with byte-identical repeated entries. No arithmetic is replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp, YieldOp
from xdsl.ir import Block, BlockArgument, Region
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter

from .quant_round_check import pure_scalar_body

PASS_NAME = "prepare-repeated-pointwise"


def selected():
    """The same explicit selection controls preparation and recipe membership."""
    from .optional_passes import switched

    return switched(PASS_NAME, os.environ.get("MERLIN_PREPARE_REPEATED_POINTWISE", "0") == "1")


def _strict(operation):
    return any(
        "strictfp" in name or "strictfp" in str(value)
        for name, value in (*operation.attributes.items(), *operation.properties.items())
    )


@dataclass(frozen=True)
class _Repeat:
    base: object
    factor: int
    chunk: int


def _shape(value):
    typ = value.type
    if not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr):
        return None
    shape = tuple(typ.get_shape())
    return shape if all(d > 0 for d in shape) else None


def _axis(shape, repeat):
    return next(
        (i for i, d in enumerate(shape) if d % repeat.factor == 0 and math.prod(shape[i + 1 :]) == repeat.chunk), None
    )


def _reassociation(rank):
    return builtin.ArrayAttr([builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(rank)])])


def _reshape(value, shape, before):
    if math.prod(value.type.get_shape()) != math.prod(shape):
        raise ValueError("repeated tensor view changed its complete element count")
    if tuple(value.type.get_shape()) == tuple(shape):
        return value
    flat_type = builtin.TensorType(value.type.element_type, [math.prod(shape)])
    if len(value.type.get_shape()) != 1:
        flat = tensor.CollapseShapeOp(
            operands=[value],
            result_types=[flat_type],
            properties={"reassociation": _reassociation(len(value.type.get_shape()))},
        )
        before.parent.insert_op_before(flat, before)
        value = flat.result
    if len(shape) == 1:
        return value
    expanded = tensor.ExpandShapeOp(
        value, [], _reassociation(len(shape)), shape, builtin.TensorType(value.type.element_type, shape)
    )
    before.parent.insert_op_before(expanded, before)
    return expanded.result


def _broadcast(base, shape, axis, factor, before):
    small = list(shape)
    small[axis] //= factor
    expanded = [*small[: axis + 1], 1, *small[axis + 1 :]]
    full = [*small[: axis + 1], factor, *small[axis + 1 :]]
    source = _reshape(base, expanded, before)
    empty = tensor.EmptyOp([], builtin.TensorType(base.type.element_type, full))
    before.parent.insert_op_before(empty, before)
    dims = len(full)
    coordinates = [AffineDimExpr(i) if i != axis + 1 else AffineConstantExpr(0) for i in range(dims)]
    block = Block(arg_types=[base.type.element_type] * 2)
    block.add_op(YieldOp(block.args[0]))
    broadcast = GenericOp(
        [source],
        [empty.tensor],
        Region(block),
        [
            builtin.AffineMapAttr(AffineMap(dims, 0, tuple(coordinates))),
            builtin.AffineMapAttr(AffineMap.identity(dims)),
        ],
        [IteratorTypeAttr(IteratorType.PARALLEL)] * dims,
        [empty.tensor.type],
    )
    before.parent.insert_op_before(broadcast, before)
    return _reshape(broadcast.results[0], shape, before)


def _generic(operation):
    if (
        not isinstance(operation, GenericOp)
        or len(operation.results) != 1
        or len(operation.outputs) != 1
        or not pure_scalar_body(operation.body.block)
        or operation.library_call is not None
        or _strict(operation)
    ):
        return None
    shape = _shape(operation.results[0])
    if shape is None or not shape:
        return None
    maps = tuple(attr.data for attr in operation.indexing_maps)
    rank = len(shape)
    if (
        maps[-1] != AffineMap.identity(rank)
        or len(operation.iterator_types) != rank
        or any(str(it) != "#linalg.iterator_type<parallel>" for it in operation.iterator_types)
        or any(operation.body.block.args[-1].uses)
    ):
        return None
    for value, affine in zip(operation.operands, maps, strict=True):
        source = _shape(value)
        if source is None or affine.num_dims != rank or affine.num_symbols or len(affine.results) != len(source):
            return None
        seen = set()
        for d, extent in zip(affine.results, source, strict=True):
            if isinstance(d, AffineDimExpr):
                if d.position in seen or extent != shape[d.position]:
                    return None
                seen.add(d.position)
            elif not isinstance(d, AffineConstantExpr) or d.value != 0 or extent != 1:
                return None
    return shape, maps


def _constant(value, axis, factor, loader, before, budget, report):
    import numpy as np

    raw = None
    if isinstance(value.owner, arith.ConstantOp):
        attr = value.owner.value
        if isinstance(attr, builtin.DenseIntOrFPElementsAttr):
            raw = attr.data.data
    elif isinstance(value, BlockArgument) and loader is not None:
        raw = loader(value)
    if raw is None or len(raw) > budget:
        return None
    shape = _shape(value)
    element = value.type.element_type
    if (
        shape is None
        or not isinstance(element, builtin.Float32Type)
        or shape[axis] % factor
        or len(raw) != math.prod(shape) * 4
    ):
        return None
    # Compare complete words, including signed zero and NaN payloads. Never use
    # approximate floating equality or substitute a reciprocal for division.
    words = np.frombuffer(raw, dtype=np.dtype("V4")).reshape(shape)
    reduced = np.take(words, range(0, shape[axis], factor), axis=axis)
    if np.repeat(reduced, factor, axis=axis).tobytes() != raw:
        report["refused_differing_constants"] += 1
        return None
    result_shape = list(shape)
    result_shape[axis] //= factor
    typ = builtin.TensorType(element, result_shape)
    data = reduced.tobytes()
    constant = arith.ConstantOp(builtin.DenseIntOrFPElementsAttr(typ, builtin.BytesAttr(data)))
    before.parent.insert_op_before(constant, before)
    report["constant_bytes"] += len(data)
    report["constant_bindings"].append(
        {
            "source_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "reduced_bytes_sha256": hashlib.sha256(data).hexdigest(),
            "original_shape": list(shape),
            "reduced_shape": result_shape,
        }
    )
    return constant.result


def hoist_repeated_pointwise(module, *, immutable_argument_loader=None, max_constant_bytes=4096):
    """Return checked typed IR and a structural work report; default ABI unchanged.

    The optional loader is an explicit caller declaration that the selected
    public argument has exactly these immutable bytes for this compiled image.
    Missing bindings never authorize specializing an ordinary runtime argument.
    """
    if type(max_constant_bytes) is not int or max_constant_bytes < 0:
        raise ValueError("repeated pointwise preparation needs a nonnegative constant budget")
    current = module.clone()
    current.verify()
    repeats = {}
    report = {
        "schema": "merlin.repeated_pointwise_preparation.v1",
        "rewrites": 0,
        "constant_bytes": 0,
        "constant_bindings": [],
        "refused_differing_constants": 0,
        "computed_side_producers": 0,
        "cloned_side_scalar_evaluations": {},
        "removed_scalar_evaluations": {},
        "scope": "source-exact structural work; numerical/machine/timing qualification independent",
    }
    computed = {}

    def reduced_input(value, axis, factor, before, depth=0):
        relation = repeats.get(value)
        if relation is not None and relation.factor == factor and _axis(_shape(value), relation) == axis:
            shape = list(_shape(value))
            shape[axis] //= factor
            return _reshape(relation.base, shape, before)
        key = (value, axis, factor)
        if key in computed:
            return computed[key]
        constant = _constant(
            value,
            axis,
            factor,
            immutable_argument_loader,
            before,
            max_constant_bytes - report["constant_bytes"],
            report,
        )
        if constant is not None:
            return constant
        # Pull the demanded coordinate relation through a bounded pure DAG.
        # Inputs omitting this coordinate remain runtime values; a varying
        # input must independently prove its complete repeated coordinates.
        owner = value.owner
        specification = _generic(owner)
        if specification is None or owner.parent is not before.parent or depth >= 8 or len(computed) >= 128:
            return None
        shape, maps = specification
        if shape[axis] % factor:
            return None
        planned = []
        for operand, affine in zip(owner.inputs, maps, strict=False):
            positions = [
                i for i, expr in enumerate(affine.results) if isinstance(expr, AffineDimExpr) and expr.position == axis
            ]
            if not positions:
                planned.append(operand)
            else:
                reduced = reduced_input(operand, positions[0], factor, before, depth + 1)
                if reduced is None:
                    return None
                planned.append(reduced)
        shape = list(shape)
        shape[axis] //= factor
        empty = tensor.EmptyOp([], builtin.TensorType(value.type.element_type, shape))
        before.parent.insert_op_before(empty, before)
        reduced = GenericOp(
            planned,
            [empty.tensor],
            owner.body.clone(),
            owner.indexing_maps,
            owner.iterator_types,
            [empty.tensor.type],
        )
        reduced.attributes = dict(owner.attributes)
        before.parent.insert_op_before(reduced, before)
        computed[key] = reduced.results[0]
        report["computed_side_producers"] += 1
        for scalar in owner.body.block.ops:
            if scalar.name != "linalg.yield":
                costs = report["cloned_side_scalar_evaluations"]
                costs[scalar.name] = costs.get(scalar.name, 0) + math.prod(shape)
        return reduced.results[0]

    for function in tuple(current.body.block.ops):
        if (
            not isinstance(function, func.FuncOp)
            or not function.body.blocks
            or _strict(function)
            or len(function.body.blocks) != 1
        ):
            continue
        block = function.body.block
        for operation in tuple(block.ops):
            # Only the exact static row-major reshape operations change the view.
            if isinstance(operation, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
                source = operation.operands[0]
                relation = repeats.get(source)
                shape = _shape(operation.results[0])
                if relation is not None and shape is not None and math.prod(shape) == math.prod(_shape(source)):
                    axis = _axis(shape, relation)
                    reduced_shape = [math.prod(shape) // relation.factor] if axis is None else list(shape)
                    if axis is not None:
                        reduced_shape[axis] //= relation.factor
                    repeats[operation.results[0]] = _Repeat(
                        _reshape(relation.base, reduced_shape, operation), relation.factor, relation.chunk
                    )
                continue
            if isinstance(operation, TransposeOp):
                relation = repeats.get(operation.inputs[0])
                transpose_body = operation.body.block
                if (
                    relation is None
                    or len(transpose_body.ops) != 1
                    or transpose_body.last_op.name != "linalg.yield"
                    or tuple(transpose_body.last_op.operands) != (transpose_body.args[0],)
                ):
                    continue
                shape = _shape(operation.inputs[0])
                axis = _axis(shape, relation)
                if axis is None:
                    continue
                small = list(shape)
                small[axis] //= relation.factor
                source = _reshape(relation.base, small, operation)
                permutation = tuple(operation.permutation.get_values())
                output = [small[i] for i in permutation]
                empty = tensor.EmptyOp([], builtin.TensorType(source.type.element_type, output))
                block.insert_op_before(empty, operation)
                reduced = TransposeOp(source, empty.tensor, operation.permutation)
                block.insert_op_before(reduced, operation)
                out_axis = permutation.index(axis)
                repeats[operation.results[0]] = _Repeat(
                    reduced.results[0], relation.factor, math.prod(output[out_axis + 1 :])
                )
                continue
            specification = _generic(operation)
            if specification is None:
                continue
            shape, maps = specification
            body = operation.body.block
            if len(operation.inputs) == 1 and len(body.ops) == 1 and body.last_op.operands[0] is body.args[0]:
                for axis, expression in enumerate(maps[0].results):
                    if isinstance(expression, AffineConstantExpr) and shape[axis] > 1:
                        # The initial repeat has matching ranks and only one
                        # expanded unit axis; permutations are handled separately.
                        if (
                            len(maps[0].results) == len(shape)
                            and sum(isinstance(x, AffineConstantExpr) for x in maps[0].results) == 1
                            and all(
                                isinstance(x, AffineConstantExpr) or x == AffineDimExpr(i)
                                for i, x in enumerate(maps[0].results)
                            )
                        ):
                            repeats[operation.results[0]] = _Repeat(
                                operation.inputs[0], shape[axis], math.prod(shape[axis + 1 :])
                            )
                continue
            candidates = []
            for value, affine in zip(operation.inputs, maps, strict=False):
                relation = repeats.get(value)
                if relation is None:
                    continue
                axis = _axis(_shape(value), relation)
                if axis is not None and isinstance(affine.results[axis], AffineDimExpr):
                    candidates.append((affine.results[axis].position, relation.factor))
            for axis, factor in candidates:
                planned = []
                for value, affine in zip(operation.inputs, maps, strict=False):
                    positions = [
                        i
                        for i, expr in enumerate(affine.results)
                        if isinstance(expr, AffineDimExpr) and expr.position == axis
                    ]
                    if not positions:
                        planned.append(value)
                        continue
                    input_axis = positions[0]
                    reduced = reduced_input(value, input_axis, factor, operation)
                    if reduced is None:
                        break
                    planned.append(reduced)
                else:
                    reduced_shape = list(shape)
                    reduced_shape[axis] //= factor
                    empty = tensor.EmptyOp(
                        [], builtin.TensorType(operation.results[0].type.element_type, reduced_shape)
                    )
                    block.insert_op_before(empty, operation)
                    reduced = GenericOp(
                        planned,
                        [empty.tensor],
                        operation.body.clone(),
                        operation.indexing_maps,
                        operation.iterator_types,
                        [empty.tensor.type],
                    )
                    reduced.attributes = dict(operation.attributes)
                    block.insert_op_before(reduced, operation)
                    expanded = _broadcast(reduced.results[0], shape, axis, factor, operation)
                    repeats[expanded] = _Repeat(reduced.results[0], factor, math.prod(shape[axis + 1 :]))
                    saved = math.prod(shape) - math.prod(reduced_shape)
                    for scalar in body.ops:
                        if scalar.name != "linalg.yield":
                            costs = report["removed_scalar_evaluations"]
                            costs[scalar.name] = costs.get(scalar.name, 0) + saved
                    Rewriter.replace_op(operation, [], [expanded])
                    report["rewrites"] += 1
                    break
    current.verify()
    return (current if report["rewrites"] else module.clone()), report


def _hoist_stored_repeated_pointwise(module, *, bundle_dir):
    """Internal source preparation with a fresh generated-caller/span binding."""
    from .weight_prepack_runtime import fixed_arguments
    from .weights_pack import load_safetensors_header

    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "weights.safetensors.manifest.json"
    weights = bundle_dir / "weights.safetensors"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    header, offset = load_safetensors_header(weights)
    caller_bytes = (bundle_dir / "model.mlir").read_bytes()
    entries = [
        op
        for op in module.body.block.ops
        if isinstance(op, func.FuncOp)
        and op.body.blocks
        and (op.sym_visibility is None or op.sym_visibility.data != "private")
    ]
    if len(entries) != 1:
        raise ValueError("stored repeated-pointwise binding requires one original public entry")
    entry = entries[0]
    # Manifest storage roles alone do not establish immutability. The existing
    # generated-caller proof excludes state/stream slots and opaque effects.
    fixed = fixed_arguments(bundle_dir, manifest, entry=entry.sym_name.data)
    loaded = {}

    def load(argument):
        owner = argument.owner.parent_op()
        if (
            not isinstance(owner, func.FuncOp)
            or argument.owner is not owner.body.block
            or owner.sym_name != entry.sym_name
            or owner.function_type != entry.function_type
            or owner.sym_visibility is not None
            and owner.sym_visibility.data == "private"
            or argument.index not in fixed
        ):
            return None
        row = manifest.get(str(argument.index))
        if not row or row.get("kind") not in {"param", "buffer"}:
            return None
        metadata = header.get(row.get("weight"))
        if (
            not metadata
            or metadata.get("dtype") != "F32"
            or row.get("dtype") != "float32"
            or list(argument.type.get_shape()) != row.get("shape")
            or row.get("shape") != metadata.get("shape")
        ):
            return None
        begin, end = metadata["data_offsets"]
        if end - begin > 4096:
            return None
        with weights.open("rb") as handle:
            handle.seek(offset + begin)
            data = handle.read(end - begin)
        if len(data) != end - begin:
            return None
        loaded[argument.index] = {
            "argument_index": argument.index,
            "parameter": row["weight"],
            "storage_kind": row["kind"],
            "dtype": row["dtype"],
            "shape": row["shape"],
            "storage_path": str(weights.resolve()),
            "payload_offset": offset + begin,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        return data

    rewritten, report = hoist_repeated_pointwise(module, immutable_argument_loader=load)
    def require_binding():
        if (manifest_path.read_bytes() != manifest_bytes
                or (bundle_dir / "model.mlir").read_bytes() != caller_bytes
                or load_safetensors_header(weights) != (header, offset)):
            raise ValueError("repeated pointwise preparation changed its selected source or argument binding")
        if fixed_arguments(bundle_dir, manifest, entry=entry.sym_name.data) != fixed:
            raise ValueError("repeated pointwise preparation lost its immutable generated-caller binding")
        # Reopen each exact stored span before publishing/reusing specialization.
        with weights.open("rb") as handle:
            for item in loaded.values():
                handle.seek(item["payload_offset"])
                if hashlib.sha256(handle.read(item["bytes"])).hexdigest() != item["sha256"]:
                    raise ValueError("repeated pointwise preparation lost its immutable stored argument")

    require_binding()
    report.update(
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        immutable_argument_sources=list(loaded.values()),
        immutable_generated_argument_indices=list(fixed),
    )
    return rewritten, report, require_binding


def prepare_repeated_pointwise(prepared: Path, directory: Path, *, bundle_dir: Path):
    """Explicit ordinary stored-parameter selection, preserving public interfaces."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    if not selected():
        return prepared
    source_bytes = prepared.read_bytes()
    rewritten, report, require_binding = _hoist_stored_repeated_pointwise(
        parse_mlir_text(source_bytes.decode()), bundle_dir=bundle_dir)
    if prepared.read_bytes() != source_bytes:
        raise ValueError("repeated pointwise preparation changed its selected source or argument binding")
    require_binding()
    directory.mkdir(parents=True, exist_ok=False)
    output = prepared
    if report["rewrites"]:
        output = directory / "model.mlir"
        output.write_text(text(rewritten, generic=True))
    report.update(source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  output_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output
