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


def extract_pointwise_view_chain(
    source_text: str, source_indices: Sequence[int], *, max_extent: int = 3, max_elements: int = 4096
) -> tuple[str, dict[str, Any]]:
    """Clone pointwise -> unique static collapse/expand chain at a task boundary.

    Only the indexed top-level slice is parsed, avoiding a full multi-megabyte source parse in
    the 60-second qualification action.  Text scanning establishes source positions and global
    SSA use counts; xdsl parsing and verification establish the operation/type/map semantics.
    Every intermediate use is retained, and every view must be a static dtype-preserving
    element-count bijection.  Ambiguity, fanout, dynamic dimensions, unsupported views, or an
    inconsistent reduced factorization abstains.
    """
    from xdsl.dialects.builtin import DenseArrayBase, FunctionType, ModuleOp, TensorType, i64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.analysis.optimization.witnesses.gathers import (
        _SSA_NAME,
        _TENSOR_TYPE,
        _chunk_result_name,
        _fast_top_level_source_operations,
        _info,
        _props,
        _typed_operands,
    )
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if (
        type(max_extent) is not int
        or not 1 <= max_extent <= 8
        or type(max_elements) is not int
        or not 1 <= max_elements <= 16_384
    ):
        raise ValueError("view-chain witness requires bounded extents and tensor elements")
    chunks, return_chunk = _fast_top_level_source_operations(source_text)
    permitted = set(source_indices)
    if not permitted or any(type(index) is not int or not 0 <= index < len(chunks) for index in permitted):
        raise ValueError("view-chain source indices are outside the source function")
    indexed = {index: chunks[index] for index in sorted(permitted)}
    result_to_index = {
        name: index for index, chunk in indexed.items() if (name := _chunk_result_name(chunk)) is not None
    }
    views = {"tensor.collapse_shape", "tensor.expand_shape"}
    view_by_input: dict[str, list[tuple[int, str]]] = {}
    for index, chunk in indexed.items():
        name = _chunk_result_name(chunk)
        operation = re.match(  # regex-ok: classify a canonical top-level MLIR record
            r"^    %[-A-Za-z0-9_.$]+(?::\d+)?\s*=\s*([A-Za-z0-9_.]+)", chunk
        )
        if name is None or operation is None or operation.group(1) not in views:
            continue
        operands = _SSA_NAME.findall(chunk.split(" : tensor<", 1)[0])
        if len(operands) != 2 or operands[0] != name:
            continue
        view_by_input.setdefault(operands[1], []).append((index, name))

    options = []
    for producer_index, producer_chunk in indexed.items():
        result = _chunk_result_name(producer_chunk)
        if result is None or "= linalg.generic " not in producer_chunk or result not in view_by_input:
            continue
        chain, value, unique = [], result, True
        while value in view_by_input:
            following_views = view_by_input[value]
            if len(following_views) != 1:
                unique = False
                break
            view_index, following = following_views[0]
            if chain and view_index <= chain[-1]:
                raise ValueError("view-chain source order is cyclic or ambiguous")
            # The producer and every intermediate view must have exactly one source use.
            token = re.compile(  # regex-ok: count exact SSA-token uses, including task-external uses
                rf"(?<![-A-Za-z0-9_.$]){re.escape(value)}(?![-A-Za-z0-9_.$])"  # regex-ok: quote an SSA name
            )  # regex-ok: quote an SSA name
            if len(token.findall(source_text)) != 2:
                unique = False
                break
            chain.append(view_index)
            value = following
        if chain and unique:
            options.append((len(chain), producer_index, chain, value))
    if not options:
        raise ValueError("changed host task has no unique pointwise-to-static-view boundary chain")
    _, producer_index, view_indices, final_name = max(options)
    producer_chunk = indexed[producer_index]
    ins = _typed_operands(producer_chunk, "ins")
    outs = _typed_operands(producer_chunk, "outs")
    if not ins or len(outs) != 1:
        raise ValueError("view-chain pointwise producer has unsupported tensor operands")
    initializer_name, output_type_text = outs[0]
    initializer_index = result_to_index.get(initializer_name)
    if (
        initializer_index is None
        or "= tensor.empty()" not in indexed[initializer_index]
        or _TENSOR_TYPE.findall(indexed[initializer_index]) != [output_type_text]
    ):
        raise ValueError("view-chain pointwise output is not a selected exact tensor.empty")
    final_types = _TENSOR_TYPE.findall(indexed[view_indices[-1]])
    if len(final_types) != 2:
        raise ValueError("view-chain final tensor type is ambiguous")
    source_final_type = final_types[-1]

    # Parse only the exact source suffix.  Boundary tensors are explicit arguments; no producer
    # semantics are synthesized and no unrelated full-model operation is executed.
    arguments = ", ".join(f"{name}: {typ}" for name, typ in ins)
    selected_order = [initializer_index, producer_index, *view_indices]
    tiny_source = (
        "builtin.module {\n"
        f"  func.func @forward({arguments}) -> {source_final_type} {{\n"
        + "".join(indexed[index] for index in selected_order)
        + f"    func.return {final_name} : {source_final_type}\n"
        "  }\n}\n"
    )
    try:
        module = parse_mlir_text(tiny_source)
        module.verify()
    except Exception as exc:
        raise ValueError("view-chain selected source slice is not valid typed IR") from exc
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("view-chain selected source slice did not parse as one function")
    function = functions[0]
    source_ops = [op for op in function.body.block.ops if op.name != "func.return"]
    if [op.name for op in source_ops] != [
        "tensor.empty",
        "linalg.generic",
        *[
            re.match(  # regex-ok: classify already-isolated canonical MLIR records
                r"^    %[-A-Za-z0-9_.$]+(?::\d+)?\s*=\s*([A-Za-z0-9_.]+)", indexed[index]
            ).group(1)
            for index in view_indices
        ],
    ]:
        raise ValueError("view-chain selected source slice contains an unexpected operation")
    initializer, producer, *view_ops = source_ops
    info = _info(producer, allow_quantized_epilogue=True)
    if info is None:
        raise ValueError("view-chain producer is not an exact supported all-parallel pointwise map")
    source_shape, maps, n_inputs = info
    if n_inputs != len(ins) or len(producer.operands) != n_inputs + 1:
        raise ValueError("view-chain producer tensor arity changed in the selected slice")
    dtype = str(producer.results[0].type.get_element_type())
    if dtype not in {"f32", "i8", "i16", "i32", "i64"}:
        raise ValueError("view-chain producer dtype is unsupported")
    reduced_shape = tuple(min(extent, max_extent) for extent in source_shape)
    output_map = maps[-1]
    permutation = [expr.position for expr in output_map.results]
    loop_bounds = [reduced_shape[permutation.index(dimension)] for dimension in range(len(source_shape))]
    shapes = {initializer.results[0]: reduced_shape, producer.results[0]: reduced_shape}
    for value, amap in zip(producer.operands[:n_inputs], maps[:n_inputs]):
        needed = tuple(
            loop_bounds[expr.position]
            if isinstance(expr, AffineDimExpr)
            else expr.value + 1
            if isinstance(expr, AffineConstantExpr)
            else -1
            for expr in amap.results
        )
        original = tuple(value.type.get_shape())
        if len(needed) != len(original) or any(
            extent <= 0 or extent > source_extent for extent, source_extent in zip(needed, original)
        ):
            raise ValueError("view-chain pointwise indexing does not admit bounded extents")
        shapes[value] = needed

    current = producer.results[0]
    source_view_shapes, probe_view_shapes = [], []
    for view in view_ops:
        if (
            len(view.operands) != 1
            or len(view.results) != 1
            or view.operands[0] is not current
            or not isinstance(view.results[0].type, TensorType)
            or str(view.results[0].type.get_element_type()) != dtype
        ):
            raise ValueError("view-chain is not a dtype-preserving single-result path")
        source_input = tuple(current.type.get_shape())
        source_output = tuple(view.results[0].type.get_shape())
        probe_input = shapes[current]
        if any(extent <= 0 for extent in (*source_input, *source_output)) or math.prod(source_input) != math.prod(
            source_output
        ):
            raise ValueError("view-chain contains a dynamic or non-bijective shape change")
        reassociation = [[int(item.value.data) for item in group] for group in _props(view, "reassociation")]
        if view.name == "tensor.collapse_shape":
            if [dimension for group in reassociation for dimension in group] != list(range(len(source_input))):
                raise ValueError("view-chain collapse reassociation is not complete and ordered")
            probe_output = tuple(math.prod(probe_input[dimension] for dimension in group) for group in reassociation)
        elif view.name == "tensor.expand_shape":
            if [dimension for group in reassociation for dimension in group] != list(range(len(source_output))):
                raise ValueError("view-chain expand reassociation is not complete and ordered")
            candidate = tuple(min(extent, max_extent) for extent in source_output)
            if any(
                math.prod(candidate[dimension] for dimension in group) != probe_input[index]
                for index, group in enumerate(reassociation)
            ):
                raise ValueError("view-chain reduced expand factorization is not source-derived")
            probe_output = candidate
        else:
            raise ValueError("view-chain contains an unsupported operation")
        if math.prod(probe_input) != math.prod(probe_output):
            raise ValueError("view-chain reduced shape is not bijective")
        shapes[view.results[0]] = probe_output
        source_view_shapes.append([list(source_input), list(source_output)])
        probe_view_shapes.append([list(probe_input), list(probe_output)])
        current = view.results[0]
    if sum(math.prod(shape) for shape in shapes.values()) > max_elements:
        raise ValueError("source-derived pointwise view chain exceeds the tensor-element budget")

    def value_type(value):
        if not isinstance(value.type, TensorType) or value not in shapes:
            raise ValueError("view-chain tensor has no exact reduced extent")
        return TensorType(value.type.get_element_type(), shapes[value])

    block = Block(arg_types=[value_type(value) for value in function.body.block.args])
    mapping = dict(zip(function.body.block.args, block.args))
    for op in source_ops:
        clone = op.clone(value_mapper=mapping)
        if clone.name == "tensor.expand_shape":
            clone.properties["static_output_shape"] = DenseArrayBase.from_list(i64, shapes[op.results[0]])
        block.add_op(clone)
        for old, new in zip(op.results, list(clone.results)):
            mapping[old] = Rewriter.replace_value_with_new_type(new, value_type(old))
    result = mapping[current]
    block.add_op(ReturnOp(result))
    witness = ModuleOp(
        [
            FuncOp(
                "forward", FunctionType.from_lists([value.type for value in block.args], [result.type]), Region([block])
            )
        ]
    )
    try:
        witness.verify()
    except Exception as exc:
        raise ValueError("view-chain reduced source is not valid typed IR") from exc
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    scalar_printer = Printer(stream=scalar_stream, print_generic_format=True)
    scalar_ops = list(producer.regions[0].blocks[0].ops)
    for scalar in scalar_ops:
        scalar_printer.print_op(scalar)
    yielded = scalar_ops[-1].operands[0] if scalar_ops and scalar_ops[-1].name == "linalg.yield" else None
    scalar_result_name = getattr(getattr(yielded, "owner", None), "name", None)
    width = int(dtype[1:])
    spec = lambda value: {"shape": list(shapes[value]), "dtype": str(value.type.get_element_type())}
    final_token = re.compile(  # regex-ok: count exact final SSA-token uses outside the selected slice
        rf"(?<![-A-Za-z0-9_.$]){re.escape(final_name)}(?![-A-Za-z0-9_.$])"  # regex-ok: quote an SSA name
    )  # regex-ok: quote an SSA name
    final_uses = len(final_token.findall(source_text)) - 1
    output = spec(current)
    return text, {
        "schema": "actual_source_pointwise_view_chain_witness_v1",
        "mechanism": "pointwise_view_chain",
        "source_indices": [producer_index, *view_indices],
        "auxiliary_source_indices": [initializer_index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "producer_source_index": producer_index,
        "view_source_indices": view_indices,
        "view_operations": [view.name for view in view_ops],
        "producer_result_scalar_operation": scalar_result_name,
        "source_producer_shape": list(source_shape),
        "probe_producer_shape": list(reduced_shape),
        "source_view_shapes": source_view_shapes,
        "probe_view_shapes": probe_view_shapes,
        "inputs": [spec(value) for value in function.body.block.args],
        "outputs": [output],
        "output": output,
        "probe_intermediate_payload_bytes": math.prod(reduced_shape) * ((width + 7) // 8),
        "probe_output_payload_bytes": math.prod(shapes[current]) * ((width + 7) // 8),
        "all_source_producer_uses_preserved": True,
        "all_source_view_uses_preserved": True,
        "source_final_external_use_count": final_uses,
        "geometry": "source reassociation maps with independently bounded producer dimensions",
        "scope": "one exact source-cloned pointwise producer through its unique static dtype-preserving bijective collapse/expand boundary chain",
    }


def extract_pointwise_concat(
    source_text: str, source_indices: Sequence[int], *, max_extent: int = 3, max_elements: int = 4096
) -> tuple[str, dict[str, Any]]:
    """Clone one sole-use pointwise producer and its static concat consumer.

    The concat axis and segment order come from the actual source.  Every non-concatenated
    dimension is reduced uniformly, while each concatenated segment is reduced independently.
    The one-operand identity form is retained as an actual source operation: it is precisely the
    witness needed when codegen proves that the concat itself is only a type-preserving view.
    A producer with any use other than the selected concat is refused: recomputing it in one
    concat copy loop would otherwise silently drop or duplicate a source-visible value.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.analysis.optimization.witnesses.gathers import _info, _props
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if (
        type(max_extent) is not int
        or not 1 <= max_extent <= 8
        or type(max_elements) is not int
        or not 1 <= max_elements <= 16_384
    ):
        raise ValueError("pointwise concat witness requires bounded extents and tensor elements")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func" and len(op.body.blocks)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("pointwise concat witness requires one unoutlined source function")
    function = functions[0]
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("pointwise concat source indices are outside the source function")

    options = []
    supported_dtypes = {"f32", "i8", "i16", "i32", "i64"}
    for producer_index in sorted(permitted):
        producer = ops[producer_index]
        producer_info = _info(producer)
        uses = list(producer.results[0].uses) if producer_info is not None else []
        if len(uses) != 1:
            continue
        use = uses[0]
        concat = use.operation
        concat_index = positions.get(concat)
        if (
            concat.name != "tensor.concat"
            or concat_index not in permitted
            or len(concat.results) != 1
            or not concat.operands
            or not 0 <= use.index < len(concat.operands)
            or concat.operands[use.index] is not producer.results[0]
        ):
            continue
        values = [*concat.operands, concat.results[0]]
        if any(not isinstance(value.type, TensorType) for value in values):
            continue
        operand_shapes = [tuple(value.type.get_shape()) for value in concat.operands]
        result_shape = tuple(concat.results[0].type.get_shape())
        rank = len(result_shape)
        try:
            axis = int(_props(concat, "dim").value.data)
        except (AttributeError, TypeError, ValueError):
            continue
        dtype = str(concat.results[0].type.get_element_type())
        if (
            rank == 0
            or not 0 <= axis < rank
            or dtype not in supported_dtypes
            or any(
                len(shape) != rank or any(extent <= 0 for extent in shape) for shape in [*operand_shapes, result_shape]
            )
            or producer_info[0] != operand_shapes[use.index]
            or any(str(value.type.get_element_type()) != dtype for value in values)
            or any(
                shape[dimension] != result_shape[dimension]
                for shape in operand_shapes
                for dimension in range(rank)
                if dimension != axis
            )
            or sum(shape[axis] for shape in operand_shapes) != result_shape[axis]
        ):
            continue
        body = producer.regions[0].blocks[0]
        terminator = body.last_op
        scalar_result = terminator.operands[0].owner if terminator is not None else None
        if (
            terminator is None
            or terminator.name != "linalg.yield"
            or scalar_result is body
            or not hasattr(scalar_result, "name")
        ):
            continue
        options.append(
            (
                math.prod(operand_shapes[use.index]),
                -producer_index,
                producer_index,
                concat_index,
                use.index,
                producer_info,
                operand_shapes,
                result_shape,
                axis,
                dtype,
                scalar_result.name,
            )
        )
    if not options:
        raise ValueError("changed source task has no supported sole-use pointwise-to-concat edge")
    (
        _,
        _,
        producer_index,
        concat_index,
        producer_operand,
        producer_info,
        operand_shapes,
        result_shape,
        axis,
        dtype,
        scalar_result_name,
    ) = max(options)
    producer, concat = ops[producer_index], ops[concat_index]

    probe_operand_shapes = []
    for source_shape in operand_shapes:
        probe_operand_shapes.append(tuple(min(extent, max_extent) for extent in source_shape))
    probe_result_shape = list(probe_operand_shapes[0])
    probe_result_shape[axis] = sum(shape[axis] for shape in probe_operand_shapes)
    probe_result_shape = tuple(probe_result_shape)
    shapes = {value: shape for value, shape in zip(concat.operands, probe_operand_shapes)}
    shapes[concat.results[0]] = probe_result_shape

    producer_shape, maps, _ = producer_info
    reduced_producer_shape = probe_operand_shapes[producer_operand]
    output_permutation = [expr.position for expr in maps[-1].results]
    loop_bounds = [
        reduced_producer_shape[output_permutation.index(dimension)] for dimension in range(len(producer_shape))
    ]
    for value, amap in zip(producer.operands, maps):
        needed = tuple(
            loop_bounds[expr.position]
            if isinstance(expr, AffineDimExpr)
            else expr.value + 1
            if isinstance(expr, AffineConstantExpr)
            else -1
            for expr in amap.results
        )
        original = tuple(value.type.get_shape())
        if len(needed) != len(original) or any(
            extent <= 0 or extent > source_extent for extent, source_extent in zip(needed, original)
        ):
            raise ValueError("pointwise producer indexing does not admit bounded concat extents")
        if value in shapes and shapes[value] != needed:
            raise ValueError("pointwise concat witness has conflicting source tensor extents")
        shapes[value] = needed

    keep = {producer, concat}
    pending = [producer]
    constant_like = {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"}
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                if owner in positions and owner not in keep and owner.name in constant_like:
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
                    (value.owner is function.body.block or value.owner in positions)
                    and value not in produced
                    and value not in boundary
                ):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) or value not in shapes for value in boundary):
        raise ValueError("pointwise concat source boundary is not a statically bounded tensor")
    bounded_elements = (
        sum(math.prod(shapes[value]) for value in boundary)
        + math.prod(reduced_producer_shape)
        + math.prod(probe_result_shape)
    )
    if bounded_elements > max_elements:
        raise ValueError("source-derived pointwise concat exceeds the tensor-element budget")

    def value_type(value):
        if not isinstance(value.type, TensorType):
            return value.type
        if value not in shapes:
            raise ValueError("pointwise concat source tensor has no witness extent")
        return TensorType(value.type.get_element_type(), shapes[value])

    block = Block(arg_types=[value_type(value) for value in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        clone = op.clone(value_mapper=mapping)
        block.add_op(clone)
        for old, new in zip(op.results, list(clone.results)):
            mapping[old] = Rewriter.replace_value_with_new_type(new, value_type(old))
    result = mapping[concat.results[0]]
    block.add_op(ReturnOp(result))
    witness = ModuleOp(
        [
            FuncOp(
                "forward", FunctionType.from_lists([value.type for value in block.args], [result.type]), Region([block])
            )
        ]
    )
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    scalar_printer = Printer(stream=scalar_stream, print_generic_format=True)
    for scalar_op in producer.regions[0].blocks[0].ops:
        scalar_printer.print_op(scalar_op)
    width = int(dtype[1:])
    spec = lambda value: {"shape": list(shapes[value]), "dtype": str(value.type.get_element_type())}
    return text, {
        "schema": "actual_source_pointwise_concat_witness_v1",
        "mechanism": "pointwise_concat",
        "source_indices": [producer_index, concat_index],
        "auxiliary_source_indices": [positions[op] for op in kept if op not in {producer, concat}],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "producer_source_index": producer_index,
        "concat_source_index": concat_index,
        "producer_concat_operand": producer_operand,
        "producer_result_scalar_operation": scalar_result_name,
        "concat_axis": axis,
        "concat_operand_count": len(operand_shapes),
        "identity_concat": len(operand_shapes) == 1,
        "source_operand_shapes": [list(shape) for shape in operand_shapes],
        "source_result_shape": list(result_shape),
        "probe_operand_shapes": [list(shape) for shape in probe_operand_shapes],
        "probe_result_shape": list(probe_result_shape),
        "inputs": [spec(value) for value in boundary],
        "outputs": [spec(concat.results[0])],
        "output": spec(concat.results[0]),
        "probe_intermediate_payload_bytes": math.prod(reduced_producer_shape) * ((width + 7) // 8),
        "probe_output_payload_bytes": math.prod(probe_result_shape) * ((width + 7) // 8),
        "bounded_tensor_elements": bounded_elements,
        "all_source_producer_uses_preserved": True,
        "geometry": "actual concat axis and operand order; uniformly reduced non-concat extents and independently reduced segment extents",
        "scope": "one exact source-cloned sole-use pointwise producer fused into its static concat copy",
    }
