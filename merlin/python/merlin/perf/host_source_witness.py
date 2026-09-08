"""Host-owned extraction and independent evaluation of a small actual source chain.

This is a numerical mechanism witness, not a timing capsule or whole-layer surrogate.
Scalar regions and indexing maps are cloned from source; only statically related tensor
extents are reduced. Unsupported operations abstain instead of borrowing candidate semantics.
"""
from __future__ import annotations

import hashlib
import io
from itertools import product
import math
from typing import Any, Sequence


def _props(op, key):
    return {**op.attributes, **op.properties}.get(key)


def _info(op, *, allow_gather: bool = False, allow_quantized_epilogue: bool = False):
    from xdsl.dialects.builtin import TensorType
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    if op.name != "linalg.generic" or len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
        return None
    shape = tuple(op.results[0].type.get_shape())
    maps = [item.data for item in _props(op, "indexing_maps")]
    counts = list(_props(op, "operandSegmentSizes").get_values())
    if (counts[-1] != 1 or (len(shape) == 0 and not allow_quantized_epilogue) or any(n <= 0 for n in shape)
            or [a.data.value for a in _props(op, "iterator_types")] != ["parallel"] * len(shape)):
        return None
    output = maps[-1]
    if (output.num_symbols or output.num_dims != len(shape)
            or len(output.results) != len(shape)
            or not all(isinstance(e, AffineDimExpr) for e in output.results)
            or sorted(e.position for e in output.results) != list(range(len(shape)))):
        return None
    for amap in maps:
        if amap.num_symbols or not all(isinstance(e, (AffineDimExpr, AffineConstantExpr)) for e in amap.results):
            return None
    body = op.regions[0].blocks[0]
    if list(body.args[-1].uses):
        return None
    supported = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                 "arith.negf",
                 "arith.addi", "arith.subi", "arith.muli", "arith.extsi", "arith.extui",
                 "arith.trunci", "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui",
                 "math.tanh", "math.erf", "math.sqrt", "math.rsqrt", "math.exp", "linalg.yield"}
    if allow_gather:
        supported |= {"linalg.index", "arith.index_cast", "tensor.extract"}
    if allow_quantized_epilogue:
        supported |= {"arith.maximumf", "arith.minimumf", "math.roundeven"}
    if any(o.name not in supported or o.regions for o in body.ops):
        return None
    return shape, maps, counts[0]


def _generic_reduction_info(op):
    """Return the statically bounded direct-generic reduction contract, or abstain."""
    from xdsl.dialects.builtin import TensorType
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    if op.name != "linalg.generic" or len(op.results) != 1:
        return None
    maps_attr = _props(op, "indexing_maps")
    iterators_attr = _props(op, "iterator_types")
    segments = _props(op, "operandSegmentSizes")
    if maps_attr is None or iterators_attr is None or segments is None:
        return None
    maps = [item.data for item in maps_attr]
    counts = list(segments.get_values())
    if len(counts) < 2 or counts[-1] != 1:
        return None
    n_in = counts[0]
    if len(op.operands) != n_in + 1 or len(maps) != n_in + 1:
        return None
    values = [*op.operands, *op.results]
    if any(not isinstance(value.type, TensorType) for value in values):
        return None
    if (op.operands[-1].type != op.results[0].type
            or any(any(extent <= 0 for extent in value.type.get_shape()) for value in values)):
        return None
    iterator_types = [item.data.value for item in iterators_attr]
    if (not iterator_types or "reduction" not in iterator_types
            or any(kind not in {"parallel", "reduction"} for kind in iterator_types)):
        return None
    rank = len(iterator_types)
    if any(amap.num_symbols or amap.num_dims != rank for amap in maps):
        return None
    if any(len(amap.results) != len(value.type.get_shape())
           for amap, value in zip(maps, op.operands)):
        return None
    simple = (AffineDimExpr, AffineConstantExpr)
    if any(not all(isinstance(expr, simple) for expr in amap.results) for amap in maps):
        return None
    parallel = [index for index, kind in enumerate(iterator_types) if kind == "parallel"]
    reduction = [index for index, kind in enumerate(iterator_types) if kind == "reduction"]
    output = maps[-1]
    if (len(output.results) != len(parallel)
            or not all(isinstance(expr, AffineDimExpr) for expr in output.results)
            or sorted(expr.position for expr in output.results) != parallel):
        return None
    bounds: list[int | None] = [None] * rank
    for amap, value in zip(maps, op.operands):
        for extent, expr in zip(value.type.get_shape(), amap.results):
            if isinstance(expr, AffineDimExpr):
                old = bounds[expr.position]
                if old is not None and old != extent:
                    return None
                bounds[expr.position] = extent
            elif not 0 <= expr.value < extent:
                return None
    if any(bound is None or bound <= 0 for bound in bounds):
        return None
    body = op.regions[0].blocks[0]
    supported = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                 "arith.addi", "arith.subi", "arith.muli", "arith.extsi", "arith.extui",
                 "arith.trunci", "linalg.yield"}
    if (len(body.args) != n_in + 1 or not list(body.args[-1].uses)
            or any(nested.name not in supported or nested.regions for nested in body.ops)):
        return None
    dtype = str(op.results[0].type.get_element_type())
    if dtype not in {"f32", "i8", "i16", "i32", "i64"}:
        return None
    return {"maps": maps, "n_in": n_in, "iterator_types": iterator_types,
            "parallel_dimensions": parallel, "reduction_dimensions": reduction,
            "iteration_shape": tuple(int(bound) for bound in bounds)}


def extract_insert_slice(source_text: str, source_indices: Sequence[int], *,
                         max_extent: int = 3, max_elements: int = 4096) -> tuple[str, dict[str, Any]]:
    """Clone one static unit-stride insertion with source-derived bounded padding geometry."""
    from xdsl.dialects.builtin import DenseArrayBase, FunctionType, ModuleOp, TensorType, i64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter
    from merlin.frontends.linalg_mlir import parse_mlir_text

    if (type(max_extent) is not int or not 1 <= max_extent <= 8
            or type(max_elements) is not int or not 1 <= max_elements <= 16_384):
        raise ValueError("insert slice witness requires bounded extents and tensor elements")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("insert slice witness requires one unoutlined source function")
    function = functions[0]
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("insert slice source indices outside source function")

    options = []
    for index in sorted(permitted):
        insertion = ops[index]
        if insertion.name != "tensor.insert_slice" or len(insertion.operands) != 2 or len(insertion.results) != 1:
            continue
        source, destination = insertion.operands
        values = [source, destination, insertion.results[0]]
        if any(not isinstance(value.type, TensorType) for value in values) or source is destination:
            continue
        source_shape = tuple(source.type.get_shape())
        destination_shape = tuple(destination.type.get_shape())
        try:
            offsets, sizes, strides = [tuple(_props(insertion, key).get_values()) for key in
                                       ("static_offsets", "static_sizes", "static_strides")]
        except (AttributeError, TypeError):
            continue
        dtype = str(source.type.get_element_type())
        if (not source_shape or len(source_shape) != len(destination_shape)
                or tuple(insertion.results[0].type.get_shape()) != destination_shape
                or len(offsets) != len(source_shape) or sizes != source_shape
                or strides != (1,) * len(source_shape)
                or any(extent <= 0 for extent in (*source_shape, *destination_shape))
                or any(offset < 0 or offset + size > extent
                       for offset, size, extent in zip(offsets, sizes, destination_shape))
                or any(str(value.type.get_element_type()) != dtype for value in values)
                or dtype not in {"f32", "i8", "i16", "i32", "i64"}):
            continue
        options.append((math.prod(destination_shape), index, source_shape,
                        destination_shape, offsets, dtype))
    if not options:
        raise ValueError("no supported static insert slice in changed source task")
    _, index, source_shape, destination_shape, offsets, dtype = max(options)
    insertion = ops[index]
    source, destination = insertion.operands
    reduced_source = tuple(min(extent, max_extent) for extent in source_shape)
    leading = tuple(min(offset, max_extent) for offset in offsets)
    trailing = tuple(min(extent - offset - size, max_extent)
                     for extent, offset, size in zip(destination_shape, offsets, source_shape))
    reduced_destination = tuple(before + size + after
                                for before, size, after in zip(leading, reduced_source, trailing))
    if max(math.prod(reduced_source), math.prod(reduced_destination)) > max_elements:
        raise ValueError("source-derived reduced insertion exceeds the tensor-element budget")
    shapes = {source: reduced_source, destination: reduced_destination,
              insertion.results[0]: reduced_destination}

    keep = {insertion}
    pending = [insertion]
    constant_like = {"arith.constant", "tensor.splat", "linalg.fill"}
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                if owner in positions and owner not in keep and owner.name in constant_like:
                    keep.add(owner)
                    pending.append(owner)
    if destination.owner in positions and destination.owner not in keep:
        raise ValueError("insert slice destination is not a defined boundary or constant fill")
    kept = [op for op in ops if op in keep]
    for op in reversed(kept):
        if op.name in {"tensor.splat", "linalg.fill"} and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
    produced = {value for op in kept for value in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if ((value.owner is function.body.block or value.owner in positions)
                        and value not in produced and value not in boundary):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) for value in boundary):
        raise ValueError("non-tensor insert slice boundary is unsupported")

    def value_type(value):
        if not isinstance(value.type, TensorType):
            return value.type
        if value not in shapes:
            raise ValueError("insert slice source tensor has unconstrained witness shape")
        return TensorType(value.type.get_element_type(), shapes[value])

    block = Block(arg_types=[value_type(value) for value in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        cloned = op.clone(value_mapper=mapping)
        if op is insertion:
            cloned.properties["static_offsets"] = DenseArrayBase.from_list(i64, leading)
            cloned.properties["static_sizes"] = DenseArrayBase.from_list(i64, reduced_source)
            cloned.properties["static_strides"] = DenseArrayBase.from_list(
                i64, [1] * len(reduced_source))
        block.add_op(cloned)
        for old, new in zip(op.results, list(cloned.results)):
            mapping[old] = Rewriter.replace_value_with_new_type(new, value_type(old))
    result = mapping[insertion.results[0]]
    block.add_op(ReturnOp(result))
    witness = ModuleOp([FuncOp("forward", FunctionType.from_lists(
        [value.type for value in block.args], [result.type]), Region([block]))])
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    width = int(dtype[1:])
    output = {"shape": list(reduced_destination), "dtype": dtype}
    return text, {"schema": "actual_source_insert_slice_witness_v1",
        "mechanism": "insert_slice", "source_indices": [index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_source_shape": list(source_shape),
        "source_destination_shape": list(destination_shape),
        "source_offsets": list(offsets), "probe_source_shape": list(reduced_source),
        "probe_destination_shape": list(reduced_destination), "probe_offsets": list(leading),
        "inputs": [{"shape": list(value.type.get_shape()),
                    "dtype": str(value.type.get_element_type())} for value in block.args],
        "outputs": [output], "output": output,
        "probe_inserted_payload_bytes": math.prod(reduced_source) * ((width + 7) // 8),
        "probe_destination_payload_bytes": math.prod(reduced_destination) * ((width + 7) // 8),
        "geometry": "source-derived leading/interior/trailing extents reduced independently",
        "scope": "one cloned static unit-stride insertion with actual dtype, initializer, and padding topology"}


def extract_bounded_gather(source_text: str, source_indices: Sequence[int], *,
                           max_elements: int = 1024, max_work: int = 20_000,
                           max_operations: int = 64) -> tuple[str, dict[str, Any]]:
    """Extract a bounded actual gather→views→pointwise mechanism, never a full layer.

    Keeps original geometry and scalar semantics when the isolated mechanism is already small.
    Large extents abstain; no model-specific geometry substitution or candidate helper is used.
    Every original gather/path use must remain on the sole-use path. Backward slicing includes
    bounded source-owned index generation and padding, stopping at unsupported producers as tensor
    inputs. Contractions/reductions/control flow are never cloned into this witness.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from merlin.frontends.linalg_mlir import parse_mlir_text

    if (type(max_elements) is not int or not 1 <= max_elements <= 4096 or
            type(max_operations) is not int or not 1 <= max_operations <= 128 or
            type(max_work) is not int or not 1 <= max_work <= 100_000):
        raise ValueError("gather extraction requires bounded element, operation and work limits")
    module = parse_mlir_text(source_text)
    functions = [o for o in module.body.block.ops if o.name == "func.func" and len(o.body.blocks)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("gather witness requires one unoutlined source function")
    function = functions[0]
    ops = [o for o in function.body.block.ops if o.name != "func.return"]
    positions = {op: i for i, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(i) is not int or not 0 <= i < len(ops) for i in permitted):
        raise ValueError("gather source indices are outside the source function")
    views = {"tensor.expand_shape", "tensor.collapse_shape", "linalg.transpose"}
    constructors = {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill",
                    "tensor.insert_slice"}

    def supported(op):
        if op.name == "linalg.generic":
            return _info(op, allow_gather=True) is not None
        return op.name in views | constructors and not op.regions

    options = []
    for index in sorted(permitted):
        gather = ops[index]
        if (_info(gather, allow_gather=True) is None or
                sum(o.name == "tensor.extract" for o in gather.walk()) != 1):
            continue
        chain = [gather]
        has_pointwise = False
        while len(chain) < max_operations:
            uses = list(chain[-1].results[0].uses)
            if len(uses) != 1:
                break
            use = uses[0]
            following = use.operation
            if positions.get(following) not in permitted or len(following.results) != 1:
                break
            info = _info(following)
            if following.name in views and use.index == 0:
                pass
            elif info is not None and use.index < info[2]:
                has_pointwise = True
            else:
                break
            chain.append(following)
        if not has_pointwise:
            continue
        keep = set(chain)
        pending = list(chain)
        while pending and len(keep) <= max_operations:
            for nested in pending.pop().walk():
                for value in nested.operands:
                    owner = value.owner
                    if (positions.get(owner) in permitted and owner not in keep and supported(owner)):
                        keep.add(owner)
                        pending.append(owner)
        if len(keep) > max_operations:
            continue
        kept = [op for op in ops if op in keep]
        produced = {v for op in kept for v in op.results}
        boundary = []
        for op in kept:
            for nested in op.walk():
                for value in nested.operands:
                    if value.owner is function.body.block or value.owner in positions:
                        if value not in produced and value not in boundary:
                            boundary.append(value)
        if any(not isinstance(v.type, TensorType) for v in boundary):
            continue
        tensors = [v for v in [*boundary, *produced] if isinstance(v.type, TensorType)]
        shapes = [tuple(v.type.get_shape()) for v in tensors]
        if any(any(n <= 0 for n in shape) or math.prod(shape) > max_elements for shape in shapes):
            continue
        work = sum(math.prod(shape) for shape in shapes)
        scalar_steps = sum(math.prod(op.results[0].type.get_shape()) * len(list(op.regions[0].blocks[0].ops))
                           for op in kept if op.name == "linalg.generic")
        if work > max_work or scalar_steps > max_work:
            continue
        # The output is a proper subgraph boundary. Selecting a complete source application is
        # forbidden even if it happens to be small or contains only movement/pointwise work.
        final_uses = list(chain[-1].results[0].uses)
        if len(final_uses) != 1 or final_uses[0].operation.name == "func.return":
            continue
        def outside(value):
            for use in value.uses:
                owner = use.operation
                while owner not in positions and owner.parent_op() is not function:
                    owner = owner.parent_op()
                    if owner is None:
                        return True
                if owner not in keep:
                    return True
            return False
        if any(outside(value) for value in produced if value is not chain[-1].results[0]):
            continue
        options.append((math.prod(gather.results[0].type.get_shape()), index, chain, kept, boundary, work, scalar_steps))
    if not options:
        raise ValueError("changed task has no bounded proper gather→views→pointwise subgraph")
    _, index, chain, kept, boundary, work, scalar_steps = max(options, key=lambda row: (row[0], -row[1]))
    block = Block(arg_types=[v.type for v in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        block.add_op(op.clone(value_mapper=mapping))
    result = mapping[chain[-1].results[0]]
    block.add_op(ReturnOp(result))
    witness = ModuleOp([FuncOp("forward", FunctionType.from_lists(
        [v.type for v in boundary], [result.type]), Region([block]))])
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    spec = lambda v: {"shape": list(v.type.get_shape()), "dtype": str(v.type.get_element_type())}
    return text, {"schema": "actual_source_bounded_gather_witness_v1", "mechanism": "bounded_gather",
                  "source_indices": [positions[op] for op in kept],
                  "gather_source_index": index, "consumer_source_indices": [positions[o] for o in chain[1:]],
                  "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                  "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "inputs": [spec(v) for v in boundary], "outputs": [spec(result)],
                  "bounded_tensor_elements": work, "max_tensor_elements": max_elements,
                  "bounded_reference_scalar_steps": scalar_steps,
                  "all_source_intermediate_uses_preserved": True,
                  "geometry": "original bounded mechanism extents; no full layer or model",
                  "scope": "source-cloned gather/index/padding/views/scalars; proper subgraph only"}


def extract_pointwise_concat(source_text: str, source_indices: Sequence[int], *,
                             max_extent: int = 3,
                             max_elements: int = 4096) -> tuple[str, dict[str, Any]]:
    """Clone one sole-use pointwise producer and its static concat consumer.

    The concat axis and segment order come from the actual source.  Every non-concatenated
    dimension is reduced uniformly, while each concatenated segment is reduced independently.
    A producer with any use other than the selected concat is refused: recomputing it in one
    concat copy loop would otherwise silently drop or duplicate a source-visible value.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter
    from merlin.frontends.linalg_mlir import parse_mlir_text

    if (type(max_extent) is not int or not 1 <= max_extent <= 8
            or type(max_elements) is not int or not 1 <= max_elements <= 16_384):
        raise ValueError("pointwise concat witness requires bounded extents and tensor elements")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops
                 if op.name == "func.func" and len(op.body.blocks)]
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
        if (concat.name != "tensor.concat" or concat_index not in permitted
                or len(concat.results) != 1 or len(concat.operands) < 2
                or not 0 <= use.index < len(concat.operands)
                or concat.operands[use.index] is not producer.results[0]):
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
        if (rank == 0 or not 0 <= axis < rank or dtype not in supported_dtypes
                or any(len(shape) != rank or any(extent <= 0 for extent in shape)
                       for shape in [*operand_shapes, result_shape])
                or producer_info[0] != operand_shapes[use.index]
                or any(str(value.type.get_element_type()) != dtype for value in values)
                or any(shape[dimension] != result_shape[dimension]
                       for shape in operand_shapes for dimension in range(rank)
                       if dimension != axis)
                or sum(shape[axis] for shape in operand_shapes) != result_shape[axis]):
            continue
        body = producer.regions[0].blocks[0]
        terminator = body.last_op
        scalar_result = terminator.operands[0].owner if terminator is not None else None
        if (terminator is None or terminator.name != "linalg.yield"
                or scalar_result is body or not hasattr(scalar_result, "name")):
            continue
        options.append((math.prod(operand_shapes[use.index]), -producer_index,
                        producer_index, concat_index, use.index, producer_info,
                        operand_shapes, result_shape, axis, dtype,
                        scalar_result.name))
    if not options:
        raise ValueError("changed source task has no supported sole-use pointwise-to-concat edge")
    (_, _, producer_index, concat_index, producer_operand, producer_info,
     operand_shapes, result_shape, axis, dtype, scalar_result_name) = max(options)
    producer, concat = ops[producer_index], ops[concat_index]

    probe_operand_shapes = []
    for source_shape in operand_shapes:
        probe_operand_shapes.append(tuple(
            min(extent, max_extent) for extent in source_shape))
    probe_result_shape = list(probe_operand_shapes[0])
    probe_result_shape[axis] = sum(shape[axis] for shape in probe_operand_shapes)
    probe_result_shape = tuple(probe_result_shape)
    shapes = {value: shape for value, shape in zip(concat.operands, probe_operand_shapes)}
    shapes[concat.results[0]] = probe_result_shape

    producer_shape, maps, _ = producer_info
    reduced_producer_shape = probe_operand_shapes[producer_operand]
    output_permutation = [expr.position for expr in maps[-1].results]
    loop_bounds = [reduced_producer_shape[output_permutation.index(dimension)]
                   for dimension in range(len(producer_shape))]
    for value, amap in zip(producer.operands, maps):
        needed = tuple(loop_bounds[expr.position] if isinstance(expr, AffineDimExpr)
                       else expr.value + 1 if isinstance(expr, AffineConstantExpr) else -1
                       for expr in amap.results)
        original = tuple(value.type.get_shape())
        if (len(needed) != len(original) or any(extent <= 0 or extent > source_extent
                                               for extent, source_extent in zip(needed, original))):
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
                if ((value.owner is function.body.block or value.owner in positions)
                        and value not in produced and value not in boundary):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) or value not in shapes for value in boundary):
        raise ValueError("pointwise concat source boundary is not a statically bounded tensor")
    bounded_elements = (sum(math.prod(shapes[value]) for value in boundary)
                        + math.prod(reduced_producer_shape) + math.prod(probe_result_shape))
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
    witness = ModuleOp([FuncOp("forward", FunctionType.from_lists(
        [value.type for value in block.args], [result.type]), Region([block]))])
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    scalar_printer = Printer(stream=scalar_stream, print_generic_format=True)
    for scalar_op in producer.regions[0].blocks[0].ops:
        scalar_printer.print_op(scalar_op)
    width = int(dtype[1:])
    spec = lambda value: {"shape": list(shapes[value]),
                          "dtype": str(value.type.get_element_type())}
    return text, {
        "schema": "actual_source_pointwise_concat_witness_v1",
        "mechanism": "pointwise_concat",
        "source_indices": [producer_index, concat_index],
        "auxiliary_source_indices": [positions[op] for op in kept
                                     if op not in {producer, concat}],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "producer_source_index": producer_index,
        "concat_source_index": concat_index,
        "producer_concat_operand": producer_operand,
        "producer_result_scalar_operation": scalar_result_name,
        "concat_axis": axis,
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


def extract_pointwise_chain(source_text: str, source_indices: Sequence[int], *,
                            max_extent: int = 3, max_operations: int = 8,
                            mechanism: str = "chain") -> tuple[str, dict[str, Any]]:
    """Extract the largest single-use equal-domain chain in a host-selected changed task.

    Reuses the existing section extractors' SSA cloning/boundary convention. Input constants
    remain actual source constants; boundary tensors become explicit function arguments.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter
    from merlin.frontends.linalg_mlir import parse_mlir_text
    if not 1 <= max_extent <= 8 or not 2 <= max_operations <= 16:
        raise ValueError("source witness must use bounded small extents and chain length")
    if mechanism not in {"chain", "fanout", "quantized_epilogue"}:
        raise ValueError("unsupported source witness mechanism")
    def info(op):
        return _info(op, allow_quantized_epilogue=mechanism == "quantized_epilogue")
    module = parse_mlir_text(source_text)
    functions = [o for o in module.body.block.ops if o.name == "func.func" and len(o.body.blocks)]
    if len(functions) != 1:
        raise ValueError("source witness requires one unoutlined source function")
    ops = [o for o in functions[0].body.block.ops if o.name != "func.return"]
    positions = {op: i for i, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(not isinstance(i, int) or not 0 <= i < len(ops) for i in permitted):
        raise ValueError("source operation indices are outside the source function")
    pairs = {}
    for index in permitted:
        op = ops[index]
        current_info = info(op)
        uses = list(op.results[0].uses) if current_info else []
        if not current_info or len(uses) != 1:
            continue
        use = uses[0]
        consumer = use.operation
        cindex = positions.get(consumer)
        following = info(consumer)
        if cindex not in permitted or not following or following[0] != current_info[0] or use.index >= following[2]:
            continue
        mapping = following[1][use.index]
        if (len(mapping.results) == len(current_info[0]) and
                all(isinstance(e, AffineDimExpr) for e in mapping.results) and
                sorted(e.position for e in mapping.results) == list(range(len(current_info[0])))):
            pairs[index] = cindex
    chains = []
    for start in pairs:
        if start in pairs.values():
            continue
        chain = [start]
        while chain[-1] in pairs and len(chain) < max_operations:
            chain.append(pairs[chain[-1]])
        chains.append(chain)
    fanout = None
    if mechanism == "fanout":
        stars = []
        for index in permitted:
            current_info = info(ops[index])
            if current_info is None:
                continue
            uses = list(ops[index].results[0].uses)
            consumers = {positions.get(use.operation) for use in uses}
            if len(consumers) < 2 or not consumers <= permitted or len(consumers)+1 > max_operations:
                continue
            if any(info(use.operation) is None or info(use.operation)[0] != current_info[0]
                   or use.index >= info(use.operation)[2] for use in uses):
                continue
            # Preserve every actual root use, including repeated operands of one consumer.
            # Boundary results keep both branches live; no synthetic consumer is invented.
            stars.append((math.prod(current_info[0]), index, sorted(consumers), uses))
        if not stars:
            raise ValueError("changed host task has no supported complete-use pointwise fanout")
        _, root_index, leaves, uses = max(stars, key=lambda item: item[0])
        chain = sorted([root_index, *leaves])
        result_indices = leaves
        fanout = {"root_source_index": root_index,
                  "uses": sorted([[positions[use.operation], use.index] for use in uses]),
                  "all_source_root_uses_preserved": True,
                  "root_dtype": str(ops[root_index].results[0].type.get_element_type())}
    else:
        if not chains:
            raise ValueError("changed host task has no supported single-consumer pointwise chain")
        chain = max(chains, key=lambda c: sum(math.prod(info(ops[i])[0]) for i in c[:-1]))
        result_indices = [chain[-1]]
    selected = {ops[i] for i in chain}
    constant_like = {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"}
    keep = set(selected)
    pending = list(selected)
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                scalar_tensor = (mechanism == "quantized_epilogue" and owner in positions
                                 and info(owner) is not None and info(owner)[0] == ())
                if owner in positions and owner not in keep and (owner.name in constant_like or scalar_tensor):
                    keep.add(owner)
                    pending.append(owner)
    kept = [op for op in ops if op in keep]
    produced = {v for op in kept for v in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if value.owner is functions[0].body.block or value.owner in positions:
                    if value not in produced and value not in boundary:
                        boundary.append(value)
    if any(not isinstance(v.type, TensorType) for v in boundary):
        raise ValueError("non-tensor source capture is not a supported native witness ABI")
    shapes = {}
    for index in chain:
        op = ops[index]
        shape, maps, _ = info(op)
        reduced = tuple(min(n, max_extent) for n in shape)
        permutation = [e.position for e in maps[-1].results]
        bounds = [reduced[permutation.index(d)] for d in range(len(shape))]
        shapes[op.results[0]] = reduced
        for value, amap in zip(op.operands, maps):
            needed = tuple(bounds[e.position] if isinstance(e, AffineDimExpr) else e.value + 1 for e in amap.results)
            if any(n <= 0 or n > original for n, original in zip(needed, value.type.get_shape())):
                raise ValueError("source indexing does not admit bounded shape reduction")
            if value in shapes:
                needed = tuple(max(a,b) for a,b in zip(shapes[value], needed))
            shapes[value] = needed
    for op in reversed(kept):
        if op.name == "linalg.fill" and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
        if mechanism == "quantized_epilogue":
            for value in (*op.operands, *op.results):
                if isinstance(value.type, TensorType) and value.type.get_shape() == ():
                    shapes[value] = ()
    def value_type(value):
        if isinstance(value.type, TensorType):
            if value not in shapes:
                raise ValueError("a source tensor extent has no witness constraint")
            return TensorType(value.type.get_element_type(), shapes[value])
        return value.type
    block = Block(arg_types=[value_type(v) for v in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        clone = op.clone(value_mapper=mapping)
        block.add_op(clone)
        for old, new in zip(op.results, list(clone.results)):
            replacement = Rewriter.replace_value_with_new_type(new, value_type(old))
            mapping[old] = replacement
    results = [mapping[ops[index].results[0]] for index in result_indices]
    block.add_op(ReturnOp(*results))
    witness = ModuleOp([FuncOp("forward", FunctionType.from_lists([v.type for v in block.args], [v.type for v in results]), Region([block]))])
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    printer = Printer(stream=scalar_stream, print_generic_format=True)
    for index in chain:
        for scalar in ops[index].regions[0].blocks[0].ops:
            printer.print_op(scalar)
    outputs = [{"shape": list(result.type.get_shape()), "dtype": str(result.type.get_element_type())}
               for result in results]
    if fanout is not None:
        fanout["probe_shape"] = list(shapes[ops[fanout["root_source_index"]].results[0]])
    return text, {
        "schema": "actual_source_pointwise_witness_v1", "source_indices": chain,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "source_shapes": [list(info(ops[i])[0]) for i in chain],
        "probe_shapes": [list(shapes[ops[i].results[0]]) for i in chain],
        "inputs": [{"shape": list(shapes[v]), "dtype": str(v.type.get_element_type())} for v in boundary],
        "output": outputs[0], "outputs": outputs,
        "mechanism": mechanism, "fanout": fanout,
        **({"input_source_values": [{"source_op_index": positions.get(v.owner),
                                      "entry_argument_index": v.index if v.owner is functions[0].body.block else None,
                                      "result_index": v.index} for v in boundary],
            "source_input_shapes": [list(v.type.get_shape()) for v in boundary],
            "auxiliary_source_indices": [positions[op] for op in kept if op not in selected],
            "typed_intermediate_rounding_preserved": True} if mechanism == "quantized_epilogue" else {}),
        "scope": "same scalar regions and maps at reduced extents; no timing or whole-model correctness claim",
    }


def pointwise_scalar_expression(source_text: str) -> dict[str, Any]:
    """Recover a typed scalar DAG from a reduced, source-cloned pointwise witness.

    No algebraic normalization: every f32 rounding boundary and operand order
    survives. The receipt is not a claim that reduced channel extents imply a
    uniform full-layer scale; callers must retain the extraction's source maps.
    """
    import struct
    from merlin.frontends.linalg_mlir import parse_mlir_text
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("scalar DAG requires one source witness function")
    function = functions[0]
    env = {arg: {"kind": "input", "index": index, "dtype": str(arg.type.get_element_type())}
           for index, arg in enumerate(function.body.block.args)}
    def scalar(op, scope):
        dtype = str(op.results[0].type)
        if op.name == "arith.constant":
            value = _props(op, "value").value.data
            if dtype == "f32":
                return {"kind": "constant", "dtype": dtype,
                        "bits": struct.pack("<f", value).hex(), "value": float(value)}
            return {"kind": "constant", "dtype": dtype, "value": int(value)}
        return {"kind": "operation", "opcode": op.name, "dtype": dtype,
                "operands": [scope.get(value, env.get(value)) for value in op.operands]}
    for op in function.body.block.ops:
        if op.name == "func.return":
            if len(op.operands) != 1:
                raise ValueError("scalar DAG requires one output")
            expression = env[op.operands[0]]
            def complete(node):
                return node is not None and all(complete(child) for child in node.get("operands", []))
            if not complete(expression):
                raise ValueError("scalar DAG has an unresolved source operand")
            return expression
        if op.name == "arith.constant":
            env[op.results[0]] = scalar(op, {})
        elif op.name in {"tensor.splat", "linalg.fill"}:
            env[op.results[0]] = env[op.operands[0]]
        elif op.name == "tensor.empty":
            env[op.results[0]] = None
        elif op.name == "linalg.generic":
            info = _info(op, allow_quantized_epilogue=True)
            if info is None:
                raise ValueError("scalar DAG requires supported all-parallel source regions")
            shape, maps, count = info
            # Scalar-DAG comparison intentionally requires a single reduced point.
            if any(extent != 1 for extent in shape):
                raise ValueError("scalar DAG comparison requires one reduced output point")
            local = {arg: env[value] for arg, value in zip(op.regions[0].blocks[0].args[:count], op.operands[:count])}
            for operation in op.regions[0].blocks[0].ops:
                if operation.name == "linalg.yield":
                    env[op.results[0]] = local.get(operation.operands[0], env.get(operation.operands[0]))
                else:
                    local[operation.results[0]] = scalar(operation, local)
        else:
            raise ValueError(f"unsupported scalar DAG source operation {op.name}")
    raise ValueError("scalar DAG source has no return")


def assess_narrow_readout_equivalence(source_text: str, extraction: dict, *, capability: dict) -> dict:
    """Compare source scalar DAG against a target-edge supplied narrow-readout contract.

    Admission is exact structure, not fitted numeric samples or guessed bias/scale
    conversion. A refusal still enables exact host fusion of the source stages.
    This is a semantic selection aid, not runtime/cycle promotion authority.
    """
    import struct
    if hashlib.sha256(source_text.encode()).hexdigest() != extraction.get("probe_source_sha256"):
        raise ValueError("stale epilogue source binding")
    required = {"schema", "accumulator_dtype", "output_dtype", "scale_dtype", "clamp_min", "clamp_max", "provenance"}
    if not required <= capability.keys() or capability["schema"] != "scalar_narrow_readout_contract_v1" or not capability["provenance"]:
        raise ValueError("missing target-owned narrow readout contract")
    expression = pointwise_scalar_expression(source_text)
    def op(name, dtype, *operands):
        return {"kind": "operation", "opcode": name, "dtype": dtype, "operands": list(operands)}
    def const(value):
        return {"kind": "constant", "dtype": capability["scale_dtype"],
                "bits": struct.pack("<f", value).hex(), "value": float(value)}
    nodes = []
    def walk(node):
        nodes.append(node)
        for child in node.get("operands", []): walk(child)
    walk(expression)
    accumulators = [node for node in nodes if node.get("kind") == "input" and node.get("dtype") == capability["accumulator_dtype"]]
    multiplications = [node for node in nodes if node.get("opcode") == "arith.mulf"]
    reasons = []
    expected = None
    if len(accumulators) == 1 and len(multiplications) == 1:
        multiplication = multiplications[0]
        cast = op("arith.sitofp", capability["scale_dtype"], accumulators[0])
        if multiplication["operands"][0] == cast:
            scale = multiplication["operands"][1]
            if scale.get("kind") not in {"constant", "input"} or scale.get("dtype") != capability["scale_dtype"]:
                reasons.append("scale_is_not_one_typed_scalar_operand")
            else:
                expected = op("arith.fptosi", capability["output_dtype"],
                    op("arith.minimumf", capability["scale_dtype"],
                       op("arith.maximumf", capability["scale_dtype"],
                          op("math.roundeven", capability["scale_dtype"], multiplication), const(capability["clamp_min"])),
                       const(capability["clamp_max"])))
                if scale.get("kind") == "input":
                    original_shape = extraction.get("source_input_shapes", [])[scale["index"]]
                    if math.prod(original_shape) != 1:
                        reasons.append("source_scale_is_not_uniform_per_command_requires_channel_partition")
        else:
            reasons.append("accumulator_conversion_and_scale_dataflow_do_not_match")
    else:
        if len(accumulators) != 1: reasons.append("not_one_integer_accumulator_input")
        if len(multiplications) != 1: reasons.append("multiple_rounded_float_scalings_cannot_be_reassociated")
    if any(node.get("opcode") == "arith.addf" for node in nodes):
        reasons.append("floating_add_bias_or_zero_point_has_no_integer_bias_equivalence_proof")
    if expected != expression:
        reasons.append("typed_scalar_DAG_differs_from_native_scale_round_clamp")
    return {"status": "eligible_exact_scalar_DAG" if not reasons else "refused_exact_native_readout",
            "reasons": reasons, "source_sha256": extraction["source_sha256"],
            "probe_source_sha256": extraction["probe_source_sha256"], "source_indices": extraction["source_indices"],
            "source_expression": expression, "target_contract": capability,
            "exact_host_fusion_preserving_stages": "permitted_in_principle_requires_emission_and_numerical_witness",
            "full_model_numerics": "UNPROVEN", "performance_promotion": False}


def extract_generic_reduction(source_text: str, source_indices: Sequence[int], *,
                              max_extent: int = 3) -> tuple[str, dict[str, Any]]:
    """Clone one direct ``linalg.generic`` reduction at bounded actual-source extents.

    The affine maps, iterator order, initializer, and typed scalar body are retained. This is
    the source shape emitted for depthwise contractions, where there is no separate pointwise
    producer for :func:`extract_pointwise_reduction` to select.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter
    from merlin.frontends.linalg_mlir import parse_mlir_text
    if not 1 <= max_extent <= 8:
        raise ValueError("generic reduction witness extent must lie in [1,8]")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("generic reduction witness requires one unoutlined source function")
    function = functions[0]
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("generic reduction source indices outside source function")
    options = []
    for index in sorted(permitted):
        info = _generic_reduction_info(ops[index])
        if info is not None:
            output_elements = math.prod(ops[index].results[0].type.get_shape())
            reduction_steps = math.prod(info["iteration_shape"][dim]
                                        for dim in info["reduction_dimensions"])
            options.append((output_elements * reduction_steps, index, info))
    if not options:
        raise ValueError("no supported direct generic reduction in changed source task")
    _, index, info = max(options)
    reduction = ops[index]
    reduced_bounds = tuple(min(extent, max_extent) for extent in info["iteration_shape"])
    return _clone_bounded_reduction(source_text, function, reduction, info, reduced_bounds,
                                    entry="forward")


def _clone_bounded_reduction(source_text, function, reduction, info, reduced_bounds, *, entry):
    """Shared source cloning for a prevalidated, explicitly bounded reduction.

    Callers own domain/semantic admission. This helper preserves the operation,
    scalar body, initializer producers, and boundary identity without evaluating it.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    index = positions[reduction]

    shapes = {}
    for value, amap in zip(reduction.operands, info["maps"]):
        needed = tuple(reduced_bounds[expr.position] if isinstance(expr, AffineDimExpr)
                       else expr.value + 1 for expr in amap.results)
        original = tuple(value.type.get_shape())
        if (len(needed) != len(original)
                or any(size <= 0 or size > old for size, old in zip(needed, original))):
            raise ValueError("generic reduction map does not admit reduced extents")
        if value in shapes and shapes[value] != needed:
            raise ValueError("aliased generic reduction operands need inconsistent witness shapes")
        shapes[value] = needed
    shapes[reduction.results[0]] = shapes[reduction.operands[-1]]

    # Retain actual constant initializers. A function argument remains a boundary input. An
    # undefined tensor.empty accumulator is deliberately unsupported because it has no stable
    # independent numerical meaning.
    keep = {reduction}
    pending = [reduction]
    constant_like = {"arith.constant", "tensor.splat", "linalg.fill"}
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                if owner in positions and owner not in keep and owner.name in constant_like:
                    keep.add(owner)
                    pending.append(owner)
    if reduction.operands[-1].owner in positions and reduction.operands[-1].owner not in keep:
        raise ValueError("generic reduction initializer is not a defined boundary or constant fill")
    kept = [op for op in ops if op in keep]
    for op in reversed(kept):
        if op.name in {"tensor.splat", "linalg.fill"} and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
    produced = {value for op in kept for value in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if ((value.owner is function.body.block or value.owner in positions)
                        and value not in produced and value not in boundary):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) for value in boundary):
        raise ValueError("non-tensor generic reduction boundary is unsupported")

    def value_type(value):
        if not isinstance(value.type, TensorType):
            return value.type
        if value not in shapes:
            raise ValueError("generic reduction source tensor has unconstrained witness shape")
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
    witness = ModuleOp([FuncOp(entry, FunctionType.from_lists(
        [value.type for value in block.args], [result.type]), Region([block]))])
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    printer = Printer(stream=scalar_stream, print_generic_format=True)
    for scalar_op in reduction.regions[0].blocks[0].ops:
        printer.print_op(scalar_op)
    dtype = str(result.type.get_element_type())
    width = int(dtype[1:])
    output = {"shape": list(result.type.get_shape()), "dtype": dtype}
    return text, {"schema": "actual_source_generic_reduction_witness_v1",
        "mechanism": "generic_reduction", "source_indices": [index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "source_iteration_shape": list(info["iteration_shape"]),
        "probe_iteration_shape": list(reduced_bounds),
        "parallel_dimensions": info["parallel_dimensions"],
        "reduction_dimensions": info["reduction_dimensions"],
        "inputs": [{"shape": list(value.type.get_shape()),
                    "dtype": str(value.type.get_element_type())} for value in block.args],
        "outputs": [output], "output": output,
        "input_source_values": [
            {"kind": "entry_argument", "index": list(function.body.block.args).index(value)}
            if value.owner is function.body.block else
            {"kind": "operation_result", "source_op_index": positions[value.owner],
             "result_index": list(value.owner.results).index(value)}
            for value in boundary],
        "probe_output_payload_bytes": math.prod(result.type.get_shape()) * ((width + 7) // 8),
        "probe_reduction_steps_per_output": math.prod(
            reduced_bounds[dim] for dim in info["reduction_dimensions"]),
        "scope": "one cloned generic reduction; actual affine maps/initializer/scalar body and "
                 "lexicographic iterator order at reduced extents"}


def extract_named_reduction(source_text: str, source_indices: Sequence[int], *,
                            max_extent: int = 3) -> tuple[str, dict[str, Any]]:
    """Clone one direct ``linalg.reduce`` at bounded actual-source extents.

    The source reduction dimensions, initializer, typed scalar body, and lexicographic input
    traversal are retained. Unsupported bodies and dimension orders abstain so this witness
    exercises exactly the register-resident named-reduction lowering, rather than a synthetic
    approximation of it.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter
    from merlin.frontends.linalg_mlir import parse_mlir_text

    if not 1 <= max_extent <= 8:
        raise ValueError("named reduction witness extent must lie in [1,8]")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("named reduction witness requires one unoutlined source function")
    function = functions[0]
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("named reduction source indices outside source function")

    supported = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                 "arith.addi", "arith.subi", "arith.muli", "arith.maxsi", "arith.minsi",
                 "arith.maxui", "arith.minui", "linalg.yield"}
    options = []
    for index in sorted(permitted):
        reduction = ops[index]
        if (reduction.name != "linalg.reduce" or len(reduction.operands) != 2
                or len(reduction.results) != 1):
            continue
        input_value, initial_value = reduction.operands
        values = [input_value, initial_value, reduction.results[0]]
        if any(not isinstance(value.type, TensorType) for value in values):
            continue
        input_shape = tuple(input_value.type.get_shape())
        dimensions = tuple(_props(reduction, "dimensions").get_values())
        output_shape = tuple(extent for axis, extent in enumerate(input_shape)
                             if axis not in dimensions)
        body = reduction.regions[0].blocks[0]
        dtype = str(input_value.type.get_element_type())
        if (not input_shape or any(extent <= 0 for extent in input_shape)
                or not dimensions or dimensions != tuple(sorted(set(dimensions)))
                or any(not 0 <= dim < len(input_shape) for dim in dimensions)
                or tuple(initial_value.type.get_shape()) != output_shape
                or tuple(reduction.results[0].type.get_shape()) != output_shape
                or any(str(value.type.get_element_type()) != dtype for value in values)
                or dtype not in {"f32", "i8", "i16", "i32", "i64"}
                or len(body.args) != 2 or not list(body.args[-1].uses)
                or any(nested.name not in supported or nested.regions for nested in body.ops)):
            continue
        work = math.prod(input_shape)
        options.append((work, index, dimensions, input_shape))
    if not options:
        raise ValueError("no supported direct named reduction in changed source task")
    _, index, dimensions, input_shape = max(options)
    reduction = ops[index]
    input_value, initial_value = reduction.operands
    reduced_input_shape = tuple(min(extent, max_extent) for extent in input_shape)
    reduced_output_shape = tuple(extent for axis, extent in enumerate(reduced_input_shape)
                                 if axis not in dimensions)
    shapes = {input_value: reduced_input_shape, initial_value: reduced_output_shape,
              reduction.results[0]: reduced_output_shape}

    keep = {reduction}
    pending = [reduction]
    constant_like = {"arith.constant", "tensor.splat", "linalg.fill"}
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                if owner in positions and owner not in keep and owner.name in constant_like:
                    keep.add(owner)
                    pending.append(owner)
    if initial_value.owner in positions and initial_value.owner not in keep:
        raise ValueError("named reduction initializer is not a defined boundary or constant fill")
    kept = [op for op in ops if op in keep]
    for op in reversed(kept):
        if op.name in {"tensor.splat", "linalg.fill"} and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
    produced = {value for op in kept for value in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if ((value.owner is function.body.block or value.owner in positions)
                        and value not in produced and value not in boundary):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) for value in boundary):
        raise ValueError("non-tensor named reduction boundary is unsupported")

    def value_type(value):
        if not isinstance(value.type, TensorType):
            return value.type
        if value not in shapes:
            raise ValueError("named reduction source tensor has unconstrained witness shape")
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
    witness = ModuleOp([FuncOp("forward", FunctionType.from_lists(
        [value.type for value in block.args], [result.type]), Region([block]))])
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    printer = Printer(stream=scalar_stream, print_generic_format=True)
    for scalar_op in reduction.regions[0].blocks[0].ops:
        printer.print_op(scalar_op)
    dtype = str(result.type.get_element_type())
    width = int(dtype[1:])
    output = {"shape": list(result.type.get_shape()), "dtype": dtype}
    return text, {"schema": "actual_source_named_reduction_witness_v1",
        "mechanism": "named_reduction", "source_indices": [index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "source_input_shape": list(input_shape), "probe_input_shape": list(reduced_input_shape),
        "reduction_dimensions": list(dimensions),
        "inputs": [{"shape": list(value.type.get_shape()),
                    "dtype": str(value.type.get_element_type())} for value in block.args],
        "outputs": [output], "output": output,
        "probe_output_payload_bytes": math.prod(result.type.get_shape()) * ((width + 7) // 8),
        "probe_reduction_steps_per_output": math.prod(
            reduced_input_shape[dim] for dim in dimensions),
        "scope": "one cloned named reduction; actual dimensions/initializer/scalar body and "
                 "lexicographic input traversal at reduced extents"}


def extract_pointwise_reduction(source_text: str, source_indices: Sequence[int], *,
                                max_extent: int = 3) -> tuple[str, dict[str, Any]]:
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
    from merlin.frontends.linalg_mlir import parse_mlir_text
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
    supported = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                 "arith.addi", "arith.subi", "arith.muli", "linalg.yield"}
    for index in sorted(permitted):
        reduction = ops[index]
        if reduction.name != "linalg.reduce" or len(reduction.operands) != 2 or len(reduction.results) != 1:
            continue
        producer = reduction.operands[0].owner
        info = _info(producer) if producer in positions else None
        if (not info or positions[producer] not in permitted or len(list(producer.results[0].uses)) != 1
                or any(op.name not in supported or op.regions for op in reduction.regions[0].blocks[0].ops)):
            continue
        dims = tuple(_props(reduction, "dimensions").get_values())
        shape = info[0]
        output_shape = tuple(extent for axis, extent in enumerate(shape) if axis not in dims)
        if (not dims or len(set(dims)) != len(dims) or any(not 0 <= dim < len(shape) for dim in dims)
                or tuple(reduction.results[0].type.get_shape()) != output_shape
                or tuple(reduction.operands[1].type.get_shape()) != output_shape
                or str(reduction.results[0].type.get_element_type()) not in {"f32", "i8", "i16", "i32", "i64"}):
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
    shapes = {producer.results[0]: reduced, reduction.results[0]: tuple(
        extent for axis, extent in enumerate(reduced) if axis not in dims)}
    for value, amap in zip(producer.operands, maps):
        needed = tuple(bounds[expr.position] if isinstance(expr, AffineDimExpr) else expr.value + 1
                       for expr in amap.results)
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
                if owner in positions and owner not in keep and owner.name in {
                        "arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"}:
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
                if ((value.owner is functions[0].body.block or value.owner in positions)
                        and value not in produced and value not in boundary):
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
    witness = ModuleOp([FuncOp("forward", FunctionType.from_lists(
        [value.type for value in block.args], [result.type]), Region(block))])
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
    output = {"shape": list(shapes[reduction.results[0]]),
              "dtype": str(reduction.results[0].type.get_element_type())}
    return text, {"schema": "actual_source_pointwise_reduction_witness_v1", "mechanism": "pointwise_reduction",
        "source_indices": [producer_index, reduction_index], "source_shape": list(shape), "probe_shape": list(reduced),
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "producer_result_scalar_operation": getattr(
            producer.regions[0].blocks[0].last_op.operands[0].owner, "name", None),
        "reduction_result_scalar_operation": getattr(
            reduction.regions[0].blocks[0].last_op.operands[0].owner, "name", None),
        "inputs": [{"shape": list(shapes[value]), "dtype": str(value.type.get_element_type())} for value in boundary],
        "outputs": [output], "output": output, "reduction_dimensions": list(dims),
        "source_intermediate_payload_bytes": math.prod(shape) * ((width+7)//8),
        "probe_intermediate_payload_bytes": math.prod(reduced) * ((width+7)//8),
        "all_source_producer_uses_preserved": True, "recomputation_factor": 1,
        "scope": "cloned scalar regions/maps/initializers; lexicographic full input domain at reduced extents"}


def extract_dequant_contraction(source_text: str, source_indices: Sequence[int], *,
                               max_extent: int = 3) -> tuple[str, dict[str, Any]]:
    """Clone a sole-use signed per-channel dequantization feeding a named f32 matmul.

    Preserve the original no-recomputation geometry: RHS fusion needs M=1, LHS needs N=1.
    Ranking by actual source intermediate size is only selection, not full-backend attribution.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter
    from merlin.frontends.linalg_mlir import parse_mlir_text
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
            if (len(maps) != 3 or any(amap.num_dims != 3 or amap.num_symbols for amap in maps)
                    or any(not all(isinstance(expr, AffineDimExpr) for expr in amap.results) for amap in maps)
                    or [[expr.position for expr in amap.results] for amap in maps] != [[0, 2], [2, 1], [0, 1]]):
                continue
        types = [value.type for value in (*mm.operands, *mm.results)]
        if any(not isinstance(ty, TensorType) or len(ty.get_shape()) != 2
               or str(ty.get_element_type()) != "f32" for ty in types):
            continue
        (m, k), (kk, n), output, result = [tuple(ty.get_shape()) for ty in types]
        if k != kk or output != (m, n) or result != output or min(m, k, n) <= 0:
            continue
        for side in (0, 1):
            producer = mm.operands[side].owner
            name = getattr(getattr(producer, "op_name", None), "data", getattr(producer, "name", None))
            if (name != "quant_ext.dequantize_per_channel" or positions.get(producer) not in permitted
                    or len(producer.operands) != 3 or len(producer.results) != 1
                    or len(list(producer.results[0].uses)) != 1
                    or (n if side == 0 else m) != 1):
                continue
            axis = getattr(getattr(_props(producer, "axis"), "value", None), "data", None)
            source_type, scale_type, zero_type = [value.type for value in producer.operands]
            shape = tuple(producer.results[0].type.get_shape())
            if (axis not in (0, 1) or str(producer.results[0].type.get_element_type()) != "f32"
                    or str(source_type.get_element_type()) not in {"i8", "i16", "i32"}
                    or getattr(_props(producer, "input_dtype"), "data", str(source_type.get_element_type()))
                        != str(source_type.get_element_type())
                    or str(scale_type.get_element_type()) != "f32"
                    or str(zero_type.get_element_type()) != "i32"
                    or tuple(source_type.get_shape()) != shape
                    or tuple(scale_type.get_shape()) != (shape[axis],)
                    or tuple(zero_type.get_shape()) != (shape[axis],)):
                continue
            options.append((math.prod(shape), positions[producer], index, side, axis, (m, k, n)))
    if not options:
        raise ValueError("no exact sole-use dequant-to-contraction source pair with non-amplifying geometry")
    _, producer_index, consumer_index, side, axis, geometry = max(options)
    producer, mm = ops[producer_index], ops[consumer_index]
    m, k, n = [min(extent, max_extent) for extent in geometry]
    operand_shapes = [(m, k), (k, n)]
    dq_shape = operand_shapes[side]
    boundary = [*producer.operands, mm.operands[1-side], mm.operands[2]]
    if len(set(boundary)) != len(boundary):
        raise ValueError("contraction witness shared boundary aliases require an explicit alias-preserving extractor")
    shapes = [dq_shape, (dq_shape[axis],), (dq_shape[axis],), operand_shapes[1-side], (m, n)]
    types = [TensorType(value.type.get_element_type(), shape) for value, shape in zip(boundary, shapes)]
    block = Block(arg_types=types)
    mapping = dict(zip(boundary, block.args))
    dq = producer.clone(value_mapper=mapping)
    block.add_op(dq)
    mapping[producer.results[0]] = Rewriter.replace_value_with_new_type(
        dq.results[0], TensorType(producer.results[0].type.get_element_type(), dq_shape))
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
    return text, {"schema": "actual_source_dequant_contraction_witness_v1",
        "mechanism": "dequant_contraction", "source_indices": [producer_index, consumer_index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "inputs": [{"shape": list(shape), "dtype": str(ty.get_element_type())} for shape, ty in zip(shapes, types)],
        "output": output, "outputs": [output], "source_geometry_mkn": list(geometry),
        "probe_geometry_mkn": [m, k, n], "producer_operand": side, "channel_axis": axis,
        "source_intermediate_payload_bytes": math.prod(producer.results[0].type.get_shape()) * 4,
        "probe_intermediate_payload_bytes": math.prod(dq_shape) * 4,
        "scalar_semantics": "signed convert to f32, f32 subtract zero point, f32 multiply scale; "
                            "matmul i,j,k order with distinct f32 multiply then initialized f32 add",
        "indexing_semantics": "actual cloned per-channel axis and named matmul indexing",
        "all_source_producer_uses_preserved": True, "recomputation_factor": 1,
        "scope": "exact selected source operations at reduced extents, not full-backend attribution"}


def evaluate_pointwise_source(source_text: str, inputs: Sequence[Any], *,
                              allow_quantized_epilogue: bool = False) -> list[Any]:
    """Independent concrete source semantics, rounding after each typed scalar operation.

    Index maps use xDSL's affine evaluator. Numeric values use NumPy scalar containers and
    Python libm, not candidate builders or the saturating command-buffer golden.
    """
    import numpy as np
    from merlin.frontends.linalg_mlir import parse_mlir_text
    module = parse_mlir_text(source_text)
    function = next(o for o in module.body.block.ops if o.name == "func.func")
    env = dict(zip(function.body.block.args, inputs, strict=True))
    def round_value(value, ty):
        dtype = str(ty)
        if dtype == "f32":
            return float(np.float32(value))
        if dtype == "index":
            return int(value)
        if dtype.startswith("i") and dtype[1:].isdigit():
            width = int(dtype[1:])
            unsigned = int(value) % (1 << width)
            return unsigned - (1 << width) if unsigned >= (1 << (width-1)) else unsigned
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
            lo, hi = (0, (1<<width)-1) if name.endswith("ui") else (-(1<<(width-1)), (1<<(width-1))-1)
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
                result[index] = np.float32(np.float32(np.float32(values[index]) - np.float32(zeros[channel]))
                                          * np.float32(scales[channel]))
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
                    local[operation.results[0]] = scalar(operation, [local[value] if value in local else env[value]
                                                                     for value in operation.operands])
            env[op.results[0]] = result
        elif op.name == "linalg.matmul":
            lhs, rhs, initial = [env[value] for value in op.operands]
            types = [str(value.type.get_element_type()) for value in op.operands]
            if all(dtype in {"i8", "i16", "i32", "i64"} for dtype in types):
                from xdsl.ir.affine import AffineDimExpr
                op.verify()
                maps = [item.data for item in op.get_indexing_maps()]
                if (len(maps) != 3 or any(amap.num_symbols or amap.num_dims != 3 for amap in maps)
                        or any(not all(isinstance(expr, AffineDimExpr) for expr in amap.results) for amap in maps)
                        or [[expr.position for expr in amap.results] for amap in maps] != [[0, 2], [2, 1], [0, 1]]):
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
            env[op.results[0]] = np.full(shape, fill, dtype="float32" if dtype == "f32" else f"int{dtype[1:]}")
        elif op.name in {"tensor.expand_shape", "tensor.collapse_shape"}:
            env[op.results[0]] = env[op.operands[0]].reshape(op.results[0].type.get_shape()).copy()
        elif op.name == "linalg.transpose":
            env[op.results[0]] = env[op.operands[0]].transpose(tuple(_props(op, "permutation").get_values())).copy()
        elif op.name == "tensor.insert_slice":
            if len(op.operands) != 2:
                raise ValueError("independent insert needs static same-rank offsets/sizes/strides")
            values, destination = [env[v] for v in op.operands]
            offsets, sizes, strides = [tuple(_props(op, key).get_values()) for key in
                                       ("static_offsets", "static_sizes", "static_strides")]
            if (len(offsets) != destination.ndim or sizes != values.shape or
                    len(strides) != destination.ndim or len(sizes) != destination.ndim or
                    any(off < 0 or stride <= 0 or off + (size - 1) * stride >= extent
                        for off, size, stride, extent in zip(offsets, sizes, strides, destination.shape))):
                raise ValueError("independent insert is not a bounded same-rank slice")
            result = destination.copy()
            result[tuple(slice(off, off + size * stride, stride)
                         for off, size, stride in zip(offsets, sizes, strides))] = values
            env[op.results[0]] = result
        elif op.name == "tensor.concat":
            arrays = [env[value] for value in op.operands]
            result_shape = tuple(op.results[0].type.get_shape())
            axis = _props(op, "dim").value.data
            if (not arrays or len(result_shape) == 0 or not 0 <= axis < len(result_shape)
                    or any(array.ndim != len(result_shape) for array in arrays)
                    or any(array.shape[dimension] != result_shape[dimension]
                           for array in arrays for dimension in range(len(result_shape))
                           if dimension != axis)
                    or sum(array.shape[axis] for array in arrays) != result_shape[axis]
                    or any(array.dtype != arrays[0].dtype for array in arrays)):
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
                out = np.zeros(shape, dtype="float32" if str(ty) == "f32" else f"int{str(ty)[1:]}")
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
                    local[operation.results[0]] = scalar(operation, [local[v] if v in local else env[v] for v in operation.operands], index)
            env[op.results[0]] = out
        else:
            raise ValueError(f"unsupported independent source operation {op.name}")
    raise ValueError("source witness has no return")
