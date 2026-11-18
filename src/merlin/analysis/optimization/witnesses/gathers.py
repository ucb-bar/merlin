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


def _props(op, key):
    return {**op.attributes, **op.properties}.get(key)


def bf16_bits(values):
    """Encode finite float values as IEEE bfloat16 RNE storage bits."""
    import numpy as np

    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_values(bits):
    """Decode IEEE bfloat16 storage bits into exact float32 values."""
    import numpy as np

    words = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)


def _info(op, *, allow_gather: bool = False, allow_quantized_epilogue: bool = False):
    from xdsl.dialects.builtin import TensorType
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    if op.name != "linalg.generic" or len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
        return None
    shape = tuple(op.results[0].type.get_shape())
    maps = [item.data for item in _props(op, "indexing_maps")]
    counts = list(_props(op, "operandSegmentSizes").get_values())
    if (
        counts[-1] != 1
        or (len(shape) == 0 and not allow_quantized_epilogue)
        or any(n <= 0 for n in shape)
        or [a.data.value for a in _props(op, "iterator_types")] != ["parallel"] * len(shape)
    ):
        return None
    output = maps[-1]
    if (
        output.num_symbols
        or output.num_dims != len(shape)
        or len(output.results) != len(shape)
        or not all(isinstance(e, AffineDimExpr) for e in output.results)
        or sorted(e.position for e in output.results) != list(range(len(shape)))
    ):
        return None
    for amap in maps:
        if amap.num_symbols or not all(isinstance(e, (AffineDimExpr, AffineConstantExpr)) for e in amap.results):
            return None
    body = op.regions[0].blocks[0]
    if list(body.args[-1].uses):
        return None
    supported = {
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.sitofp",
        "arith.uitofp",
        "arith.fptosi",
        "arith.fptoui",
        "math.tanh",
        "math.erf",
        "math.sqrt",
        "math.rsqrt",
        "math.exp",
        "linalg.yield",
    }
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
    if op.operands[-1].type != op.results[0].type or any(
        any(extent <= 0 for extent in value.type.get_shape()) for value in values
    ):
        return None
    iterator_types = [item.data.value for item in iterators_attr]
    if (
        not iterator_types
        or "reduction" not in iterator_types
        or any(kind not in {"parallel", "reduction"} for kind in iterator_types)
    ):
        return None
    rank = len(iterator_types)
    if any(amap.num_symbols or amap.num_dims != rank for amap in maps):
        return None
    if any(len(amap.results) != len(value.type.get_shape()) for amap, value in zip(maps, op.operands)):
        return None
    simple = (AffineDimExpr, AffineConstantExpr)
    if any(not all(isinstance(expr, simple) for expr in amap.results) for amap in maps):
        return None
    parallel = [index for index, kind in enumerate(iterator_types) if kind == "parallel"]
    reduction = [index for index, kind in enumerate(iterator_types) if kind == "reduction"]
    output = maps[-1]
    if (
        len(output.results) != len(parallel)
        or not all(isinstance(expr, AffineDimExpr) for expr in output.results)
        or sorted(expr.position for expr in output.results) != parallel
    ):
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
    supported = {
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "linalg.yield",
    }
    if (
        len(body.args) != n_in + 1
        or not list(body.args[-1].uses)
        or any(nested.name not in supported or nested.regions for nested in body.ops)
    ):
        return None
    dtype = str(op.results[0].type.get_element_type())
    if dtype not in {"f32", "i8", "i16", "i32", "i64"}:
        return None
    return {
        "maps": maps,
        "n_in": n_in,
        "iterator_types": iterator_types,
        "parallel_dimensions": parallel,
        "reduction_dimensions": reduction,
        "iteration_shape": tuple(int(bound) for bound in bounds),
    }


def extract_insert_slice(
    source_text: str, source_indices: Sequence[int], *, max_extent: int = 3, max_elements: int = 4096
) -> tuple[str, dict[str, Any]]:
    """Clone one static unit-stride insertion with source-derived bounded padding geometry."""
    from xdsl.dialects.builtin import DenseArrayBase, FunctionType, ModuleOp, TensorType, i64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if (
        type(max_extent) is not int
        or not 1 <= max_extent <= 8
        or type(max_elements) is not int
        or not 1 <= max_elements <= 16_384
    ):
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
            offsets, sizes, strides = [
                tuple(_props(insertion, key).get_values())
                for key in ("static_offsets", "static_sizes", "static_strides")
            ]
        except (AttributeError, TypeError):
            continue
        dtype = str(source.type.get_element_type())
        if (
            not source_shape
            or len(source_shape) != len(destination_shape)
            or tuple(insertion.results[0].type.get_shape()) != destination_shape
            or len(offsets) != len(source_shape)
            or sizes != source_shape
            or strides != (1,) * len(source_shape)
            or any(extent <= 0 for extent in (*source_shape, *destination_shape))
            or any(
                offset < 0 or offset + size > extent for offset, size, extent in zip(offsets, sizes, destination_shape)
            )
            or any(str(value.type.get_element_type()) != dtype for value in values)
            or dtype not in {"f32", "i8", "i16", "i32", "i64"}
        ):
            continue
        options.append((math.prod(destination_shape), index, source_shape, destination_shape, offsets, dtype))
    if not options:
        raise ValueError("no supported static insert slice in changed source task")
    _, index, source_shape, destination_shape, offsets, dtype = max(options)
    insertion = ops[index]
    source, destination = insertion.operands
    reduced_source = tuple(min(extent, max_extent) for extent in source_shape)
    leading = tuple(min(offset, max_extent) for offset in offsets)
    trailing = tuple(
        min(extent - offset - size, max_extent)
        for extent, offset, size in zip(destination_shape, offsets, source_shape)
    )
    reduced_destination = tuple(before + size + after for before, size, after in zip(leading, reduced_source, trailing))
    if max(math.prod(reduced_source), math.prod(reduced_destination)) > max_elements:
        raise ValueError("source-derived reduced insertion exceeds the tensor-element budget")
    shapes = {source: reduced_source, destination: reduced_destination, insertion.results[0]: reduced_destination}

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
                if (
                    (value.owner is function.body.block or value.owner in positions)
                    and value not in produced
                    and value not in boundary
                ):
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
            cloned.properties["static_strides"] = DenseArrayBase.from_list(i64, [1] * len(reduced_source))
        block.add_op(cloned)
        for old, new in zip(op.results, list(cloned.results)):
            mapping[old] = Rewriter.replace_value_with_new_type(new, value_type(old))
    result = mapping[insertion.results[0]]
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
    width = int(dtype[1:])
    output = {"shape": list(reduced_destination), "dtype": dtype}
    return text, {
        "schema": "actual_source_insert_slice_witness_v1",
        "mechanism": "insert_slice",
        "source_indices": [index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_source_shape": list(source_shape),
        "source_destination_shape": list(destination_shape),
        "source_offsets": list(offsets),
        "probe_source_shape": list(reduced_source),
        "probe_destination_shape": list(reduced_destination),
        "probe_offsets": list(leading),
        "inputs": [
            {"shape": list(value.type.get_shape()), "dtype": str(value.type.get_element_type())} for value in block.args
        ],
        "outputs": [output],
        "output": output,
        "probe_inserted_payload_bytes": math.prod(reduced_source) * ((width + 7) // 8),
        "probe_destination_payload_bytes": math.prod(reduced_destination) * ((width + 7) // 8),
        "geometry": "source-derived leading/interior/trailing extents reduced independently",
        "scope": "one cloned static unit-stride insertion with actual dtype, initializer, and padding topology",
    }


def extract_bounded_gather(
    source_text: str,
    source_indices: Sequence[int],
    *,
    max_elements: int = 1024,
    max_work: int = 20_000,
    max_operations: int = 64,
) -> tuple[str, dict[str, Any]]:
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

    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if (
        type(max_elements) is not int
        or not 1 <= max_elements <= 4096
        or type(max_operations) is not int
        or not 1 <= max_operations <= 128
        or type(max_work) is not int
        or not 1 <= max_work <= 100_000
    ):
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
    constructors = {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill", "tensor.insert_slice"}

    def supported(op):
        if op.name == "linalg.generic":
            return _info(op, allow_gather=True) is not None
        return op.name in views | constructors and not op.regions

    options = []
    for index in sorted(permitted):
        gather = ops[index]
        if _info(gather, allow_gather=True) is None or sum(o.name == "tensor.extract" for o in gather.walk()) != 1:
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
                    if positions.get(owner) in permitted and owner not in keep and supported(owner):
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
        scalar_steps = sum(
            math.prod(op.results[0].type.get_shape()) * len(list(op.regions[0].blocks[0].ops))
            for op in kept
            if op.name == "linalg.generic"
        )
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
        options.append(
            (math.prod(gather.results[0].type.get_shape()), index, chain, kept, boundary, work, scalar_steps)
        )
    if not options:
        raise ValueError("changed task has no bounded proper gather→views→pointwise subgraph")
    _, index, chain, kept, boundary, work, scalar_steps = max(options, key=lambda row: (row[0], -row[1]))
    block = Block(arg_types=[v.type for v in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        block.add_op(op.clone(value_mapper=mapping))
    result = mapping[chain[-1].results[0]]
    block.add_op(ReturnOp(result))
    witness = ModuleOp(
        [FuncOp("forward", FunctionType.from_lists([v.type for v in boundary], [result.type]), Region([block]))]
    )
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    spec = lambda v: {"shape": list(v.type.get_shape()), "dtype": str(v.type.get_element_type())}
    return text, {
        "schema": "actual_source_bounded_gather_witness_v1",
        "mechanism": "bounded_gather",
        "source_indices": [positions[op] for op in kept],
        "gather_source_index": index,
        "consumer_source_indices": [positions[o] for o in chain[1:]],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "inputs": [spec(v) for v in boundary],
        "outputs": [spec(result)],
        "bounded_tensor_elements": work,
        "max_tensor_elements": max_elements,
        "bounded_reference_scalar_steps": scalar_steps,
        "all_source_intermediate_uses_preserved": True,
        "geometry": "original bounded mechanism extents; no full layer or model",
        "scope": "source-cloned gather/index/padding/views/scalars; proper subgraph only",
    }


_TOP_LEVEL_SOURCE_OPERATION = re.compile(  # regex-ok: locate canonical top-level MLIR records only
    r"(?m)^    (?=(?:%[-A-Za-z0-9_.$]+(?::\d+)?\s*=|func\.return\b))"
)


_SSA_NAME = re.compile(r"%[-A-Za-z0-9_.$]+")  # regex-ok: tokenize canonical SSA names


_RESULT_NAME = re.compile(  # regex-ok: extract canonical top-level result declarations
    r"^    (%[-A-Za-z0-9_.$]+)(?::\d+)?\s*=", re.MULTILINE
)


_TENSOR_TYPE = re.compile(r"tensor<[^>]+>")  # regex-ok: tokenize canonical ranked tensor types


def _fast_top_level_source_operations(source_text: str) -> tuple[list[str], str]:
    """Split canonical one-function linalg-on-tensors source without parsing its bodies.

    This is deliberately a narrow fast path for multi-megabyte generated modules.  Operation
    indices are accepted only when the canonical four-space top-level format is internally
    consistent.  The selected slice is subsequently parsed and verified by xdsl; this scanner
    is not used as a semantic parser.
    """
    if (
        source_text.count("func.func ") != 1
        or source_text.count("func.return ") != 1
        or not source_text.lstrip().startswith("builtin.module")
    ):
        raise ValueError("view-chain witness requires one canonical unoutlined source function")
    starts = [match.start() for match in _TOP_LEVEL_SOURCE_OPERATION.finditer(source_text)]
    chunks = [
        source_text[start : (starts[index + 1] if index + 1 < len(starts) else len(source_text))]
        for index, start in enumerate(starts)
    ]
    returns = [chunk for chunk in chunks if chunk.startswith("    func.return")]
    operations = [chunk for chunk in chunks if not chunk.startswith("    func.return")]
    if len(returns) != 1 or not operations:
        raise ValueError("view-chain witness cannot index canonical source operations")
    return operations, returns[0]


def _chunk_result_name(chunk: str) -> str | None:
    match = _RESULT_NAME.search(chunk)
    return match.group(1) if match else None


def _typed_operands(chunk: str, keyword: str) -> list[tuple[str, str]]:
    match = re.search(  # regex-ok: isolate canonical linalg ins/outs operand clauses
        rf"\b{keyword}\((?P<values>[^()]*)\s+:\s+"
        rf"(?P<types>tensor<[^>]+>(?:\s*,\s*tensor<[^>]+>)*)\)",
        chunk,
    )
    if match is None:
        return []
    names = _SSA_NAME.findall(match.group("values"))
    types = _TENSOR_TYPE.findall(match.group("types"))
    if len(names) != len(types):
        raise ValueError(f"view-chain {keyword} operand/type arity is ambiguous")
    return list(zip(names, types))
