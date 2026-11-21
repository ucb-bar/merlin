"""Bound an actual integer contraction; never substitute target arithmetic or a schedule.

Extraction and independent source evaluation only. The caller must establish
before/after emitted-route correspondence and separately admit a bounded runtime.
"""

from __future__ import annotations

import hashlib
import io
from math import prod
from typing import Sequence

from xdsl.dialects.builtin import DenseArrayBase, FunctionType, ModuleOp, TensorType, i64
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Operation, Region
from xdsl.ir.affine import AffineDimExpr
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter

from merlin.analysis.optimization.witnesses.pointwise_chains import _clone_bounded_reduction
from merlin.analysis.optimization.witnesses.reductions import evaluate_pointwise_source
from merlin.analysis.programs.model_macs import _domain, _iterators
from merlin.analysis.programs.plan_evidence import _source_has_multiply_accumulate
from merlin.common import digest as _mdigest
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.development.targetgen.applications.capture_census import _terms

# ---- source_contraction_witness ----------------------------------------------------------------

_sha = _mdigest.sha256_text


def _entry(module, entry):
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("one explicit single-block source entry is required")
    return functions[0]


def _contract(op):
    """Recognize the complete typed multiply/add recurrence, not provenance tags."""
    if (
        op.name not in {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}
        or len(op.operands) != 3
        or len(op.results) != 1
    ):
        raise ValueError("selected operation is not an integer contraction")
    try:
        op.verify()
    except Exception as error:
        raise ValueError("source contraction does not verify") from error
    values = [*op.operands, *op.results]
    if any(
        not isinstance(value.type, TensorType)
        or str(value.type.get_element_type()) not in {"i8", "i16", "i32", "i64"}
        or any(type(extent) is not int or extent <= 0 for extent in value.type.get_shape())
        for value in values
    ):
        raise ValueError("contraction requires exact static signless integer tensors")
    shapes = [tuple(value.type.get_shape()) for value in values]
    lhs, rhs, output, result = shapes
    output_rank = len(output)
    if (
        output_rank < 2
        or result != output
        or values[2].type != values[3].type
        or len(lhs) not in {2, output_rank}
        or len(rhs) not in {2, output_rank}
    ):
        raise ValueError("contraction tensor domains or initialized output types disagree")
    batch, m, n = output[:-2], output[-2], output[-1]
    lhs_batched, rhs_batched = len(lhs) == output_rank, len(rhs) == output_rank
    if (
        (lhs_batched and lhs[:-2] != batch)
        or (rhs_batched and rhs[:-2] != batch)
        or lhs[-2] != m
        or rhs[-1] != n
        or lhs[-1] != rhs[-2]
    ):
        raise ValueError("contraction tensor domains or initialized output types disagree")
    k = lhs[-1]
    maps = [item.data for item in op.get_indexing_maps()]
    iterator_rank = output_rank + 1
    batch_dims = tuple(range(output_rank - 2))
    m_dim, n_dim, k_dim = output_rank - 2, output_rank - 1, output_rank
    expected_maps = [
        [*batch_dims, m_dim, k_dim] if lhs_batched else [m_dim, k_dim],
        [*batch_dims, k_dim, n_dim] if rhs_batched else [k_dim, n_dim],
        [*batch_dims, m_dim, n_dim],
    ]
    if (
        len(maps) != 3
        or any(amap.num_symbols or amap.num_dims != iterator_rank for amap in maps)
        or any(not all(isinstance(expr, AffineDimExpr) for expr in amap.results) for amap in maps)
        or [[expr.position for expr in amap.results] for amap in maps] != expected_maps
        or [item.data.value for item in op.get_iterator_types()] != [*(["parallel"] * output_rank), "reduction"]
    ):
        raise ValueError("contraction requires canonical batch...,m,n,k affine maps and iterator order")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("contraction has unsupported scalar control flow")
    block = op.regions[0].block
    operations = list(block.ops)
    if (
        len(block.args) != 3
        or not operations
        or operations[-1].name != "linalg.yield"
        or len(operations[-1].operands) != 1
    ):
        raise ValueError("contraction scalar argument/yield contract is malformed")
    allowed = {"arith.extsi", "arith.extui", "arith.trunci", "arith.muli", "arith.addi", "linalg.yield"}
    for scalar in operations:
        if scalar.name not in allowed or scalar.regions:
            raise ValueError("noncanonical scalar contraction body")
        flags = scalar.properties.get("overflowFlags", scalar.attributes.get("overflowFlags"))
        if flags is not None and getattr(flags, "data", None) != frozenset():
            raise ValueError("poison-producing overflow flags are not modular integer arithmetic")
    visited = {operations[-1]}
    addition = operations[-1].operands[0].owner
    if (
        not isinstance(addition, Operation)
        or addition not in operations
        or addition.name != "arith.addi"
        or block.args[2] not in addition.operands
    ):
        raise ValueError("contraction must add the actual initialized accumulator")
    visited.add(addition)
    term = addition.operands[1 - list(addition.operands).index(block.args[2])]
    multiply = term.owner
    if not isinstance(multiply, Operation) or multiply not in operations or multiply.name != "arith.muli":
        raise ValueError("contraction accumulator term is not the direct source product")
    visited.add(multiply)

    def input_of(value):
        while value not in block.args:
            cast = value.owner
            if (
                not isinstance(cast, Operation)
                or cast not in operations
                or cast.name not in {"arith.extsi", "arith.extui", "arith.trunci"}
                or len(cast.operands) != 1
                or len(cast.results) != 1
            ):
                raise ValueError("contraction product has an unsupported operand expression")
            visited.add(cast)
            value = cast.operands[0]
        return value

    if {input_of(value) for value in multiply.operands} != set(block.args[:2]) or visited != set(operations):
        raise ValueError("contraction product/accumulator def-use is noncanonical or contains extra work")
    return {
        "maps": maps,
        "n_in": 2,
        "iteration_shape": (*batch, m, n, k),
        "parallel_dimensions": list(range(output_rank)),
        "reduction_dimensions": [output_rank],
    }, {
        "batch_shape": batch,
        "m": m,
        "k": k,
        "n": n,
        "lhs_batched": lhs_batched,
        "rhs_batched": rhs_batched,
        "output_rank": output_rank,
        "iteration_rank": iterator_rank,
    }


def extract_source_contraction(
    source_text: str,
    source_op_index: int,
    *,
    entry: str,
    max_m: int,
    max_n: int,
    max_k: int,
    max_macs: int = 100000,
    max_batch_extent: int = 2,
) -> tuple[str, dict]:
    """Clone one source operation with host-selected bounds, preserving named/generic spelling.

    Bounds are safety limits, not inferred target tile geometry. The caller must
    check whether the reduced compilation still exercises the changed mechanism.
    """
    if (
        not isinstance(entry, str)
        or not entry
        or type(source_op_index) is not int
        or any(type(value) is not int or not 1 <= value <= 4096 for value in (max_m, max_n, max_k))
        or type(max_batch_extent) is not int
        or not 1 <= max_batch_extent <= 16
        or type(max_macs) is not int
        or not 1 <= max_macs <= 100000
        or len(source_text.encode()) > 2_000_000
    ):
        raise ValueError("invalid bounded source contraction request")
    function = _entry(parse_mlir_text(source_text), entry)
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    if not 0 <= source_op_index < len(ops):
        raise ValueError("contraction source index is outside the explicit entry")
    operation = ops[source_op_index]
    info, geometry = _contract(operation)
    m, k, n = (min(geometry[name], bound) for name, bound in (("m", max_m), ("k", max_k), ("n", max_n)))
    reduced_batch = tuple(min(extent, max_batch_extent) for extent in geometry["batch_shape"])
    reduced_bounds = (*reduced_batch, m, n, k)
    source_bounds = (*geometry["batch_shape"], geometry["m"], geometry["n"], geometry["k"])
    macs = prod(reduced_batch) * m * k * n
    if macs > max_macs or reduced_bounds == source_bounds:
        raise ValueError("contraction witness must strictly reduce source work within the host MAC budget")
    reduced_operand_shapes = [tuple(reduced_bounds[expr.position] for expr in amap.results) for amap in info["maps"]]
    element_widths = [int(str(value.type.get_element_type())[1:]) for value in operation.operands]
    if (
        sum(prod(shape) * width // 8 for shape, width in zip(reduced_operand_shapes, element_widths, strict=True))
        > 65536
    ):
        raise ValueError("contraction witness exceeds the logical operand byte budget")
    text, record = _clone_bounded_reduction(source_text, function, operation, info, reduced_bounds, entry=entry)
    record.update(
        schema="actual_source_contraction_witness_v1",
        mechanism="integer_contraction",
        entry=entry,
        source_op_index=source_op_index,
        source_op_name=operation.name,
        source_geometry_mkn=[geometry["m"], geometry["k"], geometry["n"]],
        probe_geometry_mkn=[m, k, n],
        source_batch_shape=list(geometry["batch_shape"]),
        probe_batch_shape=list(reduced_batch),
        source_output_rank=geometry["output_rank"],
        source_iteration_rank=geometry["iteration_rank"],
        operand_batching={
            "lhs": "batched" if geometry["lhs_batched"] else "broadcast",
            "rhs": "batched" if geometry["rhs_batched"] else "broadcast",
        },
        max_macs=max_macs,
        max_batch_extent=max_batch_extent,
        probe_macs=macs,
        input_shapes=[row["shape"] for row in record["inputs"]],
        input_dtypes=[row["dtype"] for row in record["inputs"]],
        output_shape=record["output"]["shape"],
        output_dtype=record["output"]["dtype"],
        arithmetic="actual source casts, multiply and initialized add; modular wrap after each typed integer op",
        emitted_route_correspondence="UNPROVEN",
        numerical_qualification="UNPROVEN",
        runtime_admitted=False,
        full_model_executed=False,
        scope="one exact source contraction and initializer at reduced extents; not a complete layer or full-shape proof",
    )
    return text, record


def evaluate_source_contraction(source_text: str, extraction: dict, inputs: Sequence):
    """Evaluate only the hash-bound short source through the shared independent scalar oracle."""
    import numpy as np

    if (
        extraction.get("schema") != "actual_source_contraction_witness_v1"
        or _sha(source_text) != extraction.get("probe_source_sha256")
        or type(extraction.get("probe_macs")) is not int
        or not 0 < extraction["probe_macs"] <= 100000
    ):
        raise ValueError("stale or unbounded contraction reference contract")
    function = _entry(parse_mlir_text(source_text), extraction["entry"])
    if len(inputs) != len(function.body.block.args):
        raise ValueError("contraction reference argument count mismatch")
    expected = []
    for value, data in zip(function.body.block.args, inputs):
        dtype = str(value.type.get_element_type())
        array = np.asarray(data)
        if (
            list(array.shape) != list(value.type.get_shape())
            or dtype not in {"i8", "i16", "i32", "i64"}
            or array.dtype != np.dtype("int" + dtype[1:])
        ):
            raise ValueError("contraction reference needs exact typed logical input bytes/shapes")
        expected.append(array)
    if extraction["input_shapes"] != [list(array.shape) for array in expected] or extraction["input_dtypes"] != [
        str(value.type.get_element_type()) for value in function.body.block.args
    ]:
        raise ValueError("contraction reference input metadata changed")
    operation = next(
        op for op in function.body.block.ops if op.name in {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}
    )
    _, geometry = _contract(operation)
    actual_mkn = [geometry["m"], geometry["k"], geometry["n"]]
    actual_macs = prod(geometry["batch_shape"]) * geometry["m"] * geometry["k"] * geometry["n"]
    if (
        actual_mkn != extraction["probe_geometry_mkn"]
        or list(geometry["batch_shape"]) != extraction.get("probe_batch_shape", [])
        or geometry["iteration_rank"] != extraction.get("source_iteration_rank", 3)
        or actual_macs != extraction["probe_macs"]
    ):
        raise ValueError("contraction reference work metadata changed")
    if str(operation.results[0].type.get_element_type()) != extraction["output_dtype"]:
        raise ValueError("contraction reference output dtype metadata changed")
    outputs = evaluate_pointwise_source(source_text, expected)
    if len(outputs) != 1 or list(outputs[0].shape) != extraction["output_shape"]:
        raise ValueError("contraction reference output metadata changed")
    return outputs[0]


# ---- source_convolution_witness ----------------------------------------------------------------
# Bound a selected source convolution without replacing its scalar or padding semantics.
#
# This is extraction/reference evidence only. A caller must separately bind changed
# source ownership, compile both immutable revisions, and qualify the actual emitted
# mixed program. It grants no simulator admission and no full-shape equivalence.


def _props(op, key):
    return op.properties.get(key, op.attributes.get(key))


def _zero(value):
    owner = value.owner
    return (
        isinstance(owner, Operation)
        and owner.name == "arith.constant"
        and getattr(getattr(_props(owner, "value"), "value", None), "data", None) == 0
    )


def extract_source_convolution(
    source_text: str,
    source_op_index: int,
    *,
    entry: str,
    max_channels: int = 2,
    max_output_extent: int = 2,
    max_macs: int = 20000,
) -> tuple[str, dict]:
    """Clone one signed NCHW/OIHW i8→i32 generic convolution and exact zero producers.

    Kernel, spatial coefficients, padding and scalar operations never shrink or
    change. Batch/channel/output extents shrink, preserving the source's trailing
    stride remainder. Unsupported layouts, padding, initializers or bounds refuse.
    """
    if (
        not entry
        or type(source_op_index) is not int
        or any(type(n) is not int or not 1 <= n <= 8 for n in (max_channels, max_output_extent))
        or type(max_macs) is not int
        or not 1 <= max_macs <= 100000
    ):
        raise ValueError("invalid bounded convolution extraction request")
    module = parse_mlir_text(source_text)
    entries = [op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == entry]
    if len(entries) != 1 or len(entries[0].body.blocks) != 1:
        raise ValueError("one explicit single-block source entry required")
    ops = [op for op in entries[0].body.block.ops if op.name != "func.return"]
    if not 0 <= source_op_index < len(ops):
        raise ValueError("source operation index is outside entry")
    op = ops[source_op_index]
    op.verify()
    if (
        op.name != "linalg.generic"
        or len(op.operands) != 3
        or len(op.results) != 1
        or not _source_has_multiply_accumulate(op)
        or _iterators(op) != ("parallel",) * 4 + ("reduction",) * 3
    ):
        raise ValueError("selected source is not a supported convolution MAC recurrence")
    parallel, reduction, dtypes = _domain(op, _iterators(op))
    if dtypes != ("i8", "i8", "i32") or op.results[0].type != op.operands[-1].type:
        raise ValueError("source witness requires exact i8 operands and i32 output")
    body = list(op.regions[0].block.ops)
    if [item.name for item in body] != ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]:
        raise ValueError("source witness requires signed extensions and an unfused i32 MAC")
    maps = [item.data for item in op.get_indexing_maps()]
    terms = [[_terms(expr) for expr in mapping.results] for mapping in maps]
    if (
        terms[2] != [{0: 1}, {1: 1}, {2: 1}, {3: 1}]
        or terms[1] != [{1: 1}, {4: 1}, {5: 1}, {6: 1}]
        or terms[0][:2] != [{0: 1}, {4: 1}]
        or len(terms[0]) != 4
        or any(term is None for row in terms for term in row)
        or set(terms[0][2]) != {2, 5}
        or set(terms[0][3]) != {3, 6}
    ):
        raise ValueError("unsupported source convolution layout or affine spatial maps")
    stride = [terms[0][2][2], terms[0][3][3]]
    dilation = [terms[0][2][5], terms[0][3][6]]
    if min(*stride, *dilation) <= 0:
        raise ValueError("nonpositive source stride/dilation")
    activation, weight, initial = op.operands
    keep = {op}
    padding = [0, 0, 0, 0]
    pad = activation.owner
    if isinstance(pad, Operation) and pad.name == "tensor.insert_slice":
        if len(pad.operands) != 2:
            raise ValueError("dynamic padding slice unsupported")
        offsets, sizes, steps = [
            tuple(_props(pad, key).get_values()) for key in ("static_offsets", "static_sizes", "static_strides")
        ]
        raw, destination = pad.operands
        splat = destination.owner
        raw_shape, padded_shape = raw.type.get_shape(), destination.type.get_shape()
        if (
            len(raw_shape) != 4
            or sizes != raw_shape
            or offsets[:2] != (0, 0)
            or steps != (1, 1, 1, 1)
            or raw_shape[:2] != padded_shape[:2]
            or not isinstance(splat, Operation)
            or splat.name != "tensor.splat"
            or not _zero(splat.operands[0])
        ):
            raise ValueError("padding is not an exact same-rank zero insertion")
        padding = [
            offsets[2],
            offsets[3],
            padded_shape[2] - raw_shape[2] - offsets[2],
            padded_shape[3] - raw_shape[3] - offsets[3],
        ]
        if min(padding) < 0:
            raise ValueError("negative padding")
        keep.update((pad, splat, splat.operands[0].owner))
        activation = raw
    fill = initial.owner
    if (
        not isinstance(fill, Operation)
        or fill.name != "linalg.fill"
        or len(fill.operands) != 2
        or not _zero(fill.operands[0])
        or not isinstance(fill.operands[1].owner, Operation)
        or fill.operands[1].owner.name != "tensor.empty"
    ):
        raise ValueError("convolution initializer is not an explicit zero fill")
    keep.update((fill, fill.operands[0].owner, fill.operands[1].owner))
    if any(item not in ops for item in keep) or activation == weight:
        raise ValueError("nonlocal initialization or aliased boundary")
    n, co, ho, wo = parallel
    ci, kh, kw = reduction
    h, w = activation.type.get_shape()[2:]
    reduced_output = [min(n, 1), min(co, max_channels), min(ho, max_output_extent), min(wo, max_output_extent)]
    reduced_input = [reduced_output[0], min(ci, max_channels)]
    for size, out, new_out, kernel, step, dil, before, after in zip(
        (h, w), (ho, wo), reduced_output[2:], (kh, kw), stride, dilation, padding[:2], padding[2:]
    ):
        effective = (kernel - 1) * dil + 1
        if (size + before + after - effective) // step + 1 != out:
            raise ValueError("source extent does not satisfy convolution identity")
        remainder = (size + before + after - effective) % step
        reduced_input.append((new_out - 1) * step + effective - before - after + remainder)
    if any(n <= 0 for n in reduced_input):
        raise ValueError("preserved padding/kernel cannot fit the bounded output domain")
    reduced_weight = [reduced_output[1], reduced_input[1], kh, kw]
    macs = prod(reduced_output) * prod(reduced_weight[1:])
    if (
        macs > max_macs
        or macs >= prod(parallel + reduction)
        or prod(reduced_weight[1:]) * 16384 >= 2**31
        or sum(prod(shape) for shape in (reduced_input, reduced_weight, reduced_output)) > 65536
    ):
        raise ValueError("witness exceeds work bound, is not reduced, or may overflow independent i32 reference")
    shapes = {
        activation: reduced_input,
        weight: reduced_weight,
        initial: reduced_output,
        op.results[0]: reduced_output,
        fill.operands[1]: reduced_output,
    }
    if isinstance(pad, Operation) and pad in keep:
        padded = [
            *reduced_input[:2],
            reduced_input[2] + padding[0] + padding[2],
            reduced_input[3] + padding[1] + padding[3],
        ]
        shapes.update({pad.operands[1]: padded, pad.results[0]: padded})
    inputs = [activation, weight]
    types = [TensorType(value.type.get_element_type(), shapes[value]) for value in inputs]
    block = Block(arg_types=types)
    mapping = dict(zip(inputs, block.args))
    for item in ops:
        if item not in keep:
            continue
        if any(value not in mapping for value in item.operands):
            raise ValueError("source initialization has an unbound dependency")
        cloned = item.clone(value_mapper=mapping)
        block.add_op(cloned)
        for original, result in zip(item.results, cloned.results, strict=True):
            if original in shapes:
                result = Rewriter.replace_value_with_new_type(
                    result, TensorType(original.type.get_element_type(), shapes[original])
                )
            mapping[original] = result
        if item is pad:
            cloned.properties["static_sizes"] = DenseArrayBase.from_list(i64, reduced_input)
    block.add_op(ReturnOp(mapping[op.results[0]]))
    witness = ModuleOp([FuncOp(entry, FunctionType.from_lists(types, [mapping[op.results[0]].type]), Region(block))])
    witness.verify()
    cloned_mac = next(item for item in block.ops if item.name == "linalg.generic")
    _domain(cloned_mac, _iterators(cloned_mac))
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    return text, {
        "schema": "actual_source_convolution_witness_v1",
        "entry": entry,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_op_index": source_op_index,
        "source_indices": [i for i, item in enumerate(ops) if item in keep],
        "input_layout": "nchw",
        "weight_layout": "oihw",
        "output_layout": "nchw",
        "source_input_shape": list(activation.type.get_shape()),
        "input_shape": reduced_input,
        "weight_shape": reduced_weight,
        "output_shape": reduced_output,
        "stride": stride,
        "dilation": dilation,
        "padding": padding,
        "macs": macs,
        "scalar_semantics": "actual cloned signed extensions and i32 multiply/add recurrence; zero initialization",
        "independent_i32_no_overflow_bound": prod(reduced_weight[1:]) * 16384,
        "scope": "reduced source mechanism only; emitted-path correspondence and full-shape equivalence UNPROVEN",
    }


def evaluate_source_convolution(extraction: dict, activation, weight):
    """Existing independent integer golden, with explicit source-layout conversions."""
    import numpy as np

    from merlin.runtime.engine.command_buffer import Tensor, conv_im2col

    if (
        extraction.get("schema") != "actual_source_convolution_witness_v1"
        or extraction.get("input_layout") != "nchw"
        or extraction.get("weight_layout") != "oihw"
        or extraction.get("output_layout") != "nchw"
    ):
        raise ValueError("unsupported source convolution reference contract")
    x, w = np.asarray(activation), np.asarray(weight)
    if (
        list(x.shape) != extraction["input_shape"]
        or list(w.shape) != extraction["weight_shape"]
        or x.dtype != np.int8
        or w.dtype != np.int8
    ):
        raise ValueError("reference operands must be exact bounded signed i8 tensors")
    macs = prod(extraction["output_shape"]) * prod(w.shape[1:])
    if (
        macs != extraction["macs"]
        or not 0 < macs <= 100000
        or x.size + w.size + prod(extraction["output_shape"]) > 65536
    ):
        raise ValueError("reference exceeds bounded extraction work/storage")
    if prod(w.shape[1:]) * 16384 >= 2**31:
        raise ValueError("mathematical integer golden has no proved i32 no-overflow domain")
    nhwc = x.transpose(0, 2, 3, 1)
    co, ci, kh, kw = w.shape
    packed = w.transpose(2, 3, 1, 0).reshape(kh * kw * ci, co)
    columns = conv_im2col(
        Tensor(tuple(nhwc.shape), nhwc.reshape(-1).tolist(), "i8"),
        ci=ci,
        kh=kh,
        kw=kw,
        stride=extraction["stride"],
        padding=extraction["padding"],
        dilation=extraction["dilation"],
        layout="nhwc",
    )
    result = columns.matmul(Tensor(tuple(packed.shape), packed.reshape(-1).tolist(), "i8"))
    n, _, h, width = extraction["output_shape"]
    return np.asarray(result.data, dtype=np.int32).reshape(n, h, width, co).transpose(0, 3, 1, 2).copy()
