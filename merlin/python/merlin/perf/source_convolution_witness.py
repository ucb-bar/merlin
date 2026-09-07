"""Bound a selected source convolution without replacing its scalar or padding semantics.

This is extraction/reference evidence only. A caller must separately bind changed
source ownership, compile both immutable revisions, and qualify the actual emitted
mixed program. It grants no simulator admission and no full-shape equivalence.
"""
from __future__ import annotations

import hashlib
import io
from math import prod

from xdsl.dialects.builtin import DenseArrayBase, FunctionType, ModuleOp, TensorType, i64
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Operation, Region
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.perf.compiler_plan_evidence import _source_has_multiply_accumulate
from merlin.perf.model_macs import _domain, _iterators
from merlin.targetgen.conv_geometry import _terms


def _props(op, key):
    return op.properties.get(key, op.attributes.get(key))


def _zero(value):
    owner = value.owner
    return (isinstance(owner, Operation) and owner.name == "arith.constant"
            and getattr(getattr(_props(owner, "value"), "value", None), "data", None) == 0)


def extract_source_convolution(source_text: str, source_op_index: int, *, entry: str,
                               max_channels: int = 2, max_output_extent: int = 2,
                               max_macs: int = 20000) -> tuple[str, dict]:
    """Clone one signed NCHW/OIHW i8→i32 generic convolution and exact zero producers.

    Kernel, spatial coefficients, padding and scalar operations never shrink or
    change. Batch/channel/output extents shrink, preserving the source's trailing
    stride remainder. Unsupported layouts, padding, initializers or bounds refuse.
    """
    if (not entry or type(source_op_index) is not int
            or any(type(n) is not int or not 1 <= n <= 8 for n in (max_channels, max_output_extent))
            or type(max_macs) is not int or not 1 <= max_macs <= 100000):
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
    if (op.name != "linalg.generic" or len(op.operands) != 3 or len(op.results) != 1
            or not _source_has_multiply_accumulate(op)
            or _iterators(op) != ("parallel",)*4 + ("reduction",)*3):
        raise ValueError("selected source is not a supported convolution MAC recurrence")
    parallel, reduction, dtypes = _domain(op, _iterators(op))
    if dtypes != ("i8", "i8", "i32") or op.results[0].type != op.operands[-1].type:
        raise ValueError("source witness requires exact i8 operands and i32 output")
    body = list(op.regions[0].block.ops)
    if [item.name for item in body] != ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]:
        raise ValueError("source witness requires signed extensions and an unfused i32 MAC")
    maps = [item.data for item in op.get_indexing_maps()]
    terms = [[_terms(expr) for expr in mapping.results] for mapping in maps]
    if (terms[2] != [{0: 1}, {1: 1}, {2: 1}, {3: 1}]
            or terms[1] != [{1: 1}, {4: 1}, {5: 1}, {6: 1}]
            or terms[0][:2] != [{0: 1}, {4: 1}]
            or len(terms[0]) != 4 or any(term is None for row in terms for term in row)
            or set(terms[0][2]) != {2, 5} or set(terms[0][3]) != {3, 6}):
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
        offsets, sizes, steps = [tuple(_props(pad, key).get_values()) for key in
                                ("static_offsets", "static_sizes", "static_strides")]
        raw, destination = pad.operands
        splat = destination.owner
        raw_shape, padded_shape = raw.type.get_shape(), destination.type.get_shape()
        if (len(raw_shape) != 4 or sizes != raw_shape or offsets[:2] != (0, 0)
                or steps != (1, 1, 1, 1) or raw_shape[:2] != padded_shape[:2]
                or not isinstance(splat, Operation) or splat.name != "tensor.splat"
                or not _zero(splat.operands[0])):
            raise ValueError("padding is not an exact same-rank zero insertion")
        padding = [offsets[2], offsets[3], padded_shape[2]-raw_shape[2]-offsets[2],
                   padded_shape[3]-raw_shape[3]-offsets[3]]
        if min(padding) < 0:
            raise ValueError("negative padding")
        keep.update((pad, splat, splat.operands[0].owner))
        activation = raw
    fill = initial.owner
    if (not isinstance(fill, Operation) or fill.name != "linalg.fill" or len(fill.operands) != 2
            or not _zero(fill.operands[0]) or not isinstance(fill.operands[1].owner, Operation)
            or fill.operands[1].owner.name != "tensor.empty"):
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
            (h, w), (ho, wo), reduced_output[2:], (kh, kw), stride, dilation, padding[:2], padding[2:]):
        effective = (kernel-1)*dil+1
        if (size+before+after-effective)//step+1 != out:
            raise ValueError("source extent does not satisfy convolution identity")
        remainder = (size+before+after-effective) % step
        reduced_input.append((new_out-1)*step+effective-before-after+remainder)
    if any(n <= 0 for n in reduced_input):
        raise ValueError("preserved padding/kernel cannot fit the bounded output domain")
    reduced_weight = [reduced_output[1], reduced_input[1], kh, kw]
    macs = prod(reduced_output)*prod(reduced_weight[1:])
    if (macs > max_macs or macs >= prod(parallel+reduction) or prod(reduced_weight[1:])*16384 >= 2**31
            or sum(prod(shape) for shape in (reduced_input, reduced_weight, reduced_output)) > 65536):
        raise ValueError("witness exceeds work bound, is not reduced, or may overflow independent i32 reference")
    shapes = {activation: reduced_input, weight: reduced_weight, initial: reduced_output,
              op.results[0]: reduced_output, fill.operands[1]: reduced_output}
    if isinstance(pad, Operation) and pad in keep:
        padded = [*reduced_input[:2], reduced_input[2]+padding[0]+padding[2],
                  reduced_input[3]+padding[1]+padding[3]]
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
                result = Rewriter.replace_value_with_new_type(result, TensorType(original.type.get_element_type(), shapes[original]))
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
    return text, {"schema": "actual_source_convolution_witness_v1", "entry": entry,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_op_index": source_op_index, "source_indices": [i for i, item in enumerate(ops) if item in keep],
        "input_layout": "nchw", "weight_layout": "oihw", "output_layout": "nchw",
        "source_input_shape": list(activation.type.get_shape()), "input_shape": reduced_input,
        "weight_shape": reduced_weight, "output_shape": reduced_output,
        "stride": stride, "dilation": dilation, "padding": padding, "macs": macs,
        "scalar_semantics": "actual cloned signed extensions and i32 multiply/add recurrence; zero initialization",
        "independent_i32_no_overflow_bound": prod(reduced_weight[1:])*16384,
        "scope": "reduced source mechanism only; emitted-path correspondence and full-shape equivalence UNPROVEN"}


def evaluate_source_convolution(extraction: dict, activation, weight):
    """Existing independent integer golden, with explicit source-layout conversions."""
    import numpy as np
    from merlin.runtime.tensor import Tensor
    from merlin.targetgen.capsule_golden import im2col
    if (extraction.get("schema") != "actual_source_convolution_witness_v1"
            or extraction.get("input_layout") != "nchw" or extraction.get("weight_layout") != "oihw"
            or extraction.get("output_layout") != "nchw"):
        raise ValueError("unsupported source convolution reference contract")
    x, w = np.asarray(activation), np.asarray(weight)
    if (list(x.shape) != extraction["input_shape"] or list(w.shape) != extraction["weight_shape"]
            or x.dtype != np.int8 or w.dtype != np.int8):
        raise ValueError("reference operands must be exact bounded signed i8 tensors")
    macs = prod(extraction["output_shape"])*prod(w.shape[1:])
    if (macs != extraction["macs"] or not 0 < macs <= 100000
            or x.size+w.size+prod(extraction["output_shape"]) > 65536):
        raise ValueError("reference exceeds bounded extraction work/storage")
    if prod(w.shape[1:])*16384 >= 2**31:
        raise ValueError("mathematical integer golden has no proved i32 no-overflow domain")
    nhwc = x.transpose(0, 2, 3, 1)
    co, ci, kh, kw = w.shape
    packed = w.transpose(2, 3, 1, 0).reshape(kh*kw*ci, co)
    columns = im2col(Tensor(tuple(nhwc.shape), nhwc.reshape(-1).tolist(), "i8"), ci, kh, kw,
        stride=extraction["stride"], padding=extraction["padding"], dilation=extraction["dilation"], layout="nhwc")
    result = columns.matmul(Tensor(tuple(packed.shape), packed.reshape(-1).tolist(), "i8"))
    n, _, h, width = extraction["output_shape"]
    return np.asarray(result.data, dtype=np.int32).reshape(n, h, width, co).transpose(0, 3, 1, 2).copy()
