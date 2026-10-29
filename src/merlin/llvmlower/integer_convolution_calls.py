"""Recover exact integer convolution islands before host layout materialization.

Recognition uses typed maps, modular arithmetic and complete closed use chains.
The backend supplies direct-gather instructions; no device is discovered here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.ir import Operation, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.rewriter import InsertPoint, Rewriter

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_ir import recognize_signed_integer_matmul

from .borrowed_buffer_calls import BorrowedBufferEffects, from_writers
from .fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from .integer_epilogue import _constant, _plain, _setup, match_integer_epilogue
from .integer_matmul_calls import ComposedIntegerMatmulProgram, PreparedIntegerMatmulCalls


@dataclass(frozen=True)
class IntegerConvolutionCall:
    operation_ordinal: int
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    symbol: str
    borrowed_symbol: str
    kernel_symbol: str
    output_dtype: str = "i32"
    readout_identity: str | None = None

    def __post_init__(self):
        shapes = (self.input_shape, self.weight_shape, self.output_shape)
        if (type(self.operation_ordinal) is not int or self.operation_ordinal < 0
                or any(type(s) is not tuple or len(s) != 4 for s in shapes)
                or any(type(d) is not int or d <= 0 for s in shapes for d in s)
                or any(type(s) is not tuple or len(s) != 2 or any(type(d) is not int or d <= 0 for d in s)
                       for s in (self.stride, self.dilation))
                or any(prod(s) * 4 >= 1 << 63 for s in shapes)
                or self.output_dtype not in ("i8", "i32")
                or (self.output_dtype == "i8") != (self.readout_identity is not None)
                or self.readout_identity is not None and re.fullmatch("[0-9a-f]{64}", self.readout_identity) is None
                or len({self.symbol, self.borrowed_symbol, self.kernel_symbol}) != 3
                or any(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None
                       for name in (self.symbol, self.borrowed_symbol, self.kernel_symbol))):
            raise ValueError("bounded exact integer convolution geometry and symbols required")
        b, h, w, ci = self.input_shape
        kh, kw, wc, co = self.weight_shape
        sh, sw = self.stride
        dh, dw = self.dilation
        expected = (b, (h - dh * (kh - 1) - 1) // sh + 1, (w - dw * (kw - 1) - 1) // sw + 1, co)
        if ci != wc or self.output_shape != expected:
            raise ValueError("convolution output does not match its exact unpadded window")


def _shape(value):
    if (not isinstance(value.type, builtin.TensorType) or type(value.type.encoding) is not builtin.NoneAttr
            or any(d <= 0 for d in value.type.get_shape())):
        raise ValueError("static unencoded tensor required")
    return tuple(value.type.get_shape())


def _pure(operation, properties=()):
    return isinstance(operation, Operation) and _plain(operation) and set(operation.properties) <= set(properties)


def _single_user(value):
    uses = tuple(value.uses)
    return uses[0].operation if len(uses) == 1 else None


def _transpose(operation, permutation):
    if (type(operation) is not TransposeOp or not _pure(operation, ("permutation",))
            or tuple(operation.permutation.get_values()) != tuple(permutation)
            or len(operation.inputs) != 1 or len(operation.outputs) != 1
            or type(operation.outputs[0].owner) is not tensor.EmptyOp
            or not _pure(operation.outputs[0].owner) or operation.outputs[0].owner.operands
            or len(operation.body.blocks) != 1):
        return False
    block = operation.body.block
    body = tuple(block.ops)
    return (len(block.args) == 2 and len(body) == 1 and body[0].name == "linalg.yield"
            and _pure(body[0]) and tuple(body[0].operands) == (block.args[0],)
            and _shape(operation.results[0]) == tuple(_shape(operation.inputs[0])[i] for i in permutation)
            and operation.results[0].type.element_type == operation.inputs[0].type.element_type)


def _reshape_pair(expand, source_shape, target_shape):
    if (type(expand) is not tensor.ExpandShapeOp
            or not _pure(expand, ("reassociation", "static_output_shape")) or len(expand.operands) != 1
            or _shape(expand.results[0]) != target_shape):
        return None
    collapse = expand.operands[0].owner
    if (type(collapse) is not tensor.CollapseShapeOp or not _pure(collapse, ("reassociation",))
            or len(collapse.operands) != 1 or _shape(collapse.operands[0]) != source_shape
            or _shape(collapse.results[0]) != (prod(source_shape),)
            or _single_user(collapse.results[0]) is not expand):
        return None
    if any(v.type.element_type != collapse.operands[0].type.element_type
           for v in (collapse.results[0], expand.results[0])):
        return None

    def groups(op):
        return tuple(tuple(v.value.data for v in group) for group in op.reassociation)
    if groups(collapse) != (tuple(range(len(source_shape))),) or groups(expand) != (tuple(range(len(target_shape))),):
        return None
    return collapse


def _bias_add(operation, source, channels):
    if (type(operation) is not GenericOp
            or not _pure(operation, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
            or len(operation.inputs) != 2 or operation.inputs[0] is not source or len(operation.outputs) != 1
            or _shape(operation.inputs[1]) != (channels,) or str(operation.inputs[1].type.element_type) != "i32"
            or _shape(operation.results[0]) != _shape(source) or str(operation.results[0].type.element_type) != "i32"
            or tuple(str(t.data) for t in operation.iterator_types) != ("parallel",) * 4
            or type(operation.outputs[0].owner) is not tensor.EmptyOp
            or not _pure(operation.outputs[0].owner) or operation.outputs[0].owner.operands
            or len(operation.body.blocks) != 1):
        return None
    d = AffineDimExpr
    identity = AffineMap(4, 0, tuple(d(i) for i in range(4)))
    if tuple(a.data for a in operation.indexing_maps) != (identity, AffineMap(4, 0, (d(1),)), identity):
        return None
    block = operation.body.block
    body = tuple(block.ops)
    if (len(block.args) != 3 or len(body) != 2 or type(body[0]) is not arith.AddiOp
            or not _pure(body[0], ("overflowFlags",)) or body[0].overflow_flags.data
            or set(body[0].operands) != set(block.args[:2]) or body[1].name != "linalg.yield"
            or not _pure(body[1]) or tuple(body[1].operands) != (body[0].results[0],)):
        return None
    return operation.inputs[1]


def _match(operation, readout):
    from .im2col_pack import PackReport, _bounded_plain, _match_im2col

    clone = operation.clone()
    for nested in clone.walk():
        for key in tuple(nested.attributes):
            if key.startswith("prov."):
                del nested.attributes[key]
    try:
        recognize_signed_integer_matmul(clone, ("i8", "i8", "i32"))
    finally:
        clone.drop_all_references()
    seed = _constant(operation.outputs[0])
    if seed is None or seed.value.data != 0:
        return None
    match = _match_im2col(operation, PackReport())
    if match is None or not _bounded_plain(match.gather, ("indexing_maps", "iterator_types", "operandSegmentSizes")):
        return None
    if (tuple(str(t.data) for t in match.gather.iterator_types) != ("parallel",) * 6
            or not _bounded_plain(match.gather.body.block.last_op)
            or type(match.gather.outputs[0].owner) is not tensor.EmptyOp
            or not _pure(match.gather.outputs[0].owner) or match.gather.outputs[0].owner.operands
            or _reshape_pair(match.expand, _shape(match.gather.results[0]), (match.k, match.m)) is not match.collapse):
        return None
    image = match.gather.inputs[0].owner
    if not _transpose(image, (0, 3, 1, 2)):
        return None
    input_value = image.inputs[0]
    b, h, w, ci = _shape(input_value)
    co, kh, kw = match.f, match.kh, match.kw
    weight_expand = operation.inputs[0].owner
    weight_collapse = _reshape_pair(weight_expand, (co, ci, kh, kw), (co, ci * kh * kw))
    if weight_collapse is None:
        return None
    weight = weight_collapse.operands[0].owner
    if not _transpose(weight, (3, 2, 0, 1)) or _shape(weight.inputs[0]) != (kh, kw, ci, co):
        return None
    result_collapse = _single_user(operation.results[0])
    result_expand = (_single_user(result_collapse.results[0])
                     if type(result_collapse) is tensor.CollapseShapeOp else None)
    if _reshape_pair(result_expand, (co, b * match.oh * match.ow), (co, b, match.oh, match.ow)) is not result_collapse:
        return None
    result_transpose = _single_user(result_expand.results[0])
    if not _transpose(result_transpose, (1, 0, 2, 3)):
        return None
    bias = _single_user(result_transpose.results[0])
    bias_value = _bias_add(bias, result_transpose.results[0], co)
    if bias_value is None:
        return None
    nhwc = _single_user(bias.results[0])
    if not _transpose(nhwc, (0, 2, 3, 1)):
        return None
    chain = [operation, result_collapse, result_expand, result_transpose, bias, nhwc]
    epilogue = match_integer_epilogue(nhwc, readout) if readout is not None else None
    if epilogue is not None:
        chain.extend(epilogue.operations)
    # Moving the external call cannot cross unknown executable source effects.
    cursor = operation
    allowed = set(chain)
    while cursor is not chain[-1]:
        cursor = cursor.next_op
        if cursor is None:
            return None
        if cursor not in allowed and not _setup(cursor):
            return None
    call_geometry = (input_value, weight.inputs[0], bias_value, chain, epilogue,
                     (b, match.oh, match.ow, co), (match.sh, match.sw), (match.dh, match.dw))
    return call_geometry


def prepare_integer_convolution_calls(
    source, *, select, borrowed_effects, output_readout=None, entry="forward",
    symbol_prefix="__merlin_integer_conv", allocation_alignment=64,
):
    """Outline closed, zero-padded-free NHWC/HWIO convolution plus modular bias.

    Unsupported maps, initializers, extra raw observers, effects and geometry
    keep the original source. Reduction order may change because the admitted
    signed widening and i32 multiply/add use exact modular arithmetic.
    """
    if type(borrowed_effects) is not BorrowedBufferEffects or not callable(select):
        raise ValueError("explicit synchronous implementation selection required")
    borrowed_effects.__post_init__()
    if (not isinstance(symbol_prefix, str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", symbol_prefix) is None
            or type(allocation_alignment) is not int or allocation_alignment <= 0
            or allocation_alignment & (allocation_alignment - 1)):
        raise ValueError("portable symbol prefix and power-of-two allocation alignment required")
    module = parse_mlir_text(source)
    module.verify()
    functions = [op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("one single-block selected tensor entry required")
    reserved = {getattr(op.properties.get("sym_name"), "data", None) for op in module.body.block.ops}
    calls, writers, proofs = [], [], []
    for ordinal, operation in enumerate(tuple(functions[0].walk())):
        if type(operation) is not GenericOp or operation.parent_block() is not functions[0].body.block:
            continue
        try:
            matched = _match(operation, output_readout)
        except (ValueError, AttributeError, TypeError):
            continue
        if matched is None:
            continue
        image, weight, bias, chain, epilogue, output_shape, stride, dilation = matched
        index = len(calls)
        while any(name in reserved for name in (f"{symbol_prefix}_{index}", f"{symbol_prefix}_{index}_borrowed",
                                                f"{symbol_prefix}_{index}_kernel")):
            index += 1
        symbol = f"{symbol_prefix}_{index}"
        try:
            call = IntegerConvolutionCall(ordinal, _shape(image), _shape(weight), output_shape, stride, dilation,
                                          symbol, symbol + "_borrowed", symbol + "_kernel")
        except ValueError:
            continue
        selected = select(call)
        if type(selected) is not bool:
            raise ValueError("implementation selection must return a Boolean")
        if not selected:
            continue
        if epilogue is not None:
            narrow = replace(call, output_dtype="i8", readout_identity=output_readout.identity)
            admitted = select(narrow)
            if type(admitted) is not bool:
                raise ValueError("implementation selection must return a Boolean")
            if admitted:
                call = narrow
            else:
                chain = chain[:-len(epilogue.operations)]
                epilogue = None
        typ = chain[-1].results[0].type
        empty = tensor.EmptyOp((), typ)
        emitted = func.CallOp(call.symbol, [image, weight, bias, empty.results[0]], [typ])
        emitted.attributes.update(operation.attributes)
        Rewriter.insert_op([empty, emitted], InsertPoint.before(operation))
        chain[-1].results[0].replace_all_uses_with(emitted.results[0])
        for stage in reversed(chain):
            Rewriter.erase_op(stage)
        types = [image.type, weight.type, bias.type, typ]
        declaration = func.FuncOp(call.symbol, (types, [typ]), Region(), visibility="private",
                                   arg_attrs=builtin.ArrayAttr([builtin.DictionaryAttr({
                                       "bufferization.access": builtin.StringAttr(access)
                                   }) for access in ("read", "read", "read", "write")]))
        declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
        module.body.block.add_op(declaration)
        writers.append(FreshTensorWriterContract(call.symbol, 3, (3,), call.borrowed_symbol,
                                                 allocation_alignment, borrowed_effects=borrowed_effects))
        calls.append(call)
        reserved.update((call.symbol, call.borrowed_symbol, call.kernel_symbol))
        if epilogue is not None:
            proofs.append(epilogue.correspondence)
    if not calls:
        return PreparedIntegerMatmulCalls(source, hashlib.sha256(source.encode()).hexdigest(), (), ())
    rewrite_fresh_tensor_writers(module, writers)
    module.verify()
    return PreparedIntegerMatmulCalls(ir_text(module), hashlib.sha256(source.encode()).hexdigest(),
                                      tuple(calls), tuple(writers), tuple(proofs))


def lower_integer_convolution_program(
    source, workdir, *, select, emit_kernel, borrowed_effects, data_layout, output_readout=None,
    host_features=None,
):
    """Compose direct convolution with ordinary host/lifetime checks.

    Closed writers default to tensor-helper inlining before bufferization;
    an explicit ``host_features`` set, including an empty set, overrides it.
    """
    from .device_shim import LogicalBuffer, emit_logical_translation_unit
    from .lower import lower_model
    from .tensor_helper_inlining import composition_features

    prepared = prepare_integer_convolution_calls(source, select=select, borrowed_effects=borrowed_effects,
                                                 output_readout=output_readout)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "prepared.mlir"
    path.write_text(prepared.source)
    kernels, bridges, pins = [], [], []
    for index, call in enumerate(prepared.calls):
        kernel, implementation = emit_kernel(call)
        for location, expected in implementation:
            if hashlib.sha256(Path(location).read_bytes()).hexdigest() != expected:
                raise ValueError("convolution implementation changed during emission")
            pins.append((Path(location), expected))
        kernel_path, bridge_path = workdir / f"kernel{index}.ll", workdir / f"bridge{index}.c"
        kernel_path.write_text(kernel)
        shapes = (call.input_shape, call.weight_shape, (call.weight_shape[3],), call.output_shape)
        buffers = tuple(LogicalBuffer(name, "output" if i == 3 else "input", name, shape, dtype)
                        for i, (name, shape, dtype) in enumerate(zip(("input", "weight", "bias", "output"), shapes,
                                                                  ("i8", "i8", "i32", call.output_dtype), strict=True)))
        abi_symbol = call.symbol + "_logical_abi"
        unit = emit_logical_translation_unit("external", {abi_symbol: (buffers, [dict(role=b.role, shape=b.shape,
                                                                                   dtype=b.dtype) for b in buffers])},
                                             kernel_symbol_for=lambda _: call.kernel_symbol)
        if unit.skipped or unit.symbols != (abi_symbol,):
            raise ValueError("convolution logical pointer ABI could not be bound exactly")
        ranks = [len(shape) for shape in shapes]
        params = ",".join(f"const merlin_lmemref_{rank}d *d{i}" for i, rank in enumerate(ranks))
        text = unit.text + f"\nvoid _mlir_ciface_{call.borrowed_symbol}({params}) {{\n"
        text += "  merlin_lmemref_4d result;\n"
        text += f"  _mlir_ciface_{abi_symbol}(&result,d0,d1,d2,d3);\n"
        text += "  if(result.aligned!=d3->aligned || result.offset!=d3->offset) __builtin_trap();\n}\n"
        bridge_path.write_text(text)
        kernels.append(kernel_path)
        bridges.append(bridge_path)
    borrowed = (from_writers(path, prepared.writers, tuple(dict.fromkeys(pins)), lambda: None)
                if prepared.calls else None)
    host = lower_model(prepared.source, workdir / "host", targets=(), index_bits=64, data_layout=data_layout,
                       borrowed_buffer_calls=borrowed, hoist_static_allocs=False,
                       features=composition_features(host_features, has_writers=bool(prepared.calls)))
    return ComposedIntegerMatmulProgram(prepared, host, tuple(kernels), tuple(bridges))
