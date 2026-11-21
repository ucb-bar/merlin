"""Original-SSA host/device preparation through explicit ordinary emitters.

The initial source family is the existing checked sibling integer outliner.
Host epilogues and outputs remain original SSA. This adapter assembles source,
pointer transport and actual callback products; it is not an execution grant or
a replacement for final source-work, borrowed-call, object and numerical gates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from xdsl.dialects import builtin, func, llvm, tensor
from xdsl.dialects.builtin import IntegerAttr, i1, i8, i64
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.rewriter import Rewriter

from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.lower.device.logical_borrowed_abi import LogicalBorrowedABI, _symbol
from merlin.compiler.lower.device.logical_borrowed_batch_abi import emit_borrowed_batch_translation_unit
from merlin.compiler.lower.host.borrowed_buffer_calls import from_writers
from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
from merlin.compiler.lower.host.builder import PTR, FnBuilder
from merlin.compiler.lower.host.writers import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from merlin.compiler.schedule.blocks import SpatialMaxReadout
from merlin.compiler.schedule.convolution.windows import Window2D
from merlin.compiler.transforms.certificates import ir_text
from merlin.compiler.transforms.dense.head_batching import (
    plan_independent_head_batching,
    read_batched_integer_product,
)

# ---- source_program ----------------------------------------------------------------------------


def _sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


@dataclass(frozen=True)
class SourceProgram:
    source: str
    host: str
    core: str
    entry_transport: str
    core_transport: str
    abi: tuple[tuple[str, tuple[int, ...], str, str], ...]
    core_abi: tuple[tuple[str, tuple[int, ...], str, str], ...]
    source_members: tuple[tuple[int, str | None], ...]
    prepared: object
    target: str
    entry_symbol: str
    device_symbol: str

    def revalidate(self):
        current = prepare_source_program(
            self.source,
            effects=self.prepared.writer.borrowed_effects,
            target=self.target,
            entry_symbol=self.entry_symbol,
            device_symbol=self.device_symbol,
        )
        fields = ("source", "host", "core", "entry_transport", "core_transport", "abi", "core_abi", "source_members")
        if any(getattr(self, field) != getattr(current, field) for field in fields):
            raise ValueError("prepared source program changed; replan from the original source")
        return current


def prepare_source_program(source, *, effects, target, entry_symbol="kernel_entry", device_symbol="source_device"):
    """Bind the existing sibling route and all original logical inputs/results.

    ``effects`` must come from the selected implementation owner. Source reading
    cannot establish noescape or synchronous completion for an unknown helper.
    Other source families refuse through the existing exact outliner.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import UnitAttr

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from merlin.compiler.lower.device.device_shim import LogicalBuffer, emit_logical_translation_unit
    from merlin.compiler.lower.source.region import (
        emit_ranked_source_entry,
        prepare_sibling_source_host,
        tensor_abi,
    )
    from merlin.compiler.schedule.dense.source import read_zero_seeded_integer_matmul
    from merlin.compiler.transforms.host_arithmetic import _identifier

    if type(source) is not str or not source or len(source.encode()) > 16 << 20:
        raise ValueError("bounded original source text required")
    for symbol in (entry_symbol, device_symbol):
        _identifier(symbol)
    reserved = {"forward", "source_shared_core", "source_shared_core_borrowed", "source_core_transport"}
    if entry_symbol == device_symbol or {entry_symbol, device_symbol} & reserved:
        raise ValueError("source-program entry/device symbols must be distinct from original transport owners")
    if type(target) is not str or not target.isidentifier():
        raise ValueError("explicit target binding identifier required")
    module = parse_mlir_text(source)
    before = text(module, generic=True)
    original_owners = [
        op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
    ]
    if len(original_owners) != 1 or len(original_owners[0].body.blocks) != 1:
        raise ValueError("source program requires one original closed forward block")
    original_owner = original_owners[0]
    original_roots = []
    for ordinal, op in enumerate(original_owner.body.block.ops):
        inputs = tuple(getattr(op, "inputs", ()))
        if (
            op.name in ("linalg.generic", "linalg.matmul")
            and len(inputs) == 2
            and len(op.results) == 1
            and tuple(str(value.type.element_type) for value in (*inputs, op.results[0])) == ("i8", "i8", "i32")
        ):
            original_roots.append((ordinal, op, read_zero_seeded_integer_matmul(op)))
    prepared = prepare_sibling_source_host(module, effects=effects)
    owner = next(
        op for op in prepared.module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
    )
    owner.attributes["llvm.emit_c_interface"] = UnitAttr()
    original_inputs, original_outputs = tensor_abi(module)
    abi = tuple(
        (row["name"], tuple(row["shape"]), row["dtype"], "read" if index < len(original_inputs) else "write")
        for index, row in enumerate((*original_inputs, *original_outputs))
    )
    callee = next(op for op in prepared.core.body.block.ops if isinstance(op, func.FuncOp))
    types = (*callee.function_type.inputs, *callee.function_type.outputs)
    count = len(callee.function_type.inputs)
    if any(len(typ.get_shape()) != 2 for typ in types):
        raise ValueError("initial source-program transport supports the exact rank-two sibling family")
    buffers = tuple(
        LogicalBuffer(
            "core_" + str(i),
            "input" if i < count else "output",
            "input_" + str(i) if i < count else "out_" + str(i - count),
            tuple(typ.get_shape()),
            str(typ.element_type),
        )
        for i, typ in enumerate(types)
    )
    core_abi = tuple((b.name, b.shape, b.dtype, "read" if i < count else "write") for i, b in enumerate(buffers))
    transport = emit_logical_translation_unit(
        target,
        {
            "source_core_transport": (
                buffers,
                [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers],
            )
        },
        kernel_symbol_for=lambda _: device_symbol,
    )
    if transport.skipped or transport.symbols != ("source_core_transport",):
        raise ValueError("ordinary source-core pointer transport refused the complete ABI")
    n = len(buffers)
    bridge = (
        transport.text
        + "\nvoid _mlir_ciface_"
        + prepared.writer.borrowed_symbol
        + "("
        + ",".join("const merlin_lmemref_2d*d" + str(i) for i in range(n))
        + ") {source_core_transport_results result;_mlir_ciface_source_core_transport(&result,"
        + ",".join("d" + str(i) for i in range(n))
        + ");}\n"
    )
    # Read members from the actual outlined core; the existing outliner retains
    # source region attributes. These are source links, not instruction credit.
    roots = [op for op in prepared.core.walk() if op.name in ("linalg.generic", "linalg.matmul")]
    if len(roots) != len(original_roots) or any(
        read_zero_seeded_integer_matmul(root) != geometry
        or root.attributes.get("prov.region_id") != original.attributes.get("prov.region_id")
        for root, (_, original, geometry) in zip(roots, original_roots, strict=True)
    ):
        raise ValueError("outlined core does not retain every ordered original integer root")
    # Ordinals refer to ORIGINAL source, not positions after view CSE/hoisting.
    members = tuple(
        (ordinal, getattr(op.attributes.get("prov.region_id"), "data", None)) for ordinal, op, _ in original_roots
    )
    if text(module, generic=True) != before:
        raise ValueError("source-program preparation changed original SSA")
    return SourceProgram(
        source,
        text(prepared.module, generic=True),
        text(prepared.core, generic=True),
        emit_ranked_source_entry(module, entry_symbol=entry_symbol),
        bridge,
        abi,
        core_abi,
        members,
        prepared,
        target,
        entry_symbol,
        device_symbol,
    )


@dataclass(frozen=True)
class SourceEmission:
    source_sha256: str
    data: bytes
    format: str


@dataclass(frozen=True)
class SourceProgramEmission:
    device: SourceEmission
    host: SourceEmission
    entry_transport: str
    core_transport: str
    receipt: dict


def emit_source_program(program, *, emit_device, emit_host):
    """Consume actual source components, preserving their separately owned ABI.

    emit_device(core_module, *, symbol, abi, source_members) lowers the returned core.
    emit_host(host_module, *, writers) uses ordinary stock host lowering with the
    original borrowed-call contracts. Both return actual SourceEmission bytes.
    The owner compiles/links the two generated transport translation units and
    runs its unchanged final object, source-attribution and numerical checks.
    """
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if type(program) is not SourceProgram or not callable(emit_device) or not callable(emit_host):
        raise ValueError("prepared original source and actual emitter callbacks required")
    current = program.revalidate()
    outputs = []
    for source, callback, options in (
        (
            current.core,
            emit_device,
            {"symbol": current.device_symbol, "abi": current.core_abi, "source_members": current.source_members},
        ),
        (current.host, emit_host, {"writers": (current.prepared.writer,)}),
    ):
        product = callback(parse_mlir_text(source), **options)
        if (
            type(product) is not SourceEmission
            or product.source_sha256 != _sha(source)
            or type(product.data) is not bytes
            or not product.data
            or len(product.data) > 64 << 20
            or product.format not in ("llvm_ir", "llvm_dialect", "object")
        ):
            raise ValueError("actual emission does not match the prepared source component")
        outputs.append(product)
        program.revalidate()
    device, host = outputs
    if device.format != "llvm_dialect":
        raise ValueError("device source bridge requires inspectable LLVM-dialect ABI")
    from xdsl.dialects import llvm
    from xdsl.parser import Parser

    from merlin.compiler.ir.llvm_context import make_llvm_context

    emitted = Parser(make_llvm_context(), device.data.decode()).parse_module()
    emitted.verify()
    symbols = [
        op for op in emitted.body.block.ops if isinstance(op, llvm.FuncOp) and op.sym_name.data == current.device_symbol
    ]
    if (
        len(symbols) != 1
        or not symbols[0].body.blocks
        or symbols[0].linkage != llvm.LinkageAttr("external")
        or tuple(symbols[0].function_type.inputs) != (llvm.LLVMPointerType(),) * len(current.core_abi)
        or symbols[0].function_type.output != llvm.LLVMVoidType()
    ):
        raise ValueError("actual device function does not implement the complete logical pointer ABI")
    return SourceProgramEmission(
        device,
        host,
        current.entry_transport,
        current.core_transport,
        {
            "schema": "merlin.source_program_emission.v1",
            "source_sha256": _sha(current.source),
            "host_source_sha256": _sha(current.host),
            "core_source_sha256": _sha(current.core),
            "device_product_sha256": _sha(device.data),
            "host_product_sha256": _sha(host.data),
            "entry_transport_sha256": _sha(current.entry_transport),
            "core_transport_sha256": _sha(current.core_transport),
            "logical_abi": current.abi,
            "device_abi": current.core_abi,
            "outlined_source_members": current.source_members,
            "original_epilogues_and_outputs_retained": True,
            "scope": "actual component emission and ABI transport; final owner admission remains required",
            "physical_effects_qualified": False,
            "target_execution": False,
            "unavoidable_source_work_attribution": "requires independent final owner audit",
        },
    )


# ---- integer_batched_matmul_calls --------------------------------------------------------------
# Ordinary source-to-host composition for independent integer batch products.
#
# An explicit backend supplies the same synchronous matrix primitive used for a
# single product. The shared bridge owns ranked views and counted head loops;
# the ordinary host keeps every surrounding readout and numerical plane sum.


@dataclass(frozen=True)
class IntegerBatchedMatmulCall:
    operation_ordinal: int
    problem: object
    symbol: str
    borrowed_symbol: str
    kernel_symbol: str

    @property
    def m(self):
        return self.problem.m

    @property
    def k(self):
        return self.problem.k

    @property
    def n(self):
        return self.problem.n

    @property
    def shapes(self):
        return self.problem.lhs.shape, self.problem.rhs.shape, self.problem.output_shape


@dataclass(frozen=True)
class PreparedBatchedMatmulCalls:
    source: str
    original_sha256: str
    calls: tuple
    writers: tuple
    batching: object = None

    def borrowed_calls(self, path, *, pins=()):
        path = Path(path)

        def validate():
            if path.read_text() != self.source:
                raise ValueError("batched source changed before ordinary host lowering")
            if self.batching is not None:
                self.batching.revalidate()

        validate()
        return from_writers(path, self.writers, pins, validate)


def prepare_integer_batched_matmul_calls(
    source,
    *,
    select,
    borrowed_effects,
    entry="forward",
    symbol_prefix="integer_batch",
    allocation_alignment=64,
    refold_heads=False,
):
    """Outline actual zero-seeded signed batches; unsupported source stays intact.

    ``refold_heads`` explicitly applies the checked extract/reshape source
    transform first. Selection sees only typed current geometry. No target,
    numerical relaxation, implicit primitive or performance winner is supplied.
    """
    if (
        not callable(select)
        or type(borrowed_effects) is not BorrowedBufferEffects
        or type(refold_heads) is not bool
        or type(allocation_alignment) is not int
        or not 0 < allocation_alignment < 1 << 63
        or allocation_alignment & (allocation_alignment - 1)
    ):
        raise ValueError("explicit selection, borrowed effects and bounded alignment required")
    borrowed_effects.__post_init__()
    _symbol(symbol_prefix)
    module = parse_mlir_text(source)
    module.verify()
    entries = [op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == entry]
    if len(entries) != 1 or len(entries[0].body.blocks) != 1:
        raise ValueError("one original single-block tensor entry required")
    batching = plan_independent_head_batching(module) if refold_heads else None
    if batching is not None:
        module = batching.candidate.clone()
    reserved = {op.sym_name.data for op in module.body.block.ops if type(op) is func.FuncOp}
    plans = []
    from merlin.compiler.ir.contraction_shapes import zero_initialised

    for ordinal, op in enumerate(tuple(module.walk())):
        if type(op) is not GenericOp:
            continue
        try:
            problem = read_batched_integer_product(op)
        except (ValueError, AttributeError):
            continue
        if not zero_initialised(op):
            continue
        index = len(plans)
        while True:
            symbol = f"{symbol_prefix}_{index}"
            names = symbol, symbol + "_borrowed", symbol + "_kernel"
            if not reserved.intersection(names):
                break
            index += 1
        call = IntegerBatchedMatmulCall(ordinal, problem, *names)
        chosen = select(call)
        if type(chosen) is not bool:
            raise ValueError("batch selection must return an explicit Boolean")
        if chosen:
            plans.append((op, call))
            reserved.update(names)
    writers = []
    for op, binding in plans:
        lhs, rhs = op.inputs
        typ = op.results[0].type
        empty = tensor.EmptyOp((), typ)
        call = func.CallOp(binding.symbol, [lhs, rhs, empty.tensor], [typ])
        call.attributes.update(op.attributes)
        Rewriter.replace_op(op, [empty, call], call.results)
        attrs = [
            builtin.DictionaryAttr(
                {
                    "bufferization.access": builtin.StringAttr("read"),
                    "bufferization.buffer_layout": builtin.StridedLayoutAttr([None] * len(v.type.get_shape()), None),
                }
            )
            for v in (lhs, rhs)
        ]
        attrs.append(builtin.DictionaryAttr({"bufferization.access": builtin.StringAttr("write")}))
        declaration = func.FuncOp(
            binding.symbol,
            ([lhs.type, rhs.type, typ], [typ]),
            Region(),
            visibility="private",
            arg_attrs=builtin.ArrayAttr(attrs),
        )
        declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
        module.body.block.add_op(declaration)
        writers.append(
            FreshTensorWriterContract(
                binding.symbol,
                2,
                (2,),
                binding.borrowed_symbol,
                allocation_alignment,
                borrowed_effects=borrowed_effects,
            )
        )
    if not plans:
        return PreparedBatchedMatmulCalls(source, hashlib.sha256(source.encode()).hexdigest(), (), ())
    rewrite_fresh_tensor_writers(module, writers)
    module.verify()
    return PreparedBatchedMatmulCalls(
        ir_text(module),
        hashlib.sha256(source.encode()).hexdigest(),
        tuple(call for _, call in plans),
        tuple(writers),
        batching,
    )


@dataclass(frozen=True)
class ComposedBatchedMatmulProgram:
    preparation: PreparedBatchedMatmulCalls
    host: object
    kernels: tuple
    bridges: tuple


def lower_integer_batched_matmul_program(
    source,
    workdir,
    *,
    select,
    emit_matrix_kernel,
    logical_abi,
    data_layout,
    entry="forward",
    allocation_alignment=64,
    refold_heads=False,
    host_features=None,
):
    """Execute the existing ordinary host pipeline around explicit batch calls.

    ``emit_matrix_kernel(call)`` returns LLVM text plus immutable implementation
    pins. It implements ONE original (M,K,N) matrix, not a guessed flattened
    batch. The shared ranked bridge invokes it for every source head. Build,
    linking and complete numerical qualification remain ordinary caller steps.
    """
    from merlin.compiler.lower.model.lower import lower_model
    from merlin.compiler.lower.options.tensor_helper_inlining import composition_features

    if type(logical_abi) is not LogicalBorrowedABI or not callable(emit_matrix_kernel):
        raise ValueError("explicit selected logical ABI and matrix implementation required")
    logical_abi.__post_init__()
    prepared = prepare_integer_batched_matmul_calls(
        source,
        select=select,
        borrowed_effects=logical_abi.qualified_effects,
        entry=entry,
        allocation_alignment=allocation_alignment,
        refold_heads=refold_heads,
    )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "prepared.mlir"
    path.write_text(prepared.source)
    kernels, bridges, pins = [], [], []
    for index, call in enumerate(prepared.calls):
        llvm, implementation = emit_matrix_kernel(call)
        if not isinstance(llvm, str) or type(implementation) is not tuple:
            raise ValueError("matrix emitter needs LLVM and explicit immutable implementation pins")
        for location, digest in implementation:
            location = Path(location)
            if hashlib.sha256(location.read_bytes()).hexdigest() != digest:
                raise ValueError("matrix implementation changed during batch emission")
            pins.append((location, digest))
        kernel, bridge = workdir / f"kernel{index}.ll", workdir / f"bridge{index}.c"
        kernel.write_text(llvm)
        bridge.write_text(
            emit_borrowed_batch_translation_unit(
                logical_abi,
                call.problem,
                symbol="_mlir_ciface_" + call.borrowed_symbol,
                kernel=call.kernel_symbol,
            ).text
        )
        kernels.append(kernel)
        bridges.append(bridge)
    borrowed = prepared.borrowed_calls(path, pins=tuple(dict.fromkeys(pins))) if prepared.calls else None
    host = lower_model(
        prepared.source,
        workdir / "host",
        targets=(),
        index_bits=64,
        data_layout=data_layout,
        borrowed_buffer_calls=borrowed,
        hoist_static_allocs=False,
        features=composition_features(host_features, has_writers=bool(prepared.calls)),
    )
    return ComposedBatchedMatmulProgram(prepared, host, tuple(kernels), tuple(bridges))


# ---- convolution_packing -----------------------------------------------------------------------
# Bounded NHWC window transport into a caller-owned row-major matrix panel.
#
# Only input bytes and zero padding move here. The caller selects panel capacity,
# proves disjoint storage, and owns the device contraction and its completion.


class _WindowBuilder:
    def __init__(self, builder):
        self.builder = builder

    def value(self, value, dtype=i64):
        return self.builder.const(value, dtype) if type(value) is int else value

    def op(self, name, lhs, rhs, dtype="i64"):
        typ = i1 if dtype == "i1" else i64
        operation = {
            "add": llvm.AddOp,
            "sub": llvm.SubOp,
            "mul": llvm.MulOp,
            "udiv": llvm.UDivOp,
            "urem": llvm.URemOp,
            "and": llvm.AndOp,
        }[name]
        return self.builder.add(operation(self.value(lhs, typ), self.value(rhs, typ))).results[0]

    def cmp(self, predicate, lhs, rhs):
        flag = {
            "ult": llvm.ICmpPredicateFlag.ULT,
            "ule": llvm.ICmpPredicateFlag.ULE,
            "slt": llvm.ICmpPredicateFlag.SLT,
        }[predicate]
        return self.builder.add(llvm.ICmpOp(self.value(lhs), self.value(rhs), IntegerAttr(flag.to_int(), i64))).results[
            0
        ]

    def select(self, condition, lhs, rhs):
        return self.builder.add(llvm.SelectOp(condition, self.value(lhs), self.value(rhs))).results[0]


@dataclass(frozen=True)
class ClippedSpatialHaloRows:
    """The original pool's rectangular producer rows, in local row-major order.

    Clamping duplicates border producer values. This is legal only when the
    source observer proof separately admits removing minimum-valued pool pads
    after the original readout. The mapping does not collapse either readout.
    """

    spatial: SpatialMaxReadout
    output_columns: int

    def __post_init__(self):
        if type(self.spatial) is not SpatialMaxReadout:
            raise ValueError("halo packing requires the original typed pool geometry")
        self.spatial.__post_init__()
        s = self.spatial
        if type(self.output_columns) is not int or not 0 < self.output_columns <= s.output_hw[1]:
            raise ValueError("halo packing requires a bounded original output stripe")
        if any(s.padding) and s.pad_value != -128:
            raise ValueError("clipped halos require minimum-valued signed-i8 pool padding")
        kh, kw = s.window
        sh, sw = s.stride
        pt, pl, _, _ = s.padding
        h, w = s.input_hw
        oh, ow = s.output_hw
        if kh <= pt or kw <= pl or (oh - 1) * sh - pt >= h or (ow - 1) * sw - pl >= w:
            raise ValueError("clipped halos cannot replace completely padded pool windows")
        if max(self.rows, oh * sh + kh, ow * sw + kw) >= 1 << 63:
            raise ValueError("halo packing exceeds signed-i64 coordinate arithmetic")

    @property
    def width(self):
        return (self.output_columns - 1) * self.spatial.stride[1] + self.spatial.window[1]

    @property
    def rows(self):
        return self.spatial.window[0] * self.width


def emit_nhwc_window_panel(
    builder, window, source, destination, row_start, rows, *, source_destination_disjoint, maximum_panel_bytes
):
    """Emit counted byte-copy loops for ``[rows, kh*kw*channels]`` signed-i8 input.

    ``row_start`` is an unsigned logical output-row offset (i64). Rows beyond the
    window domain produce zeros and never read source memory. Real convolution
    callers must select only original output rows. Destination has at least the
    declared panel bytes; source holds the complete NHWC input. No allocation,
    target operation, reduction, quantization or implicit asynchronous lifetime.
    """
    if type(builder) is not FnBuilder or type(window) is not Window2D:
        raise ValueError("typed LLVM builder and explicit window geometry required")
    window.__post_init__()
    if source_destination_disjoint is not True:
        raise ValueError("window packing requires disjoint input and output storage")
    if any(not isinstance(p, SSAValue) or p.type != PTR for p in (source, destination)):
        raise ValueError("window packing requires byte-addressable pointers")
    if not isinstance(row_start, SSAValue) or row_start.type != i64:
        raise ValueError("window packing requires an i64 output-row start")
    if (
        type(rows) is not int
        or rows <= 0
        or type(maximum_panel_bytes) is not int
        or maximum_panel_bytes <= 0
        or maximum_panel_bytes >= 1 << 63
    ):
        raise ValueError("window packing requires a positive bounded panel budget")
    kh, kw = window.kernel
    channels = window.input_shape[-1]
    reduction = kh * kw * channels
    panel_bytes = rows * reduction
    if panel_bytes > maximum_panel_bytes:
        raise ValueError("window panel exceeds caller-selected packing capacity")
    if _emit_complete_unit_window(builder, window, source, destination, row_start, rows):
        return dict(_panel_receipt(rows, reduction, panel_bytes), transport="complete_unit_window_row_spans")
    ops = _WindowBuilder(builder)
    oh, ow = window.output_spatial
    start_valid = ops.cmp("ult", row_start, window.input_shape[0] * oh * ow)

    def row_mapping(panel_row):
        return ops.op("add", row_start, panel_row), start_valid

    _emit_window_bytes(builder, window, source, destination, rows, row_mapping)
    return _panel_receipt(rows, reduction, panel_bytes)


def _emit_complete_unit_window(builder, window, source, destination, row_start, rows):
    """Copy complete unit-stride 1x1 windows as contiguous row spans.

    Only a proved constant-zero start and the complete padded output domain
    enter this path. The original disjointness and byte budget were checked by
    the public caller. Runtime/partial starts retain the general checked map.
    No alignment or target memory-operation implementation is assumed.
    """
    start = getattr(row_start.owner, "value", None)
    if (
        window.kernel != (1, 1)
        or window.stride != (1, 1)
        or not isinstance(start, IntegerAttr)
        or start.value.data != 0
    ):
        return False
    b, h, w, c = window.input_shape
    oh, ow = window.output_spatial
    if rows != b * oh * ow:
        return False
    pt, pl, pb, pr = window.padding
    ops = _WindowBuilder(builder)

    def pointer(base, offset):
        return builder.gep(base, ops.value(offset), i8)

    def fill(offset, count):
        if count:
            builder.add(
                llvm.CallIntrinsicOp(
                    "llvm.memset.p0.i64",
                    [pointer(destination, offset), builder.const(0, i8), builder.const(count), builder.const(0, i1)],
                    [],
                )
            )

    def copy(source_offset, destination_offset, count):
        builder.add(
            llvm.CallIntrinsicOp(
                "llvm.memcpy.p0.p0.i64",
                [
                    pointer(destination, destination_offset),
                    pointer(source, source_offset),
                    builder.const(count),
                    builder.const(0, i1),
                ],
                [],
            )
        )

    if not any(window.padding):
        copy(0, 0, b * h * w * c)
        return True

    def image(image):
        destination_base = ops.op("mul", image, oh * ow * c)
        source_base = ops.op("mul", image, h * w * c)
        fill(destination_base, pt * ow * c)

        def row(y):
            destination_row = ops.op("add", destination_base, ops.op("mul", ops.op("add", y, pt), ow * c))
            source_row = ops.op("add", source_base, ops.op("mul", y, w * c))
            fill(destination_row, pl * c)
            copy(source_row, ops.op("add", destination_row, pl * c), w * c)
            fill(ops.op("add", destination_row, (pl + w) * c), pr * c)

        builder.for_loop(0, h, 1, row, retain_loop=True)
        fill(ops.op("add", destination_base, (pt + h) * ow * c), pb * ow * c)

    builder.for_loop(0, b, 1, image, retain_loop=True)
    return True


def _emit_window_bytes(builder, window, source, destination, rows, row_mapping):
    """Closed byte-copy implementation shared by the two declared row maps."""
    ops = _WindowBuilder(builder)
    kh, kw = window.kernel
    channels = window.input_shape[-1]
    reduction = kh * kw * channels

    def row_body(panel_row):
        row, row_valid = row_mapping(panel_row)

        def ky_body(ky):
            def kx_body(kx):
                run = window.emit_row_run(ops, row, 1, ky, kx, 0, channels)
                packed = ops.op(
                    "add",
                    ops.op("mul", panel_row, reduction),
                    ops.op("mul", ops.op("add", ops.op("mul", ky, kw), kx), channels),
                )
                valid, padding, after = Block(), Block(), Block()
                for block in (valid, padding, after):
                    builder.region.add_block(block)
                builder.add(llvm.CondBrOp(ops.op("and", row_valid, run["predicate"], "i1"), valid, [], padding, []))
                builder.blk = valid

                def copy(channel):
                    src = builder.gep(source, ops.op("add", run["first_element"], channel), i8)
                    dst = builder.gep(destination, ops.op("add", packed, channel), i8)
                    value = builder.add(llvm.LoadOp(src, i8)).results[0]
                    builder.add(llvm.StoreOp(value, dst))

                builder.for_loop(0, channels, 1, copy, retain_loop=True)
                builder.add(llvm.BrOp(after))
                builder.blk = padding

                def zero(channel):
                    dst = builder.gep(destination, ops.op("add", packed, channel), i8)
                    builder.add(llvm.StoreOp(builder.const(0, i8), dst))

                builder.for_loop(0, channels, 1, zero, retain_loop=True)
                builder.add(llvm.BrOp(after))
                builder.blk = after

            builder.for_loop(0, kw, 1, kx_body, retain_loop=True)

        builder.for_loop(0, kh, 1, ky_body, retain_loop=True)

    builder.for_loop(0, rows, 1, row_body, retain_loop=True)


def _panel_receipt(rows, reduction, panel_bytes):
    return {
        "kind": "host_input_transport",
        "panel_shape": [rows, reduction],
        "panel_bytes": panel_bytes,
        "input_layout": "nhwc",
        "reduction_order": "kh,kw,channel",
        "zero_padding": True,
        "device_arithmetic": False,
        "storage_completion": "synchronous host copies; caller owns device consumer lifetime",
    }


def emit_nhwc_halo_panel(
    builder,
    window,
    halo,
    source,
    destination,
    image,
    output_y,
    output_x,
    row_start,
    rows,
    *,
    source_destination_disjoint,
    maximum_panel_bytes,
):
    """Pack a bounded subset of the original pool halo into ``[rows, K]``.

    Runtime coordinates are unsigned domain-checked before any source load.
    K is exactly kh/kw/channel order, including original convolution zero pads.
    Rows crossing a halo scanline retain their correct source coordinates.
    Out-of-domain calls and rows produce zeros. No allocation or device effect
    occurs; the caller must bind the source observer and private storage proofs.
    """
    if type(builder) is not FnBuilder or type(window) is not Window2D or type(halo) is not ClippedSpatialHaloRows:
        raise ValueError("halo packing requires typed builder, source window and pool row mapping")
    window.__post_init__()
    halo.__post_init__()
    if window.output_spatial != halo.spatial.input_hw:
        raise ValueError("halo pool input differs from the original producer output")
    if source_destination_disjoint is not True:
        raise ValueError("halo packing requires disjoint input and output storage")
    if any(not isinstance(p, SSAValue) or p.type != PTR for p in (source, destination)):
        raise ValueError("halo packing requires byte-addressable pointers")
    if any(not isinstance(v, SSAValue) or v.type != i64 for v in (image, output_y, output_x, row_start)):
        raise ValueError("halo packing coordinates require i64")
    reduction = window.kernel[0] * window.kernel[1] * window.input_shape[-1]
    if (
        type(rows) is not int
        or not 0 < rows < 1 << 63
        or type(maximum_panel_bytes) is not int
        or not 0 < maximum_panel_bytes < 1 << 63
        or rows * reduction > maximum_panel_bytes
    ):
        raise ValueError("halo panel exceeds caller-selected packing capacity")
    ops = _WindowBuilder(builder)
    s = halo.spatial
    h, w = s.input_hw
    oh, ow = s.output_hw

    def both(lhs, rhs):
        return ops.op("and", lhs, rhs, "i1")

    valid = both(ops.cmp("ult", image, window.input_shape[0]), ops.cmp("ult", output_y, oh))
    valid = both(valid, ops.cmp("ule", output_x, ow - halo.output_columns))
    valid = both(valid, ops.cmp("ult", row_start, halo.rows))

    def clip(value, extent):
        nonnegative = ops.select(ops.cmp("slt", value, 0), 0, value)
        return ops.select(ops.cmp("ult", nonnegative, extent), nonnegative, extent - 1)

    def row_mapping(panel_row):
        # The subtraction predicate prevents row_start+panel_row wrapping back
        # into the domain even when the runtime start is UINT64_MAX.
        row_valid = both(valid, ops.cmp("ult", panel_row, ops.op("sub", halo.rows, row_start)))
        local = ops.op("add", row_start, panel_row)
        y = ops.op(
            "add", ops.op("sub", ops.op("mul", output_y, s.stride[0]), s.padding[0]), ops.op("udiv", local, halo.width)
        )
        x = ops.op(
            "add", ops.op("sub", ops.op("mul", output_x, s.stride[1]), s.padding[1]), ops.op("urem", local, halo.width)
        )
        row = ops.op("add", ops.op("mul", ops.op("add", ops.op("mul", image, h), clip(y, h)), w), clip(x, w))
        return row, row_valid

    _emit_window_bytes(builder, window, source, destination, rows, row_mapping)
    return dict(
        _panel_receipt(rows, reduction, rows * reduction),
        row_mapping="clipped_original_pool_halo",
        halo_shape=[s.window[0], halo.width],
        output_columns=halo.output_columns,
        source_observer_qualification="caller-owned; not inferred",
    )
