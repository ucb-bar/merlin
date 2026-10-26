"""Original-SSA host/device preparation through explicit ordinary emitters.

The initial source family is the existing checked sibling integer outliner.
Host epilogues and outputs remain original SSA. This adapter assembles source,
pointer transport and actual callback products; it is not an execution grant or
a replacement for final source-work, borrowed-call, object and numerical gates.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


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
        current = prepare_source_program(self.source, effects=self.prepared.writer.borrowed_effects,
            target=self.target, entry_symbol=self.entry_symbol, device_symbol=self.device_symbol)
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

    from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.device_shim import LogicalBuffer, emit_logical_translation_unit
    from merlin.llvmlower.source_region_host import emit_ranked_source_entry, prepare_sibling_source_host, tensor_abi
    from merlin.runtime.host_arithmetic import _identifier
    from merlin.xdsl_dialects._common import text

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
    original_owners = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)
                       and op.sym_name.data == "forward"]
    if len(original_owners) != 1 or len(original_owners[0].body.blocks) != 1:
        raise ValueError("source program requires one original closed forward block")
    original_owner = original_owners[0]
    original_roots = []
    for ordinal, op in enumerate(original_owner.body.block.ops):
        inputs = tuple(getattr(op, "inputs", ()))
        if (op.name in ("linalg.generic", "linalg.matmul") and len(inputs) == 2 and len(op.results) == 1
                and tuple(str(value.type.element_type) for value in (*inputs, op.results[0])) == ("i8", "i8", "i32")):
            original_roots.append((ordinal, op, read_zero_seeded_integer_matmul(op)))
    prepared = prepare_sibling_source_host(module, effects=effects)
    owner = next(op for op in prepared.module.body.block.ops if isinstance(op, func.FuncOp)
                 and op.sym_name.data == "forward")
    owner.attributes["llvm.emit_c_interface"] = UnitAttr()
    original_inputs, original_outputs = tensor_abi(module)
    abi = tuple((row["name"], tuple(row["shape"]), row["dtype"],
                 "read" if index < len(original_inputs) else "write")
                for index, row in enumerate((*original_inputs, *original_outputs)))
    callee = next(op for op in prepared.core.body.block.ops if isinstance(op, func.FuncOp))
    types = (*callee.function_type.inputs, *callee.function_type.outputs)
    count = len(callee.function_type.inputs)
    if any(len(typ.get_shape()) != 2 for typ in types):
        raise ValueError("initial source-program transport supports the exact rank-two sibling family")
    buffers = tuple(LogicalBuffer("core_" + str(i), "input" if i < count else "output",
        "input_" + str(i) if i < count else "out_" + str(i - count), tuple(typ.get_shape()), str(typ.element_type))
        for i, typ in enumerate(types))
    core_abi = tuple((b.name, b.shape, b.dtype, "read" if i < count else "write")
                     for i, b in enumerate(buffers))
    transport = emit_logical_translation_unit(target, {"source_core_transport": (
        buffers, [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers])},
        kernel_symbol_for=lambda _: device_symbol)
    if transport.skipped or transport.symbols != ("source_core_transport",):
        raise ValueError("ordinary source-core pointer transport refused the complete ABI")
    n = len(buffers)
    bridge = (transport.text + "\nvoid _mlir_ciface_" + prepared.writer.borrowed_symbol + "("
        + ",".join("const merlin_lmemref_2d*d" + str(i) for i in range(n))
        + ") {source_core_transport_results result;_mlir_ciface_source_core_transport(&result,"
        + ",".join("d" + str(i) for i in range(n)) + ");}\n")
    # Read members from the actual outlined core; the existing outliner retains
    # source region attributes. These are source links, not instruction credit.
    roots = [op for op in prepared.core.walk() if op.name in ("linalg.generic", "linalg.matmul")]
    if (len(roots) != len(original_roots)
            or any(read_zero_seeded_integer_matmul(root) != geometry
                   or root.attributes.get("prov.region_id") != original.attributes.get("prov.region_id")
                   for root, (_, original, geometry) in zip(roots, original_roots, strict=True))):
        raise ValueError("outlined core does not retain every ordered original integer root")
    # Ordinals refer to ORIGINAL source, not positions after view CSE/hoisting.
    members = tuple((ordinal, getattr(op.attributes.get("prov.region_id"), "data", None))
                    for ordinal, op, _ in original_roots)
    if text(module, generic=True) != before:
        raise ValueError("source-program preparation changed original SSA")
    return SourceProgram(source, text(prepared.module, generic=True), text(prepared.core, generic=True),
        emit_ranked_source_entry(module, entry_symbol=entry_symbol), bridge, abi, core_abi, members,
        prepared, target, entry_symbol, device_symbol)


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
    from merlin.frontends.linalg_mlir import parse_mlir_text

    if type(program) is not SourceProgram or not callable(emit_device) or not callable(emit_host):
        raise ValueError("prepared original source and actual emitter callbacks required")
    current = program.revalidate()
    outputs = []
    for source, callback, options in (
        (current.core, emit_device, {"symbol": current.device_symbol, "abi": current.core_abi,
                                    "source_members": current.source_members}),
        (current.host, emit_host, {"writers": (current.prepared.writer,)}),
    ):
        product = callback(parse_mlir_text(source), **options)
        if (type(product) is not SourceEmission or product.source_sha256 != _sha(source)
                or type(product.data) is not bytes or not product.data or len(product.data) > 64 << 20
                or product.format not in ("llvm_ir", "llvm_dialect", "object")):
            raise ValueError("actual emission does not match the prepared source component")
        outputs.append(product)
        program.revalidate()
    device, host = outputs
    if device.format != "llvm_dialect":
        raise ValueError("device source bridge requires inspectable LLVM-dialect ABI")
    from xdsl.dialects import llvm
    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    emitted = Parser(make_llvm_context(), device.data.decode()).parse_module()
    emitted.verify()
    symbols = [op for op in emitted.body.block.ops if isinstance(op, llvm.FuncOp)
               and op.sym_name.data == current.device_symbol]
    if (len(symbols) != 1 or not symbols[0].body.blocks
            or symbols[0].linkage != llvm.LinkageAttr("external")
            or tuple(symbols[0].function_type.inputs) != (llvm.LLVMPointerType(),) * len(current.core_abi)
            or symbols[0].function_type.output != llvm.LLVMVoidType()):
        raise ValueError("actual device function does not implement the complete logical pointer ABI")
    return SourceProgramEmission(device, host, current.entry_transport, current.core_transport, {
        "schema": "merlin.source_program_emission.v1", "source_sha256": _sha(current.source),
        "host_source_sha256": _sha(current.host), "core_source_sha256": _sha(current.core),
        "device_product_sha256": _sha(device.data), "host_product_sha256": _sha(host.data),
        "entry_transport_sha256": _sha(current.entry_transport), "core_transport_sha256": _sha(current.core_transport),
        "logical_abi": current.abi, "device_abi": current.core_abi,
        "outlined_source_members": current.source_members,
        "original_epilogues_and_outputs_retained": True,
        "scope": "actual component emission and ABI transport; final owner admission remains required",
        "physical_effects_qualified": False, "target_execution": False,
        "unavoidable_source_work_attribution": "requires independent final owner audit",
    })
