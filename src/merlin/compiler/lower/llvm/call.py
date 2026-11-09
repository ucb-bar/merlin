"""Admit actual native memref interfaces and build a typed synchronous CPU adapter.

Only immutable tensor source, a complete observed descriptor forwarding body,
and output-rooted native writes are admitted. LLVM owns descriptor layout and
calling convention; this helper imports no compiler driver or planning policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from math import prod

from xdsl.context import Context
from xdsl.dialects import builtin, func, llvm
from xdsl.ir import Block, Dialect, Region
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.compiler.backend.cabi import CRegionABI, TensorPort
from merlin.frontends.linalg_mlir import parse_mlir_text


@dataclass(frozen=True)
class ScalarCall:
    source: str
    source_entry: str
    compute: str
    symbol: str
    inputs: tuple[TensorPort, ...]
    outputs: tuple[TensorPort, ...]
    initialization: dict | None = None
    workspace: object | None = None
    workspace_implementation: dict | None = None


def prepare(source, entry, namespace, *, workspace_bytes=0):
    """Rename a private clone; retain every original tensor computation."""
    module = parse_mlir_text(source)
    module.verify()
    definitions = tuple(module.body.block.ops)
    if (len(definitions) != 1 or type(definitions[0]) is not func.FuncOp or
            definitions[0].sym_name.data != entry or len(definitions[0].body.blocks) != 1):
        raise ValueError("scalar call requires one complete selected tensor entry")
    function = definitions[0]
    if (module.properties or any(not key.startswith("prov.") and key != "llvm.data_layout"
                                 for key in module.attributes) or
            any(not key.startswith("prov.") and key != "llvm.emit_c_interface" for key in function.attributes)):
        raise ValueError("scalar call source has unsupported module/function semantic or ABI attributes")
    if function.arg_attrs is not None or function.res_attrs is not None:
        raise ValueError("scalar call does not remap additional argument/result ABI attributes")
    if set(function.properties) - {"sym_name", "function_type", "sym_visibility"}:
        raise ValueError("scalar call entry has unsupported signature properties")
    inputs = tuple(TensorPort.from_type(t) for t in function.function_type.inputs)
    outputs = tuple(TensorPort.from_type(t) for t in function.function_type.outputs)
    if not outputs or any(len(p.shape) > 16 for p in (*inputs, *outputs)):
        raise ValueError("scalar call requires tensor results and bounded static descriptor ranks")
    for operation in function.body.walk():
        if (isinstance(operation, builtin.UnregisteredOp) or
                operation.name not in {"func.return"} and
                operation.name.partition(".")[0] not in {"arith", "math", "tensor", "linalg", "scf", "cf"}):
            raise ValueError("scalar call source cannot contain opaque calls, buffers, vectors or external effects")
    transformed = module.clone()
    target = transformed.body.block.first_op
    target.properties["sym_name"] = builtin.StringAttr(namespace + "_compute")
    # Only symbol identity changes. The whole function body is the original SSA.
    if not function.body.is_structurally_equivalent(target.body):
        raise ValueError("scalar call entry renaming changed source computations")
    transformed.verify()
    import hashlib
    from pathlib import Path

    from merlin.common.dense_float_ir import DenseFloatPrinter
    from merlin.compiler.artifact import digest
    from merlin.compiler.lower.llvm import workspace as workspace_owner
    from merlin.compiler.transforms import initialization
    from merlin.llvmlower import prov_cse

    workspace_path = Path(workspace_owner.__file__)
    workspace_digest = digest(workspace_path)
    transformed, workspace = workspace_owner.promote(transformed, limit=workspace_bytes)
    if digest(workspace_path) != workspace_digest:
        raise ValueError("source tensor workspace implementation changed during preparation")
    workspace_implementation = ({"path": str(workspace_path), "sha256": workspace_digest}
                                if workspace is not None else None)
    stream = StringIO()
    DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(transformed)
    before = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    owner, policy = Path(initialization.__file__), Path(prov_cse.__file__)
    owner_digest, policy_digest = digest(owner), digest(policy)
    selected, replacements = initialization.normalize_destinations(transformed)
    stream = StringIO()
    DenseFloatPrinter(stream=stream, print_generic_format=True).print_op(selected)
    text = stream.getvalue()
    if digest(owner) != owner_digest or digest(policy) != policy_digest:
        raise ValueError("scalar destination initialization source changed during preparation")
    record = {"input_sha256": before, "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "replacements": list(replacements), "implementation": {"path": str(owner), "sha256": owner_digest},
              "policy_implementation": {"path": str(policy), "sha256": policy_digest},
              "verification": "whole-original-SSA structural replay; original scalar values retained",
              "scope": "static destination splats; static empty provenance in sidecar; "
                       "not numerical or execution qualification"}
    return ScalarCall(text, entry, namespace + "_compute", namespace, inputs, outputs, record,
                      workspace, workspace_implementation)


def _name(operation):
    return operation.attributes["op_name__"].data if isinstance(operation, builtin.UnregisteredOp) else operation.name


def _descriptor(rank, index):
    pointer = llvm.LLVMPointerType()
    return llvm.LLVMStructType.from_type_list(
        [pointer, pointer, index, llvm.LLVMArrayType(rank, index), llvm.LLVMArrayType(rank, index)])


def _void(function):
    if (set(function.properties) - {"CConv", "function_type", "linkage", "sym_name", "visibility_",
                                        "sym_visibility", "unnamed_addr"}
            or any(not key.startswith("prov.") and key != "llvm.emit_c_interface" for key in function.attributes)):
        raise ValueError("native scalar call has unsupported function ABI or effect attributes")
    unnamed = function.properties.get("unnamed_addr")
    if unnamed is not None and (type(unnamed) is not builtin.IntegerAttr or unnamed.value.data != 0):
        raise ValueError("native scalar call cannot change function address/ownership obligations")
    if function.linkage.linkage.data != "external":
        raise ValueError("native scalar call requires uniquely linkable strong entry definitions")
    if (function.function_type.is_variadic or type(function.function_type.output) is not llvm.LLVMVoidType or
            function.CConv.convention.data != "ccc" or not function.body.blocks):
        raise ValueError("native scalar call requires defined nonvariadic void C-convention entries")
    if any(key in function.properties or key in function.attributes for key in ("arg_attrs", "res_attrs")):
        raise ValueError("native scalar call has unadmitted argument/result ABI attributes")


def _forwarding(function, compute, ports, index):
    """Replay the complete actual CIFace body against its tensor descriptor roster."""
    _void(function)
    if (len(function.body.blocks) != 1 or
            tuple(function.function_type.inputs) != (llvm.LLVMPointerType(),) * len(ports)):
        raise ValueError("native scalar C interface differs from its complete ordered tensor signature")
    expected = Block(arg_types=list(function.function_type.inputs))
    forwarded = []
    for argument, port in zip(expected.args, ports, strict=True):
        descriptor = llvm.LoadOp(argument, _descriptor(len(port.shape), index))
        expected.add_op(descriptor)
        positions = [(0,), (1,), (2,)] + [(group, axis) for group in (3, 4) for axis in range(len(port.shape))]
        for position in positions:
            typ = llvm.LLVMPointerType() if position[0] < 2 else index
            extract = llvm.ExtractValueOp(builtin.DenseArrayBase.from_list(builtin.i64, position), descriptor, typ)
            expected.add_op(extract)
            forwarded.append(extract.results[0])
    expected.add_ops([llvm.CallOp(builtin.SymbolRefAttr(compute.sym_name.data), *forwarded), llvm.ReturnOp()])
    context = {expected: function.body.block}
    context.update(zip(expected.args, function.body.block.args, strict=True))
    actual, planned = tuple(function.body.block.ops), tuple(expected.ops)
    if len(actual) != len(planned) or not all(
            left.is_structurally_equivalent(right, context) for left, right in zip(planned, actual, strict=True)):
        raise ValueError("native scalar C interface does not exactly forward complete descriptor fields and outcomes")
    if tuple(compute.function_type.inputs) != tuple(value.type for value in forwarded):
        raise ValueError("native scalar implementation signature differs from its actual descriptor forwarding")


def _memory(compute, module, ports, input_count):
    """Close native pointer origins; writes cannot reach inputs, globals or escaping storage."""
    _void(compute)
    from merlin.compiler.lower.llvm.copies import CopyBounds

    layout = module.attributes.get("llvm.data_layout")
    if type(layout) is not builtin.StringAttr:
        raise ValueError("native scalar memory admission needs the selected data layout")
    copies = CopyBounds(compute, ports, input_count, layout.data)
    arguments, cursor, roots = tuple(compute.body.blocks.first.args), 0, {}
    for number, port in enumerate(ports):
        for argument in arguments[cursor:cursor + 2]:
            roots[argument] = ("input" if number < input_count else "output", number)
        cursor += 3 + 2 * len(port.shape)
    if cursor != len(arguments):
        raise ValueError("native scalar pointer roster is incomplete")
    globals_ = {}
    for operation in module.body.block.ops:
        if _name(operation) == "llvm.mlir.global":
            if "constant" not in operation.properties:
                raise ValueError("native scalar call cannot retain mutable global storage")
            globals_[operation.properties["sym_name"].data] = operation
    written = set()
    pure = {"llvm.mlir.constant", "llvm.br", "llvm.cond_br", "llvm.return", "llvm.add", "llvm.sub", "llvm.mul",
            "llvm.fadd", "llvm.fsub", "llvm.fmul", "llvm.fdiv", "llvm.frem", "llvm.fneg", "llvm.icmp", "llvm.fcmp",
            "llvm.select", "llvm.and", "llvm.or", "llvm.xor", "llvm.shl", "llvm.lshr", "llvm.ashr", "llvm.sdiv",
            "llvm.udiv", "llvm.srem", "llvm.urem", "llvm.trunc", "llvm.sext", "llvm.zext", "llvm.sitofp", "llvm.uitofp",
            "llvm.fptosi", "llvm.fptoui", "llvm.fpext", "llvm.fptrunc", "llvm.intr.fabs", "llvm.intr.sqrt",
            "llvm.intr.fma", "llvm.intr.maxnum", "llvm.intr.minnum", "llvm.intr.maximum", "llvm.intr.minimum"}
    memory_ops = {"llvm.mlir.addressof", "llvm.getelementptr", "llvm.bitcast", "llvm.load", "llvm.store"}
    for block in compute.body.blocks:
        if block is not compute.body.blocks.first and any(
                isinstance(arg.type, llvm.LLVMPointerType) for arg in block.args):
            raise ValueError("native scalar pointer loop-carried state requires a separate storage admission")
        for operation in block.ops:
            name = _name(operation)
            if copies.observe(operation, name):
                continue
            if name == "llvm.intr.memcpy":
                written.add(copies.admit(operation))
                continue
            pointers = [v for v in operation.results if isinstance(v.type, llvm.LLVMPointerType)]
            if name not in pure | memory_ops:
                raise ValueError("native scalar private workspace or external effects require "
                                 "an explicit resource/runtime ABI")
            if name == "llvm.mlir.addressof":
                ref = operation.properties["global_name"]
                if ref.nested_references or ref.root_reference.data not in globals_ or len(pointers) != 1:
                    raise ValueError("native scalar address has no admitted immutable global")
                roots[pointers[0]] = ("constant", ref.root_reference.data)
            elif name == "llvm.getelementptr" or name == "llvm.bitcast" and pointers:
                if not operation.operands or operation.operands[0] not in roots or len(pointers) != 1:
                    raise ValueError("native scalar pointer has an unknown base")
                roots[pointers[0]] = roots[operation.operands[0]]
            elif pointers:
                raise ValueError("native scalar pointer-producing operation has no admitted provenance")
            if name == "llvm.load":
                if operation.operands[0] not in roots or pointers:
                    raise ValueError("native scalar load has no admitted tensor/constant origin")
            elif name == "llvm.store":
                value, address = operation.operands
                if (roots.get(address, (None,))[0] != "output" or
                        isinstance(value.type, (llvm.LLVMPointerType, llvm.LLVMStructType, llvm.LLVMArrayType))):
                    raise ValueError("native scalar write can mutate inputs, escape pointers or reach unknown storage")
                written.add(roots[address][1])
            elif any(isinstance(v.type, llvm.LLVMPointerType) for v in operation.operands) and name not in {
                    "llvm.getelementptr", "llvm.bitcast"}:
                raise ValueError("native scalar pointer use has unadmitted effects")
    if written != set(range(input_count, len(ports))):
        raise ValueError("native scalar outputs are missing observed write owners")
    record = {"inputs": "read_only", "writes": "selected_outputs_only", "tensor_scratch_bytes": 0,
              "retained_pointers": False, "completion": "ordinary_serial_CPU_return"}
    if copies.copies:
        record["bounded_copies"] = copies.copies
    return record


def _parse_native(generic):
    context = Context(allow_unregistered=True)
    context.load_dialect(builtin.Builtin)
    context.load_dialect(Dialect("llvm", [llvm.FuncOp, llvm.LoadOp, llvm.ExtractValueOp,
                                        llvm.CallOp, llvm.ReturnOp, llvm.StoreOp], list(llvm.LLVM.attributes)))
    return Parser(context, generic).parse_module()


def admit(generic, call: ScalarCall, *, index_bits, data_layout):
    """Consume actual native generic LLVM dialect, never an inferred symbol-only ABI."""
    module = _parse_native(generic)
    if module.attributes.get("llvm.data_layout") != builtin.StringAttr(data_layout):
        raise ValueError("native scalar interface changed the selected data layout")
    definitions = tuple(op for op in module.body.block.ops if type(op) is llvm.FuncOp)
    if any(type(op) is not llvm.FuncOp and _name(op) != "llvm.mlir.global" for op in module.body.block.ops):
        raise ValueError("native scalar module has unknown top-level effects or bindings")
    functions = {op.sym_name.data: op for op in definitions}
    ciface_name = "_mlir_ciface_" + call.compute
    if len(definitions) != 2 or len(functions) != 2 or set(functions) != {call.compute, ciface_name}:
        raise ValueError("native scalar output has missing, duplicate or external function bindings")
    compute, ciface = functions[call.compute], functions[ciface_name]
    ports = (*call.inputs, *call.outputs, *(call.workspace.ports if call.workspace is not None else ()))
    index = builtin.IntegerType(index_bits)
    for port in ports:
        if prod(port.shape) * 8 >= 1 << (index_bits - 1):
            raise ValueError("scalar tensor extent exceeds the selected descriptor index range")
    ciface.verify()
    _forwarding(ciface, compute, ports, index)
    memory = _memory(compute, module, ports, len(call.inputs))
    if call.workspace is not None:
        if call.workspace.bytes >= 1 << (index_bits - 1):
            raise ValueError("source workspace exceeds selected native descriptor index range")
        memory = {**memory, "writes": "selected_outputs_and_source_workspace",
                  "source_workspace_bytes": call.workspace.bytes, "additional_tensor_scratch_bytes": 0}
    wrapper = _wrapper(call, index, data_layout)
    widths = {"i8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4, "i64": 8, "f64": 8}
    abi = CRegionABI(call.symbol, call.inputs, call.outputs,
                     max(widths[p.dtype] for p in (*call.inputs, *call.outputs)),
                     ("module.o", "call.o"), resource_kind="cpu")
    if call.workspace is not None:
        from merlin.compiler.backend.workspace import CWorkspaceABI

        abi = CWorkspaceABI(abi, call.workspace)
    return abi, wrapper, memory


def _wrapper(call, index, layout):
    """Build descriptor objects in typed LLVM so its selected layout owns padding."""
    pointer = llvm.LLVMPointerType()
    block = Block(arg_types=[pointer, pointer, pointer])
    def add(operation):
        block.add_op(operation)
        return operation.results[0] if operation.results else None
    def constant(value):
        return add(llvm.ConstantOp(builtin.IntegerAttr(value, index), index))
    one, zero = constant(1), constant(0)
    descriptors = []
    groups = ((1, call.inputs), (2, call.outputs))
    if call.workspace is not None:
        groups = (*groups, (0, call.workspace.ports))
    for group, ports in groups:
        for number, port in enumerate(ports):
            if group == 0:
                data = add(llvm.GEPOp.from_mixed_indices(block.args[0],
                           [constant(call.workspace.offsets[number])], builtin.i8))
            else:
                slot = add(llvm.GEPOp.from_mixed_indices(block.args[group], [constant(number)], pointer))
                data = add(llvm.LoadOp(slot, pointer))
            typ = _descriptor(len(port.shape), index)
            value = add(llvm.UndefOp(typ))
            fields = [((0,), data), ((1,), data), ((2,), zero)]
            stride = 1
            strides = []
            for extent in reversed(port.shape):
                strides.append(stride)
                stride *= extent
            for axis, (extent, stride) in enumerate(zip(port.shape, reversed(strides), strict=True)):
                fields.extend([((3, axis), constant(extent)), ((4, axis), constant(stride))])
            for position, item in fields:
                value = add(llvm.InsertValueOp(builtin.DenseArrayBase.from_list(builtin.i64, position), value, item))
            allocation = llvm.AllocaOp(one, typ)
            allocation.properties.pop("alignment", None)
            storage = add(allocation)
            block.add_op(llvm.StoreOp(value, storage))
            descriptors.append(storage)
    block.add_op(llvm.CallOp(builtin.SymbolRefAttr("_mlir_ciface_" + call.compute), *descriptors))
    status = add(llvm.ConstantOp(builtin.IntegerAttr(0, builtin.i32), builtin.i32))
    block.add_op(llvm.ReturnOp(status))
    declaration = llvm.FuncOp("_mlir_ciface_" + call.compute,
                              llvm.LLVMFunctionType([pointer] * len(descriptors), llvm.LLVMVoidType()),
                              linkage=llvm.LinkageAttr("external"))
    function = llvm.FuncOp(call.symbol, llvm.LLVMFunctionType([pointer] * 3, builtin.i32),
                           linkage=llvm.LinkageAttr("external"), body=Region(block))
    module = builtin.ModuleOp([declaration, function], attributes={"llvm.data_layout": builtin.StringAttr(layout)})
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()
