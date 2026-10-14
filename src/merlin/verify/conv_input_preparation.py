"""Closed, bounded address/copy proof for a reviewed logical convolution input ABI.

No tensor values are evaluated. Loaded bytes carry only input/zero provenance;
any arithmetic on them, host output store, opaque pointer escape or uncounted
control flow refuses the exemption. Original device-work/numerical gates remain.
"""

from collections import Counter
from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType, ModuleOp, UnitAttr
from xdsl.dialects.llvm import AddressOfOp, GlobalOp, LLVMArrayType, LLVMPointerType

from merlin.perf.constant_control_cfg import integer_operation
from merlin.perf.host_cfg_activity import _INTEGER, _payload_bytes
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


@dataclass(frozen=True)
class _Pointer:
    root: object
    offset: int = 0


@dataclass(frozen=True)
class _Copy:
    origins: tuple


def _constant_zero_storage(address, function):
    """Resolve only an actual, closed read-only byte initializer in this module."""
    module = function.parent_op()
    if not isinstance(module, ModuleOp) or function.parent_block() is not module.body.block:
        raise ValueError("zero padding has no enclosing original module storage")
    if any(op.name not in {"llvm.mlir.global", "llvm.func"} for op in module.body.block.ops):
        raise ValueError("zero padding module has unproved aliases or storage effects")
    reference = address.global_name
    if reference.nested_references.data:
        raise ValueError("zero padding has nested or unknown symbol provenance")
    name = reference.root_reference.data
    definitions = [op for op in module.body.block.ops if getattr(op.properties.get("sym_name"), "data", None) == name]
    if len(definitions) != 1 or not isinstance(storage := definitions[0], GlobalOp):
        raise ValueError("zero padding global is missing or ambiguous")
    typ, initializer = storage.global_type, storage.value
    if (
        not isinstance(storage.constant, UnitAttr)
        or storage.linkage.linkage.data not in {"private", "internal"}
        or storage.body.blocks
        or storage.thread_local_ is not None
        or storage.section is not None
        or storage.attributes
        or storage.addr_space.value.data != 0
        or not isinstance(address.result.type, LLVMPointerType)
        or getattr(address.result.type.addr_space, "data", 0) != 0
        or not isinstance(typ, LLVMArrayType)
        or typ.type != IntegerType(8)
        or not 0 < typ.size.data <= 1 << 20
        or not isinstance(initializer, DenseIntOrFPElementsAttr)
        or initializer.get_element_type() != IntegerType(8)
        or initializer.get_shape() != (typ.size.data,)
        or len(initializer.data.data) != typ.size.data
        or any(initializer.data.data)
    ):
        raise ValueError("zero padding global lacks bounded read-only all-zero byte storage")
    # A second function could mutate or export an address. This finite proof
    # covers only the actual kernel's interpreted pointer uses, not other code.
    for op in module.walk():
        if isinstance(op, AddressOfOp) and op.global_name.root_reference.data == name:
            if op.global_name.nested_references.data or op.parent_op() is not function:
                raise ValueError("zero padding global address escapes the proven kernel")
    return storage, typ.size.data


def _unpoisoned_pointer_add(op):
    if op.results[0].type != IntegerType(64) or any(v.type != IntegerType(64) for v in op.operands):
        raise ValueError("pointer arithmetic truncates the selected register width")
    for key in ("overflowFlags", "overflow_flags", "isExact", "isDisjoint"):
        flags = op.properties.get(key)
        value = getattr(getattr(flags, "value", None), "data", getattr(flags, "data", None))
        if flags is not None and value not in ((), 0, False):
            raise ValueError("pointer arithmetic has unproved poison flags")


def prove_input_preparation(function, activity, context, *, target):
    """Return independently counted preparation totals, or a closed refusal."""
    try:
        return _prove(function, activity, context, target=target)
    except (ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError) as exc:
        return {"status": "unproven", "detail": str(exc)[:300]}


def _prove(function, activity, context, *, target):
    from merlin.targetgen.rocc.decode import _asm_template, _parse_insn, _semantics

    if activity.get("status") != "derived":
        raise ValueError("input preparation requires independently finite CFG counts")
    buffers = context["call_buffers"]
    if len(buffers) != 3 or [b["role"] for b in buffers] != ["input_0", "weight_0", "out_0"]:
        raise ValueError("input preparation requires reviewed three-role ABI")
    if [b["dtype"] for b in buffers] != ["i8", "i8", "i32"]:
        raise ValueError("input preparation requires exact reviewed integer ABI")
    blocks = tuple(function.body.blocks)
    if not blocks or len(blocks[0].args) != 3:
        raise ValueError("actual kernel argument count differs from reviewed ABI")
    counts = {blocks[r["block"]]: r["execution_count"] for r in activity["blocks"]}
    if (
        any(type(n) is not int or n < 0 for n in counts.values())
        or sum(counts.values()) > 100000
        or sum(len(tuple(b.ops)) * n for b, n in counts.items()) > 1000000
    ):
        raise ValueError("input preparation exceeds finite proof bound")
    semantics = getattr(_semantics(target), "operand_observer", None)
    if semantics is None or getattr(semantics, "word_bits", None) != 64:
        raise ValueError("input preparation has no explicitly selected 64-bit operand observer")
    isa = semantics.isa_constants(target)
    from merlin.targetgen.rtl.firrtl_load_states import LoadStateMapping

    load_states = getattr(semantics, "load_states", None)
    if load_states is not None and type(load_states) is not LoadStateMapping:
        raise ValueError("input preparation load states lack typed original source evidence")
    values = {arg: _Pointer(i) for i, arg in enumerate(blocks[0].args)}
    capacity = {i: prod(b["shape"]) * (4 if b["dtype"] == "i32" else 1) for i, b in enumerate(buffers)}
    memory, consumed, initialized = {}, set(), set()
    readonly = set()
    visited = Counter()
    arithmetic = payload = transfers = scratch_transfers = 0
    load_strides = {}
    store_stride = None
    block = blocks[0]
    incoming = ()

    def bound(pointer, width):
        if not isinstance(pointer, _Pointer) or pointer.offset < 0 or pointer.offset + width > capacity[pointer.root]:
            raise ValueError("input preparation pointer is unknown or outside bounded storage")

    def integer(value):
        if type(value) is not int:
            raise ValueError("loaded tensor value or pointer used as scalar arithmetic/control")
        return value

    while block is not None:
        visited[block] += 1
        if visited[block] > counts.get(block, 0):
            raise ValueError("input preparation traversal differs from independently proven CFG")
        if incoming:
            if len(incoming) != len(block.args):
                raise ValueError("input preparation block argument mismatch")
            values.update(zip(block.args, incoming, strict=True))
        next_block = None
        for op in block.ops:
            name = op.name
            args = [values[v] for v in op.operands]
            result = None
            if name == "llvm.mlir.constant":
                result = constant_integer(op.results[0])
                if result is None:
                    raise ValueError("input preparation contains noninteger constant")
            elif isinstance(op, AddressOfOp):
                storage, size = _constant_zero_storage(op, function)
                capacity[storage] = size
                readonly.add(storage)
                result = _Pointer(storage)
            elif name == "llvm.alloca":
                size = integer(args[0]) * _payload_bytes(op.properties["elem_type"])
                if size <= 0 or size > 1 << 20:
                    raise ValueError("input scratch allocation exceeds bounded proof")
                capacity[op] = size
                result = _Pointer(op)
            elif name == "llvm.getelementptr":
                pointer = args[0]
                indices = list(op.properties["rawConstantIndices"].get_values())
                if len(indices) != 1 or op.properties["elem_type"] != IntegerType(8):
                    raise ValueError("input preparation has unsupported pointer layout")
                offset = integer(args[1]) if indices[0] == -(1 << 31) else indices[0]
                if not isinstance(pointer, _Pointer):
                    raise ValueError("input preparation has unknown pointer ancestry")
                result = _Pointer(pointer.root, pointer.offset + offset)
            elif name == "llvm.ptrtoint":
                if not isinstance(args[0], _Pointer) or op.results[0].type != IntegerType(64):
                    raise ValueError("input preparation pointer conversion is unbound or truncates register width")
                result = args[0]
            elif name in _INTEGER:
                left, right = args
                if name == "llvm.add" and isinstance(left, _Pointer) and type(right) is int:
                    _unpoisoned_pointer_add(op)
                    result = _Pointer(left.root, left.offset + right)
                else:
                    integer(left)
                    integer(right)
                    result = integer_operation(op, args)
                    if result is None:
                        raise ValueError("input preparation has unsupported integer semantics")
                arithmetic += 1
            elif name == "llvm.icmp":
                for value in args:
                    integer(value)
                result = integer_operation(op, args)
            elif name == "llvm.select":
                result = args[1] if integer(args[0]) else args[2]
            elif name == "llvm.load":
                pointer = args[0]
                width = _payload_bytes(op.results[0].type)
                bound(pointer, width)
                if pointer.root == 0:
                    result = _Copy(tuple((0, pointer.offset + i) for i in range(width)))
                elif pointer.root in {1, 2}:
                    raise ValueError("input preparation reads weights or output on CPU")
                elif pointer.root in readonly:
                    result = _Copy(("zero",) * width)
                else:
                    result = _Copy(tuple(memory[(pointer.root, pointer.offset + i)] for i in range(width)))
                payload += width
            elif name == "llvm.store":
                value, pointer = args
                width = _payload_bytes(op.operands[0].type)
                bound(pointer, width)
                if pointer.root in {0, 1, 2}:
                    raise ValueError("input preparation writes model input, weight or output on CPU")
                if pointer.root in readonly:
                    raise ValueError("input preparation writes read-only zero padding storage")
                origins = value.origins if isinstance(value, _Copy) else ("zero",) * width if value == 0 else None
                if origins is None or len(origins) != width:
                    raise ValueError("input scratch store is not an unchanged input copy or zero padding")
                for i, origin in enumerate(origins):
                    key = pointer.root, pointer.offset + i
                    memory[key] = origin
                    initialized.add(key)
                payload += width
            elif name == "llvm.inline_asm":
                if (
                    op.results
                    or not isinstance(op.has_side_effects, UnitAttr)
                    or "~{memory}" not in op.constraints.data.split(",")
                ):
                    raise ValueError("input preparation command has unclosed effects")
                template = _asm_template(op)
                if template == "fence" and args:
                    raise ValueError("input preparation pointer/scalar escapes opaque fence")
                if template != "fence":
                    parsed = _parse_insn(template, isa["CUSTOM_OPCODE"])
                    if parsed is None or not parsed[1] or len(args) != 2:
                        raise ValueError("input preparation has unknown command")
                    operands = [
                        {"kind": "argbase", "raw": None, "arg_index": a.root, "offset": a.offset}
                        if isinstance(a, _Pointer)
                        else {"kind": "const", "raw": integer(a)}
                        for a in args
                    ]
                    klass, decoded = semantics.decode_instruction(parsed[0], *operands, isa)
                    if klass in {"MVIN2", "MVIN3"} and load_states is None:
                        raise ValueError("input preparation does not prove independent secondary load-port state")
                    pointer_args = [a for a in args if isinstance(a, _Pointer)]
                    if klass in {"MVIN", "MVOUT"} and not pointer_args:
                        raise ValueError("DRAM transfer has no bound declared pointer")
                    if klass == "UNKNOWN":
                        raise ValueError("input preparation contains undecoded command")
                    if pointer_args:
                        pointer = args[0]
                        if (
                            not isinstance(pointer, _Pointer)
                            or len(pointer_args) != 1
                            or decoded.get("dram") != operands[0]
                        ):
                            raise ValueError("pointer command lacks decoded DRAM operand role")
                        if klass in {"MVIN", "MVIN2", "MVIN3"}:
                            if load_states is not None and decoded.get("is_acc_addr") is not False:
                                raise ValueError("input transfer byte width has no selected scratchpad address proof")
                            port = load_states.port(klass) if load_states is not None else 0
                            if port is None:
                                raise ValueError("input transfer has no source-selected load port")
                            load_stride = load_strides.get(port)
                            if pointer.root == 2:
                                raise ValueError("CPU output used as accelerator input")
                            rows, cols = decoded.get("rows"), decoded.get("cols")
                            if type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0:
                                raise ValueError("input transfer extent is unknown")
                            if type(load_stride) is not int or load_stride < 0:
                                raise ValueError("input transfer stride is unknown")
                            bound(pointer, (rows - 1) * load_stride + cols)
                            if pointer.root not in {0, 1}:
                                scratch_transfers += 1
                                for row_index in range(rows):
                                    for column in range(cols):
                                        key = pointer.root, pointer.offset + row_index * load_stride + column
                                        if pointer.root not in readonly and key not in memory:
                                            raise ValueError("accelerator consumes uninitialized input scratch")
                                        consumed.add(key)
                            if pointer.root != 1:
                                transfers += 1
                        elif klass == "MVOUT" and pointer.root == 2:
                            rows, cols = decoded.get("rows"), decoded.get("cols")
                            if (
                                type(rows) is not int
                                or type(cols) is not int
                                or rows <= 0
                                or cols <= 0
                                or type(store_stride) is not int
                                or store_stride < 0
                                or decoded.get("readout") != "i32"
                            ):
                                raise ValueError("output transfer extent/stride/type is unknown")
                            bound(pointer, (rows - 1) * store_stride + cols * 4)
                        else:
                            raise ValueError("input scratch/output pointer escapes decoded transfer role")
                    elif klass == "CONFIG_LD":
                        state = decoded.get("state_id")
                        if type(state) is not int or state not in (load_states.states if load_states else (0,)):
                            raise ValueError("input preparation does not prove nonzero load-port state")
                        stride = decoded.get("stride")
                        if (type(stride) is not int or stride < 0
                                or (load_states is not None and stride >= 1 << load_states.stride_width)):
                            raise ValueError("input load stride differs from its actual source storage width")
                        load_strides[state] = stride
                    elif klass == "CONFIG_ST":
                        store_stride = decoded.get("out_stride_bytes")
            elif name in {"llvm.br", "llvm.cond_br"}:
                if name == "llvm.br":
                    next_block, incoming = op.successors[0], tuple(args)
                else:
                    sizes = list(op.properties["operandSegmentSizes"].get_values())
                    taken = integer(args[0]) != 0
                    next_block = op.successors[0 if taken else 1]
                    incoming = tuple(args[1 : 1 + sizes[1]] if taken else args[1 + sizes[1] :])
            elif name == "llvm.return":
                if args:
                    raise ValueError("input preparation returns a scalar or pointer")
            else:
                raise ValueError(f"input preparation contains unsupported operation {name}")
            if op.results:
                if len(op.results) != 1 or result is None:
                    raise ValueError("input preparation result has unknown provenance")
                values[op.results[0]] = result
        block = next_block
    if visited != Counter({b: n for b, n in counts.items() if n}) or not transfers:
        raise ValueError("input preparation execution/transfer roster is incomplete")
    # Padding scratch may feed only copied zero bytes; every nonzero input-copy
    # destination must itself have appeared as actual decoded accelerator input.
    if any(key not in consumed and memory[key] != "zero" for key in initialized):
        raise ValueError("input copy scratch never feeds decoded accelerator input")
    return {
        "status": "proven",
        "arithmetic": arithmetic,
        "payload_bytes": payload,
        "input_transfers": transfers,
        "scratch_input_transfers": scratch_transfers,
        "readonly_zero_globals": len(readonly),
        "scope": "bounded input address/copy only; no tensor values evaluated",
    }
