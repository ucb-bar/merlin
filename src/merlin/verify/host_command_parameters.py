"""Bounded proof that immutable scalar bits only pack selected device configurations.

This is not a tensor evaluator. A closed SSA slice may contain integer casts and
bit packing, and must terminate exclusively in an explicitly selected operand
observer's configuration commands. All other loads and arithmetic stay charged.
"""

from collections import Counter

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType, ModuleOp, UnitAttr
from xdsl.dialects.llvm import AddressOfOp, GlobalOp, LLVMArrayType, LLVMPointerType, LLVMVoidType

from merlin.perf.constant_control_cfg import integer_operation
from merlin.perf.host_cfg_activity import _payload_bytes
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer

_CASTS = {"llvm.zext", "llvm.sext", "llvm.trunc", "llvm.bitcast"}
_PACKING = {"llvm.and", "llvm.or", "llvm.xor", "llvm.shl", "llvm.lshr", "llvm.ashr"}


def _packed_mask(op, masks):
    """Follow only bits originating at an immutable load, never scalar values."""
    mask = masks.get(op.operands[0], 0)
    width = op.results[0].type.width.data
    limit = (1 << width) - 1
    source_width = op.operands[0].type.width.data
    if op.name in _CASTS:
        if op.name == "llvm.sext" and mask & (1 << (source_width - 1)):
            mask |= limit ^ ((1 << source_width) - 1)
        return mask & limit
    if op.name in {"llvm.shl", "llvm.lshr", "llvm.ashr"}:
        shift = constant_integer(op.operands[1])
        if type(shift) is not int or not 0 <= shift < source_width:
            raise ValueError("command parameter shift has unproved register semantics")
        if op.name == "llvm.shl":
            return (mask << shift) & limit
        if op.name == "llvm.ashr" and mask & (1 << (source_width - 1)):
            mask |= limit << source_width
        return (mask >> shift) & limit
    constant = constant_integer(op.operands[1])
    if constant is None:
        constant, mask = constant_integer(op.operands[0]), masks.get(op.operands[1], 0)
    if op.name == "llvm.and":
        mask &= constant
    elif op.name == "llvm.or":
        mask &= ~constant
    return mask & limit


def _configuration_fields(isa, label, masks):
    """Every loaded bit must project into an actual selected configuration field."""
    fields = []
    selector = isa["CONFIG_SELECTOR"]
    selected_mask = ((1 << selector["width"]) - 1) << selector["offset"]
    if masks[0 if selector["operand"] == "rs1" else 1] & selected_mask:
        raise ValueError("loaded command parameters may not change the selected configuration subtype")
    for slot, mask in zip(("rs1", "rs2"), masks, strict=True):
        covered = 0
        for field in isa["OPERAND_FIELDS"][label]["fields"]:
            if field.get("operand") != slot:
                continue
            if field["as"] in {"raw", "operand"}:
                bits = -1
            elif type(field.get("offset")) is int and type(field.get("width")) is int:
                bits = ((1 << field["width"]) - 1) << field["offset"]
            else:
                continue
            covered |= bits
            if bits & mask:
                fields.append(label + "." + field["name"])
        if mask & ~covered:
            raise ValueError("loaded command parameter bits have no selected source operand field")
    if not any(masks):
        raise ValueError("loaded command parameter bits do not reach a selected operand field")
    return fields


def _storage(address, function):
    module = function.parent_op()
    if not isinstance(module, ModuleOp) or function.parent_block() is not module.body.block:
        raise ValueError("command parameter has no original enclosing module")
    if any(
        (op.op_name.data if op.name == "builtin.unregistered" else op.name)
        not in {"llvm.mlir.global", "llvm.func", "llvm.module_flags"}
        for op in module.body.block.ops
    ):
        raise ValueError("command parameter module has unproved aliases or storage effects")
    reference = address.global_name
    if reference.nested_references.data:
        raise ValueError("command parameter has nested symbol provenance")
    name = reference.root_reference.data
    definitions = [op for op in module.body.block.ops if getattr(op.properties.get("sym_name"), "data", None) == name]
    if len(definitions) != 1 or not isinstance(storage := definitions[0], GlobalOp):
        raise ValueError("command parameter storage is missing or ambiguous")
    typ, initializer = storage.global_type, storage.value
    if (
        storage.linkage.linkage.data not in {"private", "internal"}
        or not isinstance(storage.constant, UnitAttr)
        or storage.body.blocks
        or storage.thread_local_ is not None
        or storage.section is not None
        or storage.attributes
        or storage.addr_space.value.data != 0
        or not isinstance(address.result.type, LLVMPointerType)
        or getattr(address.result.type.addr_space, "data", 0) != 0
        or not isinstance(typ, LLVMArrayType)
        or not isinstance(typ.type, IntegerType)
        or typ.type.width.data not in {8, 16, 32, 64}
        or not 0 < typ.size.data * _payload_bytes(typ.type) <= 1 << 20
        or not isinstance(initializer, DenseIntOrFPElementsAttr)
        or initializer.get_element_type() != typ.type
        or initializer.get_shape() != (typ.size.data,)
        or len(initializer.data.data) != typ.size.data * _payload_bytes(typ.type)
    ):
        raise ValueError("command parameter requires bounded immutable private integer storage")
    for op in module.walk():
        if isinstance(op, AddressOfOp) and op.global_name.root_reference.data == name:
            if op.global_name.nested_references.data or op.parent_op() is not function:
                raise ValueError("command parameter address escapes the proven function")
    width = typ.type.width.data
    return storage, tuple(int(v) & ((1 << width) - 1) for v in initializer.get_values())


def _slice(address, storage, observer, isa):
    from merlin.targetgen.rocc.decode import _asm_template, _parse_insn

    pointers, loads, packing, endpoints = {address.result}, set(), set(), set()
    queue = [address.result]
    while queue:
        value = queue.pop()
        for use in value.uses:
            op = use.operation
            if op.name == "llvm.getelementptr" and use.index == 0:
                if op.properties["elem_type"] != storage.global_type.type:
                    raise ValueError("command parameter pointer changes its integer element layout")
                if op.results[0] not in pointers:
                    pointers.add(op.results[0])
                    queue.append(op.results[0])
            elif op.name == "llvm.load" and use.index == 0 and op.results[0].type == storage.global_type.type:
                # Volatile/atomic or extra operands are not a closed immutable read.
                if len(op.operands) != 1 or op.attributes or op.ordering.value.data != 0:
                    raise ValueError("command parameter load has unproved memory semantics")
                loads.add(op)
            else:
                raise ValueError("command parameter global pointer has a non-load use")
    values = {op.results[0] for op in loads}
    queue = list(values)
    while queue:
        value = queue.pop()
        if not value.uses:
            raise ValueError("command parameter value has no selected configuration use")
        for use in value.uses:
            op = use.operation
            if op.name in _CASTS | _PACKING:
                if not all(
                    isinstance(v.type, IntegerType) and 0 < v.type.width.data <= observer.word_bits
                    for v in (*op.operands, *op.results)
                ):
                    raise ValueError("command parameter packing changes its integer register domain")
                if op.name in _PACKING:
                    constants = [constant_integer(operand) for operand in op.operands]
                    if (
                        len(op.operands) != 2
                        or all(v is None for v in constants)
                        or (op.name in {"llvm.shl", "llvm.lshr", "llvm.ashr"} and constants[1] is None)
                    ):
                        raise ValueError("command parameter packing mixes nonconstant scalar values")
                packing.add(op)
                if op.results[0] not in values:
                    values.add(op.results[0])
                    queue.append(op.results[0])
            elif op.name == "llvm.inline_asm":
                parsed = _parse_insn(_asm_template(op), isa["CUSTOM_OPCODE"])
                fields = [field.strip() for field in _asm_template(op).split(",")]
                void = len(op.results) == 1 and isinstance(op.results[0].type, LLVMVoidType) and not op.results[0].uses
                if (
                    parsed is None
                    or not parsed[1]
                    or len(op.operands) != 2
                    or len(fields) != 6
                    or fields[3:] != ["x0", "$0", "$1"]
                    or int(fields[1], 0) != isa["FUNCT3"]
                    or isa["FUNCT_CLASS"].get(parsed[0]) != isa.get("CONFIG_CLASS")
                    or not isa.get("CONFIG_CLASS")
                    or not isa.get("CONFIG_SELECTOR")
                    or not isinstance(op.has_side_effects, UnitAttr)
                    or op.constraints.data != "r,r,~{memory}"
                    or (op.results and not void)
                ):
                    raise ValueError("immutable scalar value does not end in a selected configuration command")
                endpoints.add(op)
            else:
                raise ValueError("immutable scalar value is used as tensor arithmetic or other host work")
    if not loads or not endpoints:
        raise ValueError("command parameter has no closed configuration slice")
    return pointers, loads, packing, endpoints


def prove_command_parameters(function, activity, *, target):
    """Return exact exclusions, or a refusal that leaves all original charges intact."""
    try:
        return _prove(function, activity, target=target)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, ZeroDivisionError) as exc:
        return {"status": "unproven", "detail": str(exc)[:300]}


def _prove(function, activity, *, target):
    from merlin.targetgen.rocc.decode import _asm_template, _parse_insn, _semantics
    from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver

    observer = getattr(_semantics(target), "operand_observer", None) if target is not None else None
    if type(observer) is not BoundOperandObserver:
        raise ValueError("command parameters require the explicitly selected operand observer")
    isa = observer.isa_constants(target)
    blocks = tuple(function.body.blocks)
    if activity.get("status") != "derived" or not blocks:
        raise ValueError("command parameters require independently finite CFG counts")
    counts = {blocks[row["block"]]: row["execution_count"] for row in activity["blocks"]}
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("command parameters have unknown dynamic counts")
    # Bound diagnostic effort independently of model size; unsupported large CFGs
    # retain their existing audit rather than spawning tensor interpretation.
    if sum(counts[b] * len(tuple(b.ops)) for b in blocks) > 1000000:
        raise ValueError("command parameter scalar proof exceeds its operation bound")
    addresses = [op for op in function.walk() if isinstance(op, AddressOfOp)]
    if not addresses:
        raise ValueError("no immutable command parameter storage")
    if len(addresses) > 64:
        raise ValueError("command parameter storage roots exceed the finite proof bound")
    storages, pointers, loads, packing, endpoints = {}, set(), set(), set(), set()
    for address in addresses:
        storage, data = _storage(address, function)
        roots, reads, packed, commands = _slice(address, storage, observer, isa)
        storages[address] = (storage, data)
        pointers.update(roots)
        loads.update(reads)
        packing.update(packed)
        endpoints.update(commands)
    values = {arg: None for arg in blocks[0].args}
    masks = {}
    visited, executed = Counter(), Counter()
    block, incoming = blocks[0], ()
    classes, fields_used = set(), set()
    while block is not None:
        visited[block] += 1
        if visited[block] > counts.get(block, 0):
            raise ValueError("command parameter traversal differs from independent CFG counts")
        if block is not blocks[0] and len(incoming) != len(block.args):
            raise ValueError("command parameter block argument mismatch")
        if incoming:
            values.update(zip(block.args, incoming, strict=True))
        successor = None
        for op in block.ops:
            args = [values.get(v) for v in op.operands]
            result = None
            if op in storages:
                result = (op, 0)
            elif op.name == "llvm.getelementptr" and op.results[0] in pointers:
                indices = list(op.properties["rawConstantIndices"].get_values())
                offset = (
                    args[1]
                    if len(indices) == 1 and indices[0] == -(1 << 31)
                    else (indices[0] if len(indices) == 1 else None)
                )
                if not isinstance(args[0], tuple) or type(offset) is not int:
                    raise ValueError("command parameter pointer index is not independently constant-control")
                result = (args[0][0], args[0][1] + offset)
                if not 0 <= result[1] <= len(storages[result[0]][1]):
                    raise ValueError("command parameter pointer leaves its immutable storage")
            elif op in loads:
                address, index = args[0] if isinstance(args[0], tuple) else (None, None)
                data = storages[address][1] if address in storages else ()
                if type(index) is not int or not 0 <= index < len(data):
                    raise ValueError("command parameter load is outside its exact immutable initializer")
                result = data[index]
                masks[op.results[0]] = (1 << op.results[0].type.width.data) - 1
            elif op in endpoints:
                if not all(type(v) is int for v in args):
                    raise ValueError("configuration operands are not closed immutable scalar bits")
                funct = _parse_insn(_asm_template(op), isa["CUSTOM_OPCODE"])[0]
                label, fields = observer.decode_instruction(
                    funct, {"kind": "const", "raw": args[0]}, {"kind": "const", "raw": args[1]}, isa
                )
                if label not in isa["CONFIG_SUBTYPE"].values() or not fields:
                    raise ValueError("configuration selector or operand projection is unresolved")
                fields_used.update(_configuration_fields(isa, label, [masks.get(v, 0) for v in op.operands]))
                classes.add(label)
            elif op.name == "llvm.br":
                successor, incoming = op.successors[0], tuple(args)
            elif op.name == "llvm.cond_br":
                if type(args[0]) is not int:
                    raise ValueError("command parameter CFG depends on tensor values")
                sizes = list(op.properties["operandSegmentSizes"].get_values())
                successor = op.successors[0 if args[0] else 1]
                incoming = tuple(args[1 : 1 + sizes[1]] if args[0] else args[1 + sizes[1] :])
            elif len(op.results) == 1:
                result = integer_operation(op, args)
            if op in packing and result is None:
                raise ValueError("command parameter packing has unknown or poison integer semantics")
            if op in packing:
                masks[op.results[0]] = _packed_mask(op, masks)
            if op in loads or op in packing or op in endpoints:
                executed[op] += 1
            values.update((v, result if len(op.results) == 1 else None) for v in op.results)
        if block.last_op.name not in {"llvm.br", "llvm.cond_br", "llvm.return"}:
            raise ValueError("command parameter CFG has an unsupported terminator")
        block = successor
    if any(visited[b] != counts[b] for b in blocks):
        raise ValueError("command parameter execution differs from independent CFG accounting")
    return {
        "status": "proven",
        "basis": "immutable_global_selected_configuration_v1",
        "globals": sorted({storage.sym_name.data for storage, _ in storages.values()}),
        "configuration_classes": sorted(classes),
        "configuration_fields": sorted(fields_used),
        "selected_config_class": isa["CONFIG_CLASS"],
        "selected_config_selector": isa["CONFIG_SELECTOR"],
        "word_bits": observer.word_bits,
        "payload_bytes": sum(executed[op] * _payload_bytes(op.results[0].type) for op in loads),
        "arithmetic": sum(executed[op] for op in packing if op.name in _PACKING),
        "conversions": sum(executed[op] for op in packing if op.name in _CASTS),
        "configuration_commands": sum(executed[op] for op in endpoints),
    }
