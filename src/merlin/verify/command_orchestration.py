"""Prove that a pointer-only kernel's host work is only target-command orchestration.

The host-compute audit charges host arithmetic against the elements a group produces. A kernel
that only walks tiles can still spend many integer instructions per element on transfer addresses
and command operands without touching a tensor value. This proof decides, structurally, that a
function cannot be computing on tensor values, so its integer work may leave the element budget:

* every argument is a pointer, nothing is returned and nothing is stored;
* every effect is a recognised target command (an ``.insn r`` of the target's custom opcode whose
  funct the target's ISA facts name) or a plain fence. Each register field is parsed: the
  destination is ``x0`` (no value comes back), each source is ``x0`` or one of the asm's own
  operands, every operand is consumed, and every operand constraint is a plain register input.
  Any other literal register is refused: the kernel never set it, so it could carry a host value;
* a scalar memory load is admitted only as a configuration PARAMETER: its address derives from a
  kernel argument, its value flows only through casts and bit packing into operands of the
  target's configuration commands (the class the ISA facts name), never across a block boundary,
  and the loads execute a bounded number of times per call, independent of the output size.

Anything else refuses with the construct that defeated it, so a veto can say why. A refusal never
makes a group clean or vetoed by itself: the caller falls back to the ordinary element budget.
"""

from __future__ import annotations

from typing import Any

#: Value-preserving or narrowing casts a loaded parameter may pass through on its way to a
#: command register (a float scale reinterpreted as its bits, or converted to an integer field).
_PARAMETER_CASTS = frozenset(
    {"llvm.zext", "llvm.sext", "llvm.trunc", "llvm.bitcast", "llvm.fptosi", "llvm.fptoui"}
)
#: Bit packing that places a parameter into its field of a command operand.
_PARAMETER_PACKING = frozenset({"llvm.and", "llvm.or", "llvm.xor", "llvm.shl", "llvm.lshr", "llvm.ashr"})
#: The RISC-V host's spellings of the hard-wired zero register.
_ZERO_REGISTER = frozenset({"x0", "zero"})


class _Refused(Exception):
    """The construct that defeats the proof, in words a finding can quote."""


def prove_command_orchestration(function: Any, activity: dict, *, target: str | None, parameter_loads: int) -> dict:
    """``{"status": "proven", ...}`` with the admitted parameter loads, or ``{"status": "refused", "reason"}``."""
    try:
        return _prove(function, activity, target=target, parameter_loads=parameter_loads)
    except _Refused as exc:
        return {"status": "refused", "reason": str(exc)[:300]}


def _orchestration_ops() -> frozenset[str]:
    from merlin.perf.host_cfg_activity import _INTEGER

    return _INTEGER | {
        "llvm.func", "llvm.mlir.constant", "llvm.mlir.addressof", "llvm.getelementptr",
        "llvm.ptrtoint", "llvm.inttoptr", "llvm.bitcast", "llvm.icmp", "llvm.select",
        "llvm.trunc", "llvm.zext", "llvm.sext", "llvm.br", "llvm.cond_br", "llvm.return",
        # Stack scratch carries no host value: stores are refused and loads must come from a
        # kernel argument, so only the target's own commands can touch it.
        "llvm.alloca",
    }


def _isa(target: str | None) -> dict:
    if target is None:
        raise _Refused("no target is selected, so its commands cannot be recognised")
    from merlin.targetgen.rocc.decode import isa_constants

    try:
        isa = isa_constants(target)
        if not isinstance(isa["CUSTOM_OPCODE"], int) or not isinstance(isa["FUNCT_CLASS"], dict):
            raise TypeError("ISA facts lack an integer custom opcode or funct class table")
    except Exception as exc:  # noqa: BLE001 -- facts that cannot be read prove nothing: refuse
        raise _Refused(f"the target's ISA facts are unavailable ({type(exc).__name__})") from None
    return isa


def _int_field(text: str) -> int | None:
    try:
        return int(text, 0)
    except ValueError:
        return None


def _is_fence(template: str) -> bool:
    """A host memory fence, bare or with predecessor/successor sets (``fence iorw, iorw``).

    A fence orders memory and moves no value, so it is orchestration in any spelling the host
    ISA defines; anything else spelled ``fence`` is not recognised.
    """
    parts = template.split(None, 1)
    if not parts or parts[0] != "fence":
        return False
    if len(parts) == 1:
        return True
    sets = parts[1]
    fields = [field.strip() for field in sets.split(",")]
    return len(fields) == 2 and all(field and set(field) <= set("iorw") for field in fields)


def _command(op: Any, isa: dict) -> str | None:
    """The command class of one ``llvm.inline_asm``; ``None`` for a fence. Refuses anything else."""
    from xdsl.dialects.builtin import IntegerType, UnitAttr
    from xdsl.dialects.llvm import LLVMPointerType, LLVMVoidType

    from merlin.targetgen.rocc.decode import _asm_template

    template = _asm_template(op)
    void_result = len(op.results) == 1 and isinstance(op.results[0].type, LLVMVoidType) and not op.results[0].uses
    if op.results and not void_result:
        raise _Refused(f"inline asm {template[:60]!r} returns a value to the host")
    if not isinstance(op.has_side_effects, UnitAttr):
        raise _Refused(f"inline asm {template[:60]!r} is not marked as having side effects")
    constraints = [entry.strip() for entry in op.constraints.data.split(",")] if op.constraints.data else []
    inputs, clobbers = constraints[: len(op.operands)], constraints[len(op.operands) :]
    if inputs != ["r"] * len(op.operands) or not all(c.startswith("~{") and c.endswith("}") for c in clobbers):
        raise _Refused(f"inline asm constraints {op.constraints.data!r} are not plain register inputs and clobbers")
    if "~{memory}" not in clobbers:
        raise _Refused(f"inline asm {template[:60]!r} does not clobber memory, so its ordering is unproven")
    if any(not isinstance(v.type, (IntegerType, LLVMPointerType)) for v in op.operands):
        raise _Refused(f"inline asm {template[:60]!r} takes a non-integer register operand")
    if _is_fence(template):
        if op.operands:
            raise _Refused("a fence takes operands")
        return None
    fields = [field.strip() for field in template.split(",")]
    head = fields[0].split()
    if len(fields) != 6 or len(head) != 3 or head[:2] != [".insn", "r"]:
        raise _Refused(f"inline asm {template[:60]!r} is not one six-field '.insn r' command")
    opcode, funct3, funct = _int_field(head[2]), _int_field(fields[1]), _int_field(fields[2])
    if opcode is None or funct3 is None or funct is None:
        raise _Refused(f"command {template[:60]!r} has a non-integer opcode or funct field")
    if opcode != isa["CUSTOM_OPCODE"]:
        raise _Refused(f"inline asm {template[:60]!r} is not the target's custom opcode")
    label = isa["FUNCT_CLASS"].get(funct)
    if label is None:
        raise _Refused(f"funct {funct} in {template[:60]!r} is not a command the target defines")
    rd, sources = fields[3], fields[4:]
    if rd not in _ZERO_REGISTER:
        raise _Refused(f"command {template[:60]!r} writes register {rd!r} back to the host")
    used = set()
    for source in sources:
        if source in _ZERO_REGISTER:
            continue
        index = _int_field(source[1:]) if source.startswith("$") and source[1:].isdigit() else None
        if index is None:
            raise _Refused(
                f"command {template[:60]!r} reads literal register {source!r}, which the kernel never set "
                "and which could carry a host value"
            )
        if index >= len(op.operands):
            raise _Refused(f"command {template[:60]!r} names operand {source!r} the asm does not have")
        used.add(index)
    if used != set(range(len(op.operands))):
        raise _Refused(f"command {template[:60]!r} leaves asm operands unconsumed")
    return str(label)


def _argument_root(pointer: Any, entry: Any) -> bool:
    from xdsl.ir import BlockArgument

    while not isinstance(pointer, BlockArgument):
        owner = pointer.owner
        if getattr(owner, "name", None) != "llvm.getelementptr":
            return False
        pointer = owner.operands[0]
    return pointer.block is entry


def _parameter_slice(load: Any, commands: dict, config_class: Any, entry: Any) -> set:
    """The casts and packing between a parameter load and the configuration commands it feeds."""
    from xdsl.dialects.builtin import IntegerType, VectorType
    from xdsl.dialects.llvm import LLVMPointerType

    from merlin.perf.host_cfg_activity import _payload_bytes, _real_name

    value = load.results[0]
    if (
        isinstance(value.type, (LLVMPointerType, VectorType))
        or _payload_bytes(value.type) is None
        or set(load.properties) - {"ordering", "alignment"}
        or load.attributes
        or getattr(getattr(load.properties.get("ordering"), "value", None), "data", 0) != 0
    ):
        raise _Refused("a memory load that is not one plain scalar read")
    if not _argument_root(load.operands[0], entry):
        raise _Refused("a memory load whose address does not derive from a kernel argument")
    if not value.uses:
        raise _Refused("a memory load whose value reaches no command")
    sliced, queue = set(), [value]
    while queue:
        current = queue.pop()
        for use in current.uses:
            op, name = use.operation, _real_name(use.operation)
            if name in _PARAMETER_CASTS | _PARAMETER_PACKING:
                if not all(isinstance(r.type, IntegerType) or _payload_bytes(r.type) for r in op.results):
                    raise _Refused(f"memory load feeding {name} with a non-scalar result")
                if op not in sliced:
                    sliced.add(op)
                    queue.extend(op.results)
            elif name == "llvm.inline_asm" and op in commands:
                if config_class is None or commands[op] != config_class:
                    raise _Refused(
                        f"memory load feeding a {commands[op] or 'fence'} operand; only configuration "
                        "commands take loaded parameters"
                    )
            else:
                raise _Refused(f"unproven memory load feeding {name}")
    return sliced


def _prove(function: Any, activity: dict, *, target: str | None, parameter_loads: int) -> dict:
    from xdsl.dialects.llvm import LLVMPointerType

    from merlin.perf.host_cfg_activity import _payload_bytes, _real_name

    isa = _isa(target)
    blocks = tuple(function.body.blocks)
    if not blocks or not blocks[0].args:
        raise _Refused("the kernel has no pointer arguments")
    for index, arg in enumerate(blocks[0].args):
        if not isinstance(arg.type, LLVMPointerType):
            raise _Refused(f"argument {index} is a {arg.type} value, not a pointer")
    allowed = _orchestration_ops()
    commands: dict[Any, str | None] = {}
    loads = []
    for op in function.walk():
        name = _real_name(op)
        if name == "llvm.inline_asm":
            commands[op] = _command(op, isa)
        elif name == "llvm.load":
            loads.append(op)
        elif name == "llvm.store":
            raise _Refused("a host memory store (llvm.store)")
        elif name == "llvm.return" and op.operands:
            raise _Refused("the kernel returns a host value")
    if not any(label is not None for label in commands.values()):
        raise _Refused("no target command is issued")
    sliced: set = set()
    for load in loads:
        sliced |= _parameter_slice(load, commands, isa.get("CONFIG_CLASS"), blocks[0])
    for op in function.walk():
        name = _real_name(op)
        if name not in allowed and name not in ("llvm.inline_asm", "llvm.load") and op not in sliced:
            raise _Refused(f"host operation {name} is not command orchestration")
    counts = {}
    for row in activity.get("blocks") or ():
        if type(row.get("block")) is int and 0 <= row["block"] < len(blocks):
            counts[blocks[row["block"]]] = row.get("execution_count")
    executions = 0
    payload = 0
    for load in loads:
        count = counts.get(load.parent_block())
        if type(count) is not int or count < 0:
            raise _Refused("a parameter load has no derived execution count")
        executions += count
        payload += count * _payload_bytes(load.results[0].type)
    if executions > parameter_loads:
        raise _Refused(
            f"{executions} parameter loads per call exceed the bound of {parameter_loads}: a load "
            "repeated with the output is payload, not configuration"
        )
    return {
        "status": "proven",
        "commands": sum(1 for label in commands.values() if label is not None),
        "parameter_loads": executions,
        "parameter_payload_bytes": payload,
    }
