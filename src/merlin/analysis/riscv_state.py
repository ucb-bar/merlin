"""Conservative RV64 register, private-frame, FP and summarized-call transfer.

Observations retain their exact supported domain and explicit unknowns.
No accelerator semantics, execution multiplicity or hardware law is inferred.
"""



from .riscv_decode import (
    _REGIONS as _REGIONS,
)
from .riscv_decode import (
    BRANCHES as BRANCHES,
)
from .riscv_decode import (
    CALLS as CALLS,
)
from .riscv_decode import (
    ET_REL as ET_REL,
)
from .riscv_decode import (
    CPUInstruction as CPUInstruction,
)
from .riscv_decode import (
    _Cfg as _Cfg,
)
from .riscv_decode import (
    _ends_segment as _ends_segment,
)
from .riscv_decode import (
    decode_cpu as decode_cpu,
)
from .riscv_decode import (
    linked_input_status as linked_input_status,
)
from .riscv_decode import (
    region_index as region_index,
)
from .riscv_decode import (
    section_of as section_of,
)
from .riscv_decode import (
    signed as signed,
)
from .riscv_stream import (
    _MAX_PERIOD as _MAX_PERIOD,
)
from .riscv_stream import (
    _advance as _advance,
)
from .riscv_stream import (
    _identity as _identity,
)
from .riscv_stream import (
    _run_step as _run_step,
)
from .riscv_stream import (
    _slot as _slot,
)
from .riscv_stream import (
    _StreamRecorder as _StreamRecorder,
)
from .riscv_stream import (
    expand_command_stream as expand_command_stream,
)
from .riscv_values import (
    _VALUE_SET_LIMIT as _VALUE_SET_LIMIT,
)
from .riscv_values import (
    MASK as MASK,
)
from .riscv_values import (
    UNKNOWN as UNKNOWN,
)
from .riscv_values import (
    ZERO as ZERO,
)
from .riscv_values import (
    Bits as Bits,
)
from .riscv_values import (
    _affine_record as _affine_record,
)
from .riscv_values import (
    _cut_window as _cut_window,
)
from .riscv_values import (
    _sext as _sext,
)
from .riscv_values import (
    _Table as _Table,
)
from .riscv_values import (
    _window_arithmetic as _window_arithmetic,
)
from .riscv_values import (
    _window_record as _window_record,
)
from .riscv_values import (
    arithmetic as arithmetic,
)

# Stack-model key present while the region's own frame is private: no value
# derived from the entry stack pointer has left the tracked registers and
# frame slots. Joins intersect keys, so privacy survives a join only when every
# incoming path is private. A private frame lies below the entry stack pointer,
# which no other code owned at region entry, so only an address this region
# hands out can reach it. Writes by code outside this view (a word the base ISA
# does not define, a store through an untracked pointer) therefore keep slots
# below the entry stack pointer unless the frame has escaped. Calls still drop
# every slot, since the psABI lets a callee write its stack arguments. A
# dangling pointer kept from before region entry is outside this model.
_PRIVATE_FRAME = None
_FRAME = "entry_sp"


def _forget_unprivate(stack):
    """Forget slots that other code may address: all of them once escaped."""
    if _PRIVATE_FRAME not in stack:
        stack.clear()
        return
    for key in [key for key in stack if key is not _PRIVATE_FRAME]:
        if key[1] + key[2] // 8 > 0:
            del stack[key]


def _frame_handed_out(registers, *operands):
    """True when a frame address is visible to code outside this CPU view."""
    return any(registers[r].base == _FRAME for r in range(1, 32) if r != 2 or r in operands)


def _coarse_effect(word, d):
    """Refine words decode_cpu keeps coarse, using only base RV64GC encodings.

    Returns ("none", int_sources), ("store", base, imm, bits) or
    ("foreign", operands). "foreign" covers every word whose effects the base
    ISA does not define (custom or reserved opcodes, atomics, CSRs, vectors).
    """
    if not word:
        return ("foreign", (2,))
    w = int(word, 16)
    if len(word) == 4:
        q, f = w & 3, w >> 13
        if d.kind == "fp_memory" and f in (1, 5):
            if f == 1:
                return ("none", ())  # c.fld / c.fldsp write only an FP register
            if q == 0:
                return ("store", 8 + ((w >> 7) & 7), ((w >> 5) & 3) << 6 | ((w >> 10) & 7) << 3, 64)
            return ("store", 2, ((w >> 7) & 7) << 6 | ((w >> 10) & 7) << 3, 64)
        return ("foreign", (2,))
    opcode, funct3, rs1 = w & 127, (w >> 12) & 7, (w >> 15) & 31
    if d.kind == "fp":
        # Integer-to-FP converts (funct7 0b11010xx) and moves (0b11110xx) read
        # integer rs1; every other FP computation reads FP registers only.
        return ("none", (rs1,) if opcode == 0x53 and w >> 27 in (0x1A, 0x1E) else ())
    if opcode == 0x07:
        return ("none", ())  # LOAD-FP (scalar or vector) writes no memory
    if opcode == 0x27 and 1 <= funct3 <= 4:
        return ("store", rs1, signed((w >> 25) << 5 | (w >> 7) & 31, 12), 8 << funct3)
    if opcode == 0x53:
        return ("none", ())  # OP-FP compare/convert/move to an integer register
    if opcode == 0x73 and funct3 & 3 in (2, 3) and rs1 == 0:
        # Zicsr set/clear with an x0 or zero-immediate source never writes the CSR:
        # a pure read into rd, with no memory or control effect.
        return ("none", ())
    return ("foreign", (rs1, (w >> 20) & 31))


# Absolute-address memory slots exist only when a caller seeds them through
# ``entry_state`` (asserting ordinary initialized memory, e.g. ``.data``). An
# untracked exact address may be a device register, so a store to it never
# creates a slot; a store through an unknown pointer forgets every slot.
_ABSOLUTE = "abs"


def _store(stack, address, bits, value):
    if value.base == _FRAME:
        stack.pop(_PRIVATE_FRAME, None)
    if address.base != _FRAME:
        exact = address.exact
        kept = {}
        if exact is not None:
            lo, hi = exact, exact + bits // 8
            kept = {
                key: slot
                for key, slot in stack.items()
                if key is not _PRIVATE_FRAME and key[0] == _ABSOLUTE and not (lo < key[1] + key[2] // 8 and key[1] < hi)
            }
            tracked = (_ABSOLUTE, exact, bits) in stack
        _forget_unprivate(stack)
        stack.update(kept)
        if exact is not None and tracked:
            stack[(_ABSOLUTE, exact, bits)] = value
        return
    lo, hi = address.offset, address.offset + bits // 8
    for old in [key for key in stack if key is not _PRIVATE_FRAME]:
        if old[0] == _FRAME and lo < old[1] + old[2] // 8 and old[1] < hi:
            del stack[old]
    if len(stack) < 256:
        stack[(address.base, address.offset, bits)] = value


_INTEGER_MAJOR_OPCODES = frozenset((0x03, 0x0F, 0x13, 0x17, 0x1B, 0x23, 0x33, 0x37, 0x3B, 0x63, 0x67, 0x6F))
_CALLER_SAVED = (1, 5, 6, 7, *range(10, 18), *range(28, 32))
_FP_CALLER_SAVED = (*range(0, 8), *range(10, 18), *range(28, 32))
_NAN_BOX = Bits.const(MASK ^ ((1 << 32) - 1))


def _fp_effect(word, d, registers, fp):
    """Apply a word's FP-register effect; return an exact FP-to-integer bit copy.

    ``fmv.x.w``/``fmv.x.d`` copy FP bits to an integer register and
    ``fmv.w.x``/``fmv.d.x`` copy integer bits into an FP register (single
    precision NaN-boxed). Every other word that may write an FP register makes
    it unknown; FP loads stay unknown because the rows carry no loaded bytes.
    """
    if d.kind == "opaque":
        fp[:] = [UNKNOWN] * 32
        return None
    if not word:
        return None
    w = int(word, 16)
    if len(word) == 4:
        q, f = w & 3, w >> 13
        if f == 1 and q == 0:
            fp[8 + ((w >> 2) & 7)] = UNKNOWN
        elif f == 1 and q == 2:
            fp[(w >> 7) & 31] = UNKNOWN
        return None
    opcode, rd, f3, r1, r2, f7 = w & 127, (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31, (w >> 20) & 31, w >> 25
    if opcode in _INTEGER_MAJOR_OPCODES or opcode == 0x27:
        return None
    if opcode == 0x53 and f3 == 0 and r2 == 0 and f7 in (0x70, 0x71):
        return _sext(fp[r1], 32) if f7 == 0x70 else fp[r1]
    if opcode == 0x53 and f3 == 0 and r2 == 0 and f7 in (0x78, 0x79):
        source = registers[r1]
        fp[rd] = (
            arithmetic("or", arithmetic("and", source, Bits.const((1 << 32) - 1)), _NAN_BOX) if f7 == 0x78 else source
        )
        return None
    if opcode == 0x53 and f7 in (0x50, 0x51, 0x60, 0x61, 0x70, 0x71):
        return None  # writes an integer register only
    fp[rd] = UNKNOWN
    return None


def _call_target(d, registers, pc):
    if d.kind == "call":
        return d.target
    base = registers[d.rs1].exact
    return None if base is None else (base + d.imm) & MASK & ~1


def _register_index(name):
    """ABI or architectural register name -> ("x"|"f", index)."""
    abi = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1"]
    abi += [f"a{i}" for i in range(8)] + [f"s{i}" for i in range(2, 12)] + [f"t{i}" for i in range(3, 7)]
    fabi = [f"ft{i}" for i in range(8)] + ["fs0", "fs1"] + [f"fa{i}" for i in range(8)]
    fabi += [f"fs{i}" for i in range(2, 12)] + [f"ft{i}" for i in range(8, 12)]
    if isinstance(name, int):
        return "x", name
    if name == "fp":
        return "x", 8
    if name in abi:
        return "x", abi.index(name)
    if name in fabi:
        return "f", fabi.index(name)
    head, digits = name[:1], name[1:]
    if head in ("x", "f") and digits.isdigit() and int(digits) < 32:
        return head, int(digits)
    raise ValueError(f"unknown register name {name!r}")


def _seed(value):
    return value if isinstance(value, Bits) else Bits.const(int(value))


def _initial_state(entry_state):
    registers = [ZERO] + [UNKNOWN] * 31
    registers[2] = Bits(base=_FRAME)
    for i in range(10, 18):
        registers[i] = Bits(base=f"arg{i - 10}")
    fp, stack = [UNKNOWN] * 32, {_PRIVATE_FRAME: ZERO}
    for name, value in (entry_state or {}).items():
        if isinstance(name, tuple):
            kind, address, width = name
            if kind != "memory" or width not in (8, 16, 32, 64):
                raise ValueError(f"unsupported entry_state memory key {name!r}")
            stack[(_ABSOLUTE, address & MASK, width)] = _seed(value)
            continue
        kind, index = _register_index(name)
        if kind == "x" and index in (0, 2):
            raise ValueError("entry_state cannot seed zero or the entry stack pointer")
        (registers if kind == "x" else fp)[index] = _seed(value)
    return registers, stack, fp


def _summary_applies(summary, registers, stack):
    """True when the caller's state implies every value the summary assumed."""
    if not summary or summary.get("status") != "proved":
        return False
    for name, value in summary.get("assumptions", ()):
        if isinstance(name, tuple):
            held = stack.get((_ABSOLUTE, name[1] & MASK, name[2]))
        else:
            kind, index = _register_index(name)
            if kind != "x":
                return False
            held = registers[index]
        if held is None or held.exact is None or held.exact != value & MASK:
            return False
    return True


def _apply_summary(summary, registers, stack, fp):
    for r in _CALLER_SAVED:
        registers[r] = UNKNOWN
    for r in _FP_CALLER_SAVED:
        fp[r] = UNKNOWN
    if summary.get("writes_memory", True):
        private = _PRIVATE_FRAME in stack
        stack.clear()
        if private:
            stack[_PRIVATE_FRAME] = ZERO
        for (_, address, width), value in summary.get("memory", ()):
            if value is not None:
                stack[(_ABSOLUTE, address, width)] = Bits.const(value)
    for name, r in (("a0", 10), ("a1", 11)):
        value = summary.get("returns", {}).get(name)
        if value is not None:
            registers[r] = Bits.const(value)


def _step(d, registers, stack, pc, word=None, fp=None, summaries=None):
    fp = [UNKNOWN] * 32 if fp is None else fp
    copied = _fp_effect(word, d, registers, fp)
    a, b = registers[d.rs1], registers[d.rs2]
    # A frame address that loses its symbolic base is no longer tracked.
    computes_from_frame = d.kind not in ("load", "load_unsigned", "store") and _FRAME in (a.base, b.base)
    value = UNKNOWN
    if d.kind in ("lui", "auipc"):
        value = Bits.const(d.imm + (pc if d.kind == "auipc" else 0))
    elif d.kind.endswith("i") and d.kind[:-1] in ("add", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu"):
        if d.bits == 32 and d.kind in ("srli", "srai"):
            a = arithmetic("and", a, Bits.const((1 << 32) - 1)) if d.kind == "srli" else _sext(a, 32)
        value = arithmetic(d.kind[:-1], a, Bits.const(d.imm))
    elif d.kind in ("slti", "sltiu"):
        value = arithmetic("slt" if d.kind == "slti" else "sltu", a, Bits.const(d.imm))
    elif d.kind in ("add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "mul"):
        if d.bits == 32 and d.kind in ("sll", "srl", "sra"):
            b = arithmetic("and", b, Bits.const(31))
            if d.kind == "srl":
                a = arithmetic("and", a, Bits.const((1 << 32) - 1))
            elif d.kind == "sra":
                a = _sext(a, 32)
        value = arithmetic(d.kind, a, b)
    elif d.kind in CALLS:
        # The psABI lets a callee clobber caller-saved registers and its
        # stack-argument area, so no slot value survives the call.
        if _frame_handed_out(registers, d.rs1):
            stack.pop(_PRIVATE_FRAME, None)
        summary = (summaries or {}).get(_call_target(d, registers, pc))
        if _summary_applies(summary, registers, stack):
            _apply_summary(summary, registers, stack, fp)
        else:
            for r in _CALLER_SAVED:
                registers[r] = UNKNOWN
            for r in _FP_CALLER_SAVED:
                fp[r] = UNKNOWN
            private = _PRIVATE_FRAME in stack
            stack.clear()
            if private:
                stack[_PRIVATE_FRAME] = ZERO
        value = Bits.const(pc + d.size)
    elif d.kind in ("load", "load_unsigned", "store"):
        address = arithmetic("add", a, Bits.const(d.imm))
        key = (address.base, address.offset, d.bits)
        if d.kind == "store":
            _store(stack, address, d.bits, b)
        else:
            if address.base == _FRAME:
                value = stack.get(key, UNKNOWN)
            elif address.exact is not None:
                value = stack.get((_ABSOLUTE, address.exact, d.bits), UNKNOWN)
            if d.kind == "load_unsigned":
                value = arithmetic("and", value, Bits.const((1 << d.bits) - 1))
            elif d.bits < 64:
                value = _sext(value, d.bits)
    elif d.kind == "opaque":
        registers[:] = [ZERO] + [UNKNOWN] * 31
        stack.clear()
    elif d.kind in ("fp_memory", "unknown", "fp"):
        effect = _coarse_effect(word, d)
        if effect[0] == "none":
            if any(registers[r].base == _FRAME for r in effect[1]):
                stack.pop(_PRIVATE_FRAME, None)
        elif effect[0] == "store":
            _, base, imm, bits = effect
            _store(stack, arithmetic("add", registers[base], Bits.const(imm)), bits, UNKNOWN)
        else:
            if _frame_handed_out(registers, *effect[1]):
                stack.pop(_PRIVATE_FRAME, None)
            _forget_unprivate(stack)
    if copied is not None:
        value = copied
    if d.rd:
        registers[d.rd] = _sext(value, 32) if d.bits == 32 and d.kind not in ("load_unsigned", "store") else value
        if computes_from_frame and registers[d.rd].base != _FRAME:
            stack.pop(_PRIVATE_FRAME, None)
    registers[0] = ZERO

def _defined_registers(d):
    if d.kind == "opaque":
        return set(range(1, 32))
    if d.kind in CALLS:
        return set(_CALLER_SAVED) | ({d.rd} if d.rd else set())
    return {d.rd} if d.rd else set()

_CONDITIONS = {
    "beq": lambda x, y: x == y,
    "bne": lambda x, y: x != y,
    "blt": lambda x, y: x < y,
    "bge": lambda x, y: x >= y,
    "bltu": lambda x, y: x < y,
    "bgeu": lambda x, y: x >= y,
}
