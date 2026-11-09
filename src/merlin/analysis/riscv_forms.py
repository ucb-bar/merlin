"""Affine register/frame forms and conservative transfer over one loop iteration.

Observations retain their exact supported domain and explicit unknowns.
No accelerator semantics, execution multiplicity or hardware law is inferred.
"""

from itertools import product

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
from .riscv_state import (
    _FRAME as _FRAME,
)
from .riscv_state import (
    _coarse_effect as _coarse_effect,
)
from .riscv_state import (
    _defined_registers as _defined_registers,
)
from .riscv_state import (
    _step as _step,
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

# Affine forms are (constant, symbol, terms): ``[symbol +] constant +
# sum(step * k_header)`` over ``terms = ((header_pc, step, count), ...)``.
# A symbol is a pointer base of the fixed point (``entry_sp``, ``argN``) or,
# inside one loop iteration, the value a register/frame slot held at the
# header. Frame slots are keyed ``(offset_from_entry_sp, bits)``.


def _form_add(f, g, sign=1):
    if f is None or g is None or isinstance(f, _Table) or isinstance(g, _Table):
        return None
    (fc, fp, ft), (gc, gp, gt) = f, g
    if gp is not None:
        if sign < 0 and gp == fp:
            fp = None
        elif sign > 0 and fp is None:
            fp = gp
        else:
            return None
    terms = {h: (s, n) for h, s, n in ft}
    for h, s, n in gt:
        terms[h] = (terms.get(h, (0, n))[0] + sign * s, n)
    return (fc + sign * gc, fp, tuple(sorted((h, s, n) for h, (s, n) in terms.items() if s)))


def _form_scale(f, c):
    if f is None or isinstance(f, _Table) or (f[1] is not None and c != 1):
        return None
    return (f[0] * c, f[1], tuple((h, s * c, n) for h, s, n in f[2] if s * c))


def _form_fits32(f):
    if f is None or f[1] is not None:
        return None
    low = high = signed(f[0] & MASK, 64)
    for _, s, n in f[2]:
        low, high = low + min(0, s * (n - 1)), high + max(0, s * (n - 1))
    return (signed(f[0] & MASK, 64), None, f[2]) if -(1 << 31) <= low and high < (1 << 31) else None


class _Slots:
    """Frame-slot forms within one iteration; unwritten slots read lazily."""

    def __init__(self, known=None, lazy=True, written=()):
        self.known, self.lazy, self.written = dict(known or {}), lazy, list(written)

    def copy(self):
        return _Slots(self.known, self.lazy, self.written)

    def read(self, key, seed):
        if key in self.known:
            return self.known[key]
        lo, hi = key[0], key[0] + key[1] // 8
        if not self.lazy or hi > 0 or any(lo < h and w < hi for w, h in self.written):
            return None
        return seed(key)

    def write(self, key, form):
        lo, hi = key[0], key[0] + key[1] // 8
        for old in [k for k in self.known if lo < k[0] + k[1] // 8 and k[0] < hi]:
            del self.known[old]
        self.written.append((lo, hi))
        if form is not None:
            self.known[key] = form

    def clear(self, private):
        if private:
            for old in [k for k in self.known if k[0] + k[1] // 8 > 0]:
                del self.known[old]
        else:
            self.known.clear()
            self.lazy = False

    def join(self, other):
        return _Slots(
            {k: f for k, f in self.known.items() if k in other.known and other.known[k] == f},
            self.lazy and other.lazy,
            self.written + [w for w in other.written if w not in self.written],
        )


def _join_states(states):
    regs, slots = dict(states[0][0]), states[0][1].copy()
    for other_regs, other_slots in states[1:]:
        regs = {r: f if other_regs.get(r) == f else None for r, f in regs.items()}
        slots = slots.join(other_slots)
    return regs, slots


def _child_effect(cfg, child, regs, slots):
    defined = {r for b in child["body"] for pc in cfg.blocks[b] for r in _defined_registers(cfg.decoded[pc])}
    regs = {r: (None if r in defined else f) for r, f in regs.items()}
    slots = slots.copy()
    for b in child["body"]:
        for pc in cfg.blocks[b]:
            d = cfg.decoded[pc]
            if d.kind in CALLS or d.kind == "opaque":
                slots.clear(False)
            elif d.kind == "store" or (
                d.kind in ("fp_memory", "unknown", "fp") and _coarse_effect(cfg.words[pc], d)[0] != "none"
            ):
                key = _frame_key(cfg, pc, d)
                if key is not None:
                    slots.write(key, None)
                else:
                    slots.clear(cfg.private_after.get(pc, False))
    return regs, slots


def _frame_key(cfg, pc, d):
    """Frame slot (offset, bits) accessed by a load/store at ``pc``, else None."""
    if d.kind in ("load", "load_unsigned", "store"):
        base, imm, bits = d.rs1, d.imm, d.bits
    else:
        effect = _coarse_effect(cfg.words[pc], d)
        if effect[0] != "store":
            return None
        _, base, imm, bits = effect
    value = cfg.observations[pc][base]
    if value.base != _FRAME:
        return None
    return (value.offset + imm, bits)


def _run_iteration(cfg, loop, loops, seed_regs, seed_slot, narrow=None, visit=None, child_seed=None):
    """Forms of one iteration of ``loop`` from its header to every own block.

    Child loops are summarized by what they may write. ``narrow`` collects the
    symbols that passed through a 32-bit operation; ``visit(pc, regs, slots)``
    sees the in-state of every own instruction.
    """
    header, body = loop["header"], loop["body"]
    children = [o for o in loops if o["parent"] is loop]
    child_of = {b: child for child in children for b in child["body"]}
    block_out, child_in = {}, {}
    for block in cfg.rpo:
        if block not in body or block in child_of:
            continue
        if block == header:
            regs, slots = dict(seed_regs), _Slots()
        else:
            incoming = []
            for p in cfg.pred.get(block, ()):
                child = child_of.get(p)
                if child is not None:
                    if id(child) not in child_in:
                        incoming = None
                        break
                    regs, slots = _child_effect(cfg, child, *child_in[id(child)])
                    exit_value = child.get("exit_value")
                    if exit_value is not None and p == exit_value[0]:
                        regs[exit_value[1]] = (exit_value[2], None, ())
                    incoming.append(_refine_edge(cfg, p, block, regs, slots))
                elif p in block_out and p in body:
                    incoming.append(block_out[p])
                else:
                    incoming = None  # entered other than through the header
                    break
            if not incoming:
                regs, slots = {r: None for r in range(1, 32)}, _Slots(lazy=False)
            else:
                regs, slots = _join_states(incoming)
        for pc in cfg.blocks[block]:
            state = cfg.observations[pc]
            if visit is not None:
                visit(pc, regs, slots)

            def operand(r, state=state, regs=regs):
                if r == 0:
                    return (0, None, ())
                if state[r].exact is not None:
                    return (state[r].exact, None, ())
                form = regs.get(r)
                if form is None and state[r].base is not None:
                    return (state[r].offset, state[r].base, ())
                return form

            d = cfg.decoded[pc]
            result = None
            if d.kind in ("load", "load_unsigned"):
                key = _frame_key(cfg, pc, d)
                if key is not None and d.kind == "load" and d.bits in (32, 64):
                    result = slots.read(key, seed_slot)
                    if result is not None and d.bits == 32:
                        result = _narrow(result, narrow)
            elif d.kind == "store":
                key = _frame_key(cfg, pc, d)
                if key is None:
                    slots.clear(cfg.private_after.get(pc, False))
                else:
                    value = operand(d.rs2) if d.bits in (32, 64) else None
                    slots.write(key, _narrow(value, narrow) if value is not None and d.bits == 32 else value)
            elif d.kind in CALLS or d.kind == "opaque":
                slots.clear(False)
            elif d.kind in ("fp_memory", "unknown", "fp"):
                if _coarse_effect(cfg.words[pc], d)[0] != "none":
                    key = _frame_key(cfg, pc, d)
                    if key is not None:
                        slots.write(key, None)
                    else:
                        slots.clear(cfg.private_after.get(pc, False))
            elif d.rd:
                result = _form_transfer(d, operand, narrow)
            for r in _defined_registers(d):
                regs[r] = None
            if d.rd and result is not None:
                regs[d.rd] = result
        block_out[block] = (regs, slots)
        for child in children:
            if child["header"] in cfg.succ.get(block, ()):
                state = (dict(regs), slots.copy())
                previous = child_in.get(id(child))
                child_in[id(child)] = state if previous is None else _join_states([previous, state])
    if child_seed is not None:
        child_seed.update(child_in)
    return block_out


def _refine_edge(cfg, p, block, regs, slots):
    """Equality facts of a ``beq``/``bne`` edge from ``p`` into ``block``.

    On the equal edge the register equals the constant. On the unequal edge a
    register with exactly one unknown bit, whose known bits match the
    constant, holds the constant with that bit flipped.
    """
    pc = cfg.blocks[p][-1]
    d = cfg.decoded[pc]
    if d.kind not in ("beq", "bne") or d.target == pc + d.size:
        return regs, slots
    equal = (block == d.target) == (d.kind == "beq")
    state = cfg.observations[pc]
    for r, other in ((d.rs1, d.rs2), (d.rs2, d.rs1)):
        constant = 0 if other == 0 else state[other].exact
        if r == 0 or constant is None or state[r].exact is not None:
            continue
        unknown = ~state[r].mask & MASK
        value = None
        if equal:
            value = constant
        elif unknown & (unknown - 1) == 0 and constant & state[r].mask == state[r].value:
            value = constant ^ unknown
        if value is not None:
            regs = dict(regs)
            regs[r] = (value, None, ())
    return regs, slots


def _narrow(form, narrow):
    """A 32-bit result: exact range check for terms, recorded for symbols."""
    if form is None:
        return None
    if isinstance(form, _Table):
        return _Table(form.dims, tuple(_sext(Bits.const(v), 32).exact for v in form.values))
    if form[1] is None:
        return _form_fits32(form)
    if narrow is None or form[2]:
        return None
    narrow.add(form[1])
    return form


_TABLE_LIMIT = 4096
_POINTWISE = frozenset(
    ("addi", "andi", "ori", "xori", "slli", "srli", "srai", "slti", "sltiu")
    + ("add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "mul")
)


def _form_dims(form):
    if isinstance(form, _Table):
        return form.dims
    return tuple((h, n) for h, _, n in form[2])


def _form_at(form, point):
    if isinstance(form, _Table):
        index = 0
        for h, n in form.dims:
            index = index * n + point[h]
        return form.values[index]
    return (form[0] + sum(s * point[h] for h, s, _ in form[2])) & MASK


def _pointwise(d, a, b):
    """Apply the CPU's own transfer at every joint iteration point."""
    if d.kind not in _POINTWISE or a is None or b is None:
        return None
    if any(not isinstance(f, _Table) and f[1] is not None for f in (a, b)):
        return None
    dims = dict(_form_dims(a))
    for h, n in _form_dims(b):
        if dims.setdefault(h, n) != n:
            return None
    dims = tuple(sorted(dims.items()))
    total = 1
    for _, n in dims:
        total *= n
    if not dims or total > _TABLE_LIMIT:
        return None
    values = []
    registers = [ZERO] + [UNKNOWN] * 31
    scratch = 31 if d.rd != 31 else 30
    for point in product(*(range(n) for _, n in dims)):
        at = dict(zip((h for h, _ in dims), point))
        registers[d.rs1] = Bits.const(_form_at(a, at)) if d.rs1 else ZERO
        if d.rs2 != d.rs1:
            registers[d.rs2] = Bits.const(_form_at(b, at)) if d.rs2 else ZERO
        registers[scratch] = UNKNOWN
        _step(d, registers, {}, 0)
        value = registers[d.rd].exact
        if value is None:
            return None
        values.append(value)
    if len(set(values)) == 1:
        return (values[0], None, ())
    return _Table(dims, tuple(values))


def _form_transfer(d, operand, narrow=None):
    a, b = operand(d.rs1), operand(d.rs2)
    if isinstance(a, _Table) or isinstance(b, _Table):
        return None if narrow is not None else _pointwise(d, a, b)
    if d.kind == "addi":
        result = _form_add(a, (d.imm, None, ()))
    elif d.kind in ("add", "sub"):
        result = _form_add(a, b, 1 if d.kind == "add" else -1)
    elif d.kind == "slli":
        result = _form_scale(a, 1 << (d.imm & 63))
    elif d.kind in ("sll", "mul") and b is not None and not b[2] and b[1] is None:
        result = _form_scale(a, (1 << (b[0] & 63)) if d.kind == "sll" else signed(b[0] & MASK, 64))
    elif d.kind == "mul" and a is not None and not a[2] and a[1] is None:
        result = _form_scale(b, signed(a[0] & MASK, 64))
    else:
        result = None
    if result is not None and result[1] is None and not result[2]:
        result = (result[0] & MASK, None, ())
    if result is not None:
        result = _narrow(result, narrow) if d.bits == 32 else result
    if result is None and narrow is None and (a is not None and a[2] or b is not None and b[2]):
        return _pointwise(d, a, b)
    return result
