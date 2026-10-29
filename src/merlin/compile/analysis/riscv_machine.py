"""RV64 instruction boundaries and conservative register facts from actual words.

This is a CPU frontend, not an accelerator emulator. Accelerator meaning comes
from the caller's selected operand observer. Unknown writes, calls and joins
discard facts; a static instruction site never supplies execution multiplicity.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass

MASK = (1 << 64) - 1
_REGIONS = OrderedDict()


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


@dataclass(frozen=True)
class CPUInstruction:
    kind: str
    size: int = 4
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    target: int | None = None
    bits: int = 64


def decode_cpu(row):
    """Decode standard RV64GC scalar effects without relying on objdump aliases."""
    if row.relocations:
        return CPUInstruction("opaque", 2 if len(row.hexcode) == 4 else 4)
    if not row.hexcode:
        return CPUInstruction("opaque")
    w = int(row.hexcode, 16)
    pc = row.addr
    if w & 3 != 3:
        if w >= 1 << 16:
            return CPUInstruction("opaque", 2)
        q, f = w & 3, w >> 13
        rd, r1, r2 = (w >> 7) & 31, 8 + ((w >> 7) & 7), 8 + ((w >> 2) & 7)
        ci = signed(((w >> 2) & 31) | ((w >> 12) & 1) << 5, 6)
        if q == 0:
            if f == 0:
                imm = ((w >> 7) & 15) << 6 | ((w >> 11) & 3) << 4 | ((w >> 5) & 1) << 3 | ((w >> 6) & 1) << 2
                return CPUInstruction("addi", 2, r2, 2, imm=imm) if imm else CPUInstruction("opaque", 2)
            if f in (2, 6):
                imm = ((w >> 5) & 1) << 6 | ((w >> 10) & 7) << 3 | ((w >> 6) & 1) << 2
                return CPUInstruction("load" if f == 2 else "store", 2, r2 if f == 2 else 0, r1, r2, imm, bits=32)
            if f in (3, 7):
                imm = ((w >> 5) & 3) << 6 | ((w >> 10) & 7) << 3
                return CPUInstruction("load" if f == 3 else "store", 2, r2 if f == 3 else 0, r1, r2, imm)
            return CPUInstruction("fp_memory" if f in (1, 5) else "opaque", 2)
        if q == 1:
            if f in (0, 1, 2):
                if f == 1 and rd == 0:
                    return CPUInstruction("opaque", 2)
                return CPUInstruction("addi", 2, rd, 0 if f == 2 else rd, imm=ci, bits=32 if f == 1 else 64)
            if f == 3:
                if rd == 2:
                    imm = signed(
                        ((w >> 12) & 1) << 9
                        | ((w >> 3) & 3) << 7
                        | ((w >> 5) & 1) << 6
                        | ((w >> 2) & 1) << 5
                        | ((w >> 6) & 1) << 4,
                        10,
                    )
                    return CPUInstruction("addi", 2, 2, 2, imm=imm) if imm else CPUInstruction("opaque", 2)
                if rd == 0 or ci == 0:
                    return CPUInstruction("opaque", 2)
                return CPUInstruction("lui", 2, rd, imm=ci << 12)
            if f == 4:
                mode = (w >> 10) & 3
                if mode != 3:
                    return CPUInstruction(
                        ("srli", "srai", "andi")[mode],
                        2,
                        r1,
                        r1,
                        imm=ci if mode == 2 else ((w >> 2) & 31) | ((w >> 12) & 1) << 5,
                    )
                op = (w >> 5) & 3
                if w & (1 << 12):
                    return CPUInstruction(("sub", "add", "unknown", "unknown")[op], 2, r1, r1, r2, bits=32)
                return CPUInstruction(("sub", "xor", "or", "and")[op], 2, r1, r1, r2)
            if f == 5:
                imm = signed(
                    ((w >> 12) & 1) << 11
                    | ((w >> 8) & 1) << 10
                    | ((w >> 9) & 3) << 8
                    | ((w >> 6) & 1) << 7
                    | ((w >> 7) & 1) << 6
                    | ((w >> 2) & 1) << 5
                    | ((w >> 11) & 1) << 4
                    | ((w >> 3) & 7) << 1,
                    12,
                )
                return CPUInstruction("jump", 2, target=pc + imm)
            if f in (6, 7):
                imm = signed(
                    ((w >> 12) & 1) << 8
                    | ((w >> 5) & 3) << 6
                    | ((w >> 2) & 1) << 5
                    | ((w >> 10) & 3) << 3
                    | ((w >> 3) & 3) << 1,
                    9,
                )
                return CPUInstruction("beq" if f == 6 else "bne", 2, rs1=r1, target=pc + imm)
        if q == 2:
            if f == 0:
                return CPUInstruction("slli", 2, rd, rd, imm=((w >> 2) & 31) | ((w >> 12) & 1) << 5)
            if f in (2, 3):
                if rd == 0:
                    return CPUInstruction("opaque", 2)
                imm = (
                    (((w >> 2) & 3) << 6 | ((w >> 12) & 1) << 5 | ((w >> 4) & 7) << 2)
                    if f == 2
                    else (((w >> 2) & 7) << 6 | ((w >> 12) & 1) << 5 | ((w >> 5) & 3) << 3)
                )
                return CPUInstruction("load", 2, rd, 2, imm=imm, bits=32 if f == 2 else 64)
            if f in (6, 7):
                imm = (
                    (((w >> 7) & 3) << 6 | ((w >> 9) & 15) << 2)
                    if f == 6
                    else (((w >> 7) & 7) << 6 | ((w >> 10) & 7) << 3)
                )
                return CPUInstruction("store", 2, rs1=2, rs2=(w >> 2) & 31, imm=imm, bits=32 if f == 6 else 64)
            if f == 4:
                rs = (w >> 2) & 31
                if not rs:
                    if rd == 0:
                        return CPUInstruction("opaque", 2)
                    return CPUInstruction(
                        "call_indirect" if w & (1 << 12) else "return" if rd in (1, 5) else "indirect",
                        2,
                        1 if w & (1 << 12) else 0,
                        rd,
                    )
                return CPUInstruction("add", 2, rd, rd if w & (1 << 12) else 0, rs)
            return CPUInstruction("fp_memory" if f in (1, 5) else "opaque", 2)
        return CPUInstruction("opaque", 2)
    if w & 31 == 31:
        return CPUInstruction("opaque")
    opcode, rd, f, r1, r2, f7 = w & 127, (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31, (w >> 20) & 31, w >> 25
    imm = signed(w >> 20, 12)
    if opcode in (0x37, 0x17):
        return CPUInstruction("lui" if opcode == 0x37 else "auipc", rd=rd, imm=signed(w & 0xFFFFF000, 32))
    if opcode in (0x13, 0x1B):
        if opcode == 0x1B and f not in (0, 1, 5):
            return CPUInstruction("opaque")
        kind = {0: "addi", 2: "slti", 3: "sltiu", 4: "xori", 6: "ori", 7: "andi"}.get(f, "unknown")
        if f == 1:
            kind, imm = (
                ("slli" if w >> (25 if opcode == 0x1B else 26) == 0 else "unknown"),
                (w >> 20) & (31 if opcode == 0x1B else 63),
            )
        elif f == 5:
            top = w >> (25 if opcode == 0x1B else 26)
            kind, imm = (
                {0: "srli", 0x20 if opcode == 0x1B else 0x10: "srai"}.get(top, "unknown"),
                (w >> 20) & (31 if opcode == 0x1B else 63),
            )
        return CPUInstruction(kind, rd=rd, rs1=r1, imm=imm, bits=32 if opcode == 0x1B else 64)
    if opcode in (0x33, 0x3B):
        if opcode == 0x3B and f not in {0: (0, 1, 5), 0x20: (0, 5), 1: (0, 4, 5, 6, 7)}.get(f7, ()):
            return CPUInstruction("opaque")
        kind = (
            {0: "add", 1: "sll", 2: "slt", 3: "sltu", 4: "xor", 5: "srl", 6: "or", 7: "and"}.get(f)
            if f7 == 0
            else {0: "sub", 5: "sra"}.get(f)
            if f7 == 0x20
            else {0: "mul", 4: "div", 5: "divu", 6: "rem", 7: "remu"}.get(f)
            if f7 == 1
            else None
        )
        return CPUInstruction(kind or "unknown", rd=rd, rs1=r1, rs2=r2, bits=32 if opcode == 0x3B else 64)
    if opcode == 0x03:
        if f == 7:
            return CPUInstruction("opaque")
        return CPUInstruction("load_unsigned" if f >= 4 else "load", rd=rd, rs1=r1, imm=imm, bits=8 << (f & 3))
    if opcode == 0x23:
        if f > 3:
            return CPUInstruction("opaque")
        return CPUInstruction("store", rs1=r1, rs2=r2, imm=signed((w >> 25) << 5 | (w >> 7) & 31, 12), bits=8 << f)
    if opcode == 0x63:
        imm = signed((w >> 31) << 12 | ((w >> 7) & 1) << 11 | ((w >> 25) & 63) << 5 | ((w >> 8) & 15) << 1, 13)
        return CPUInstruction(
            {0: "beq", 1: "bne", 4: "blt", 5: "bge", 6: "bltu", 7: "bgeu"}.get(f, "opaque"),
            rs1=r1,
            rs2=r2,
            target=pc + imm,
        )
    if opcode == 0x6F:
        imm = signed((w >> 31) << 20 | ((w >> 12) & 255) << 12 | ((w >> 20) & 1) << 11 | ((w >> 21) & 1023) << 1, 21)
        return CPUInstruction("call" if rd in (1, 5) else "jump", rd=rd, target=pc + imm)
    if opcode == 0x67:
        if f != 0:
            return CPUInstruction("opaque")
        return CPUInstruction(
            "call_indirect" if rd in (1, 5) else "return" if r1 in (1, 5) and not imm and not rd else "indirect",
            rd=rd,
            rs1=r1,
            imm=imm,
        )
    if opcode == 0x0F:
        return CPUInstruction("fence" if f in (0, 1) else "opaque")
    if opcode == 0x73 and f == 0:
        return CPUInstruction("opaque")
    if opcode in (0x07, 0x27):
        return CPUInstruction("fp_memory")
    if opcode in (0x43, 0x47, 0x4B, 0x4F):
        return CPUInstruction("fp")
    if opcode == 0x53 and f7 not in (0x50, 0x51, 0x60, 0x61, 0x70, 0x71):
        return CPUInstruction("fp")
    # Includes custom opcodes: their result register is unknown until observed.
    return CPUInstruction("unknown", rd=rd)


BRANCHES = frozenset(("beq", "bne", "blt", "bge", "bltu", "bgeu"))
CALLS = frozenset(("call", "call_indirect"))


def section_of(row):
    return getattr(row, "section_name", "") or row.section or "unlabelled"


def region_index(rows, *, artifact_sha256, custom_opcode=None):
    """Partition complete executable sections; preserve every selected custom site.

    Direct targets and returns suggest boundaries. Branch-crossed intervals are
    joined. These are static code spans, not claimed source function extents.
    """
    cache_key = (
        artifact_sha256,
        custom_opcode,
        len(rows),
        rows[0].addr if rows else None,
        rows[-1].addr if rows else None,
    )
    if cache_key in _REGIONS:
        _REGIONS.move_to_end(cache_key)
        return _REGIONS[cache_key]
    sections = {}
    # A startup/driver section can call a kernel in another executable
    # section. Its target is also a boundary for tail transfers in that code.
    call_targets = {d.target for row in rows if (d := decode_cpu(row)).kind == "call"}
    for first, second in zip(rows, rows[1:]):
        high, low = decode_cpu(first), decode_cpu(second)
        if (
            high.kind == "auipc"
            and high.rd
            and low.kind == "call_indirect"
            and low.rs1 == high.rd
            and first.addr + high.size == second.addr
            and section_of(first) == section_of(second)
        ):
            call_targets.add((first.addr + high.imm + low.imm) & MASK & ~1)
    for row in rows:
        sections.setdefault(section_of(row), []).append(row)
    regions = []
    for section, body in sections.items():
        body = sorted(body, key=lambda r: r.addr)
        decoded = [decode_cpu(r) for r in body]
        starts = [0]
        for i in range(1, len(body)):
            if (
                body[i].addr in call_targets
                or decoded[i - 1].kind == "return"
                or body[i].addr != body[i - 1].addr + decoded[i - 1].size
            ):
                starts.append(i)
        bounds = starts + [len(body)]
        owner = [0] * len(body)
        for k, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            owner[lo:hi] = [k] * (hi - lo)
        addresses = {r.addr: i for i, r in enumerate(body)}
        reach = list(range(len(starts)))
        for i, d in enumerate(decoded):
            if d.kind not in BRANCHES | {"jump"} or d.target not in addresses:
                continue
            if d.kind == "jump" and d.target in call_targets:
                continue
            lo, hi = sorted((owner[i], owner[addresses[d.target]]))
            reach[lo] = max(reach[lo], hi)
        k = 0
        while k < len(starts):
            end, cursor = reach[k], k
            while cursor <= end:
                end = max(end, reach[cursor])
                cursor += 1
            selected = body[bounds[k] : bounds[end + 1]]
            start, stop = selected[0].addr, selected[-1].addr + decode_cpu(selected[-1]).size
            counts = Counter()
            for r in selected:
                if r.hexcode and len(r.hexcode) == 8 and int(r.hexcode, 16) & 127 == custom_opcode:
                    counts[int(r.hexcode, 16) >> 25] += 1
            import hashlib

            identity = hashlib.sha256(f"{artifact_sha256}:{section}:{start:x}:{stop:x}".encode()).hexdigest()
            regions.append(
                (
                    dict(
                        region_id="region-" + identity[:24],
                        section=section,
                        entry_pc=start,
                        address_range=[start, stop],
                        boundary_status="inferred_static_span",
                        accelerator_sites=sum(counts.values()),
                        funct_sites=dict(sorted(counts.items())),
                    ),
                    selected,
                )
            )
            k = end + 1
    _REGIONS[cache_key] = regions
    while len(_REGIONS) > 3:
        _REGIONS.popitem(last=False)
    return regions


@dataclass(frozen=True)
class Bits:
    mask: int = 0
    value: int = 0
    base: str | None = None
    offset: int = 0

    @classmethod
    def const(cls, value):
        return cls(MASK, value & MASK)

    @property
    def exact(self):
        return self.value if self.mask == MASK else None

    def join(self, other):
        mask = self.mask & other.mask & ~(self.value ^ other.value) & MASK
        pointer = self.base is not None and self.base == other.base and self.offset == other.offset
        return Bits(mask, self.value & mask, self.base if pointer else None, self.offset if pointer else 0)

    def record(self):
        return dict(
            kind="const" if self.exact is not None else "unknown",
            raw=self.exact,
            **(
                dict(known_bits=dict(mask=self.mask, value=self.value, width=64))
                if self.exact is None and self.mask
                else {}
            ),
            **(dict(pointer_base=self.base, byte_offset=self.offset) if self.base else {}),
        )


UNKNOWN = Bits()
ZERO = Bits.const(0)


def _sext(value, bits):
    low = (1 << bits) - 1
    mask, result = value.mask & low, value.value & low
    if mask & (1 << (bits - 1)):
        mask |= MASK ^ low
        if result & (1 << (bits - 1)):
            result |= MASK ^ low
    return Bits(mask, result)


def arithmetic(kind, a, b):
    av, bv = a.exact, b.exact
    if kind in ("add", "sub"):
        if bv is not None and a.base:
            delta = signed(bv, 64) * (1 if kind == "add" else -1)
            return Bits(base=a.base, offset=a.offset + delta)
        if av is not None and b.base and kind == "add":
            return Bits(base=b.base, offset=b.offset + signed(av, 64))
        if av is not None and bv is not None:
            return Bits.const(av + bv if kind == "add" else av - bv)
        common = a.mask & b.mask
        low = (common ^ (common + 1)) >> 1  # bits below the first unknown carry
        value = a.value + b.value if kind == "add" else a.value - b.value
        return Bits(low, value & low)
    if kind == "and":
        zero = (a.mask & ~a.value) | (b.mask & ~b.value)
        ones = a.mask & a.value & b.mask & b.value
        return Bits((zero | ones) & MASK, ones)
    if kind == "or":
        ones = (a.mask & a.value) | (b.mask & b.value)
        zero = a.mask & ~a.value & b.mask & ~b.value
        return Bits((zero | ones) & MASK, ones)
    if kind == "xor":
        return Bits(a.mask & b.mask, (a.value ^ b.value) & a.mask & b.mask)
    if kind in ("sll", "srl", "sra") and bv is not None:
        amount = bv & 63
        if not amount:
            return a
        if kind == "sll":
            return Bits(((a.mask << amount) | ((1 << amount) - 1)) & MASK, (a.value << amount) & MASK)
        high = MASK ^ ((1 << (64 - amount)) - 1)
        sign_known = bool(a.mask & (1 << 63))
        fill_mask = high if kind == "srl" or sign_known else 0
        fill_value = high if kind == "sra" and sign_known and a.value >> 63 else 0
        return Bits((a.mask >> amount) | fill_mask, (a.value >> amount) | fill_value)
    if av is not None and bv is not None:
        if kind == "mul":
            return Bits.const(av * bv)
        if kind in ("slt", "sltu"):
            return Bits.const(int((signed(av, 64) < signed(bv, 64)) if kind == "slt" else av < bv))
    if kind in ("slt", "sltu"):
        return Bits(MASK ^ 1, 0)
    return UNKNOWN


def _step(d, registers, stack, pc):
    a, b = registers[d.rs1], registers[d.rs2]
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
        for r in (1, 5, 6, 7, *range(10, 18), *range(28, 32)):
            registers[r] = UNKNOWN
        stack.clear()
        value = Bits.const(pc + d.size)
    elif d.kind in ("load", "load_unsigned", "store"):
        address = arithmetic("add", a, Bits.const(d.imm))
        key = (address.base, address.offset, d.bits)
        if d.kind == "store":
            if address.base == "entry_sp":
                lo, hi = address.offset, address.offset + d.bits // 8
                for old in list(stack):
                    if lo < old[1] + old[2] // 8 and old[1] < hi:
                        del stack[old]
                if len(stack) < 256:
                    stack[key] = b
            else:
                stack.clear()
        else:
            value = stack.get(key, UNKNOWN) if address.base == "entry_sp" else UNKNOWN
            if d.kind == "load_unsigned":
                value = arithmetic("and", value, Bits.const((1 << d.bits) - 1))
            elif d.bits < 64:
                value = _sext(value, d.bits)
    elif d.kind == "opaque":
        registers[:] = [ZERO] + [UNKNOWN] * 31
        stack.clear()
    elif d.kind in ("fp_memory", "unknown"):
        stack.clear()
    if d.rd:
        registers[d.rd] = _sext(value, 32) if d.bits == 32 and d.kind not in ("load_unsigned", "store") else value
    registers[0] = ZERO


def register_facts(rows, *, max_steps=2_000_000):
    """Fixed-point known bits at each instruction, including loops and joins.

    Only reachable blocks receive observations. An unresolved control transfer
    is retained as a gap. No register is initialized to an invented zero.
    """
    if not rows:
        return {}, dict(status="unknown", reason="empty_machine_region")
    if any(row.relocations for row in rows):
        return {}, dict(status="unknown", reason="unresolved_object_relocations_require_linked_bytes")
    decoded = {r.addr: decode_cpu(r) for r in rows}
    if len(decoded) != len(rows):
        return {}, dict(status="unknown", reason="repeated_pc_requires_execution_view")
    order = sorted(decoded)
    leaders = {rows[0].addr}
    for pc, d in decoded.items():
        following = pc + d.size
        if d.target in decoded and d.kind in BRANCHES | {"jump"}:
            leaders.add(d.target)
        if d.kind in BRANCHES | CALLS | {"jump", "return", "indirect"} and following in decoded:
            leaders.add(following)
    for left, right in zip(order, order[1:]):
        if left + decoded[left].size != right:
            leaders.add(right)
    blocks, owner = {}, {}
    for pc in order:
        if pc in leaders or not blocks:
            current = pc
            blocks[current] = []
        blocks[current].append(pc)
        owner[pc] = current
    entry = rows[0].addr
    initial = [ZERO] + [UNKNOWN] * 31
    initial[2] = Bits(base="entry_sp")
    for i in range(10, 18):
        initial[i] = Bits(base=f"arg{i - 10}")
    incoming = {entry: (tuple(initial), {})}
    queue, queued, observations = deque([entry]), {entry}, {}
    gaps, steps = set(), 0
    while queue:
        block = queue.popleft()
        queued.discard(block)
        registers, stack = list(incoming[block][0]), dict(incoming[block][1])
        for pc in blocks[block]:
            steps += 1
            if steps > max_steps:
                return {}, dict(status="unknown", reason="machine_dataflow_budget_exhausted", steps=steps)
            observations[pc] = tuple(registers)
            _step(decoded[pc], registers, stack, pc)
        last = blocks[block][-1]
        d = decoded[last]
        following = last + d.size
        successors = []
        if d.kind in BRANCHES | {"jump"}:
            if d.target in decoded:
                successors.append(d.target)
            else:
                gaps.add("branch_leaves_selected_span")
        if d.kind not in {"jump", "return", "indirect"} and following in decoded:
            successors.append(following)
        if d.kind == "indirect":
            gaps.add("unresolved_indirect_transfer")
        for target in successors:
            target = owner[target]
            old = incoming.get(target)
            new = (tuple(registers), dict(stack))
            if old:
                new = (
                    tuple(a.join(b) for a, b in zip(old[0], registers)),
                    {key: value.join(stack[key]) for key, value in old[1].items() if key in stack},
                )
            if old != new:
                incoming[target] = new
                if target not in queued:
                    queued.add(target)
                    queue.append(target)
    return observations, dict(
        status="analyzed",
        blocks=len(blocks),
        reachable_blocks=len(incoming),
        reachable_instructions=len(observations),
        steps=steps,
        gaps=sorted(gaps),
        multiplicity="unknown",
    )


def finite_control_counts(rows, *, custom_opcode, max_steps=2_000_000, max_commands=1_000_000):
    """Count a complete original body only when every scalar branch is known.

    Symbolic argument pointers preserve identity and offsets. Tensor loads and
    accelerator results stay unknown. Calls, data-dependent control, unsupported
    effects and exhausted budgets refuse the *entire* observation.
    """
    if not rows:
        return dict(status="unknown", reason="empty_machine_region")
    if any(row.relocations for row in rows):
        return dict(status="unknown", reason="unresolved_object_relocations_require_linked_bytes")
    by_pc = {row.addr: row for row in rows}
    if len(by_pc) != len(rows):
        return dict(status="unknown", reason="repeated_pc_requires_execution_view")
    registers = [ZERO] + [UNKNOWN] * 31
    registers[2] = Bits(base="entry_sp")
    for index in range(10, 18):
        registers[index] = Bits(base=f"arg{index - 10}")
    pc, stack, executions, command_counts = rows[0].addr, {}, Counter(), Counter()
    commands = fences = 0
    for step in range(max_steps):
        row = by_pc.get(pc)
        if row is None:
            return dict(status="unknown", reason="control_leaves_selected_span", pc=pc)
        d = decode_cpu(row)
        custom = len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == custom_opcode
        if d.kind in CALLS | {"indirect", "opaque"} or (d.kind == "unknown" and not custom):
            return dict(status="unknown", reason="opaque_control_requires_capture", pc=pc)
        executions[pc] += 1
        if custom:
            commands += 1
            command_counts[pc] += 1
            if commands > max_commands:
                return dict(status="unknown", reason="finite_command_budget_exhausted")
        if d.kind == "fence":
            fences += 1
        if d.kind == "return":
            return dict(
                status="proved",
                count_basis="proved_finite_scalar_control",
                scope="selected_function_body",
                instruction_executions=dict(executions),
                command_executions=dict(command_counts),
                commands=commands,
                fences=fences,
                steps=step + 1,
                hardware_execution="unknown",
            )
        following = pc + d.size
        if d.kind in BRANCHES:
            a, b = registers[d.rs1], registers[d.rs2]
            av, bv = a.exact, b.exact
            if av is None or bv is None:
                # Equality of offsets is invariant under pointer wraparound.
                # Ordered comparisons need a proven address range instead.
                if d.kind in ("beq", "bne") and a.base is not None and a.base == b.base:
                    av, bv = a.offset, b.offset
                else:
                    return dict(status="unknown", reason="data_dependent_or_unbound_control", pc=pc)
            if d.kind in ("blt", "bge"):
                av, bv = signed(av, 64), signed(bv, 64)
            condition = {
                "beq": av == bv,
                "bne": av != bv,
                "blt": av < bv,
                "bge": av >= bv,
                "bltu": av < bv,
                "bgeu": av >= bv,
            }[d.kind]
            following = d.target if condition else following
        elif d.kind == "jump":
            following = d.target
        _step(d, registers, stack, pc)
        pc = following
    return dict(status="unknown", reason="finite_instruction_budget_exhausted")
