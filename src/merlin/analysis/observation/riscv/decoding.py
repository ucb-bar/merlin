"""Decode RV64 scalar instructions and index control-flow regions."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

# ---- riscv_decode ------------------------------------------------------------------------------

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


def _ends_segment(d):
    """An unconditional transfer: control never falls through to the next word.

    Like a return, a direct or register jump ends the current segment, so a
    tail call (``jal zero``/``c.j``/``jalr zero``) no longer runs into the next
    function. A jump that stays inside a function (a loop back edge placed last,
    a jump over a block) rejoins its code through the branch-reach pass below.
    """
    return d.kind in ("return", "jump", "indirect")


def region_index(rows, *, artifact_sha256, custom_opcode=None):
    """Partition complete executable sections; preserve every selected custom site.

    Direct targets, returns and unconditional jumps suggest boundaries. Branch-crossed intervals are
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
                or _ends_segment(decoded[i - 1])
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


class _Cfg:
    def __init__(self, entry, blocks, successors, decoded):
        self.entry, self.blocks, self.succ, self.decoded = entry, blocks, successors, decoded
        self.pred = {b: [] for b in successors}
        for b, targets in successors.items():
            for t in targets:
                if t in self.pred:
                    self.pred[t].append(b)
        order, seen, stack = [], {entry}, [(entry, iter(successors.get(entry, ())))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                order.append(node)
                stack.pop()
            elif child not in seen and child in successors:
                seen.add(child)
                stack.append((child, iter(successors.get(child, ()))))
        self.rpo = order[::-1]
        self.index = {b: i for i, b in enumerate(self.rpo)}
        idom = {entry: entry}
        changed = True
        while changed:
            changed = False
            for b in self.rpo[1:]:
                done = [p for p in self.pred[b] if p in idom]
                if not done:
                    continue
                new = done[0]
                for p in done[1:]:
                    new = self._intersect(idom, p, new)
                if idom.get(b) != new:
                    idom[b] = new
                    changed = True
        self.idom = idom

    def _intersect(self, idom, a, b):
        while a != b:
            while self.index[a] > self.index[b]:
                a = idom[a]
            while self.index[b] > self.index[a]:
                b = idom[b]
        return a

    def dominates(self, a, b):
        while True:
            if a == b:
                return True
            if b == self.entry or b not in self.idom:
                return False
            b = self.idom[b]


ET_REL = 1  # ELF e_type of a relocatable object (System V gABI)


def linked_input_status(path, *, linked_names=("package_kernel.elf",)):
    """``None`` for a linked ELF; for a relocatable object an explicit refusal.

    Scalar stepping needs resolved control transfers, so an ``ET_REL`` input is
    reported as ``unlinked_object_use_linked_elf``; a linked file of one of
    ``linked_names`` beside it is named in the record, never substituted.
    """
    from pathlib import Path

    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(20)
    if header[:4] != b"\x7fELF" or len(header) < 18:
        return None
    e_type = int.from_bytes(header[16:18], "little" if header[5] == 1 else "big")
    if e_type != ET_REL:
        return None
    beside = [str(path.parent / name) for name in linked_names if (path.parent / name).is_file()]
    return dict(
        status="unknown",
        reason="unlinked_object_use_linked_elf",
        input=str(path),
        **(dict(linked_elf_beside=beside) if beside else {}),
    )


# ---- riscv_values ------------------------------------------------------------------------------
# Conservative known-bit, pointer and bounded affine value domains for RV64.


@dataclass(frozen=True)
class Bits:
    mask: int = 0
    value: int = 0
    base: str | None = None
    offset: int = 0
    # Post-fixpoint loop annotation (never part of the fixed point itself):
    # (constant, pointer_base, ((header_pc, step, count), ...)) meaning
    # value = [pointer_base +] constant + sum(step * k_header), 0 <= k < count.
    affine: tuple | None = None
    # Symbolic bit window ``(symbol, offset, at, width)``: bits ``[at, at + width)`` equal
    # ``(symbol + offset) mod 2**width`` for one unknown value ``symbol``; ``mask``/``value``
    # describe only bits outside the window. Two windows of one symbol are comparable.
    window: tuple | None = None

    @classmethod
    def const(cls, value):
        return cls(MASK, value & MASK)

    @property
    def exact(self):
        return self.value if self.mask == MASK else None

    def join(self, other):
        mask = self.mask & other.mask & ~(self.value ^ other.value) & MASK
        pointer = self.base is not None and self.base == other.base and self.offset == other.offset
        window = self.window if self.window == other.window else None
        return Bits(
            mask, self.value & mask, self.base if pointer else None, self.offset if pointer else 0, window=window
        )

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
            **(_affine_record(self.affine) if self.affine else {}),
            **(_window_record(self.window) if self.window else {}),
        )


_VALUE_SET_LIMIT = 4096


def _affine_record(affine):
    if isinstance(affine, _Table):
        return dict(
            affine=dict(
                terms=[dict(header_pc=h, count=n) for h, n in affine.dims],
                basis="proved_trip_counts_pointwise_cpu_semantics",
            ),
            value_set=sorted(set(affine.values)),
        )
    constant, pointer, terms = affine
    out = dict(
        affine=dict(
            base=constant & MASK if pointer is None else None,
            **(dict(pointer_base=pointer, byte_offset=constant) if pointer else {}),
            terms=[dict(header_pc=h, step=step, count=count) for h, step, count in terms],
            basis="proved_induction_and_trip_counts",
        )
    )
    total = 1
    for _, _, count in terms:
        total *= count
    if pointer is None and total <= _VALUE_SET_LIMIT:
        values = {constant}
        for _, step, count in terms:
            values = {v + step * k for v in values for k in range(count)}
        out["value_set"] = sorted(v & MASK for v in values)
    return out


UNKNOWN = Bits()

ZERO = Bits.const(0)


def _window_record(window):
    symbol, offset, at, width = window
    return dict(window=dict(symbol=symbol, offset=offset, at=at, width=width))


def _cut_window(window, at, width):
    """The part of a window that lies in bits ``[at, at + width)`` of its own position, or None."""
    if window is None or width <= 0:
        return None
    return (window[0], window[1] % (1 << width), at, width)


def _sext(value, bits):
    low = (1 << bits) - 1
    mask, result = value.mask & low, value.value & low
    if mask & (1 << (bits - 1)):
        mask |= MASK ^ low
        if result & (1 << (bits - 1)):
            result |= MASK ^ low
    window = value.window
    if window is not None:
        window = _cut_window(window, window[2], min(window[3], bits - window[2]))
    return Bits(mask, result, window=window)


def _window_arithmetic(kind, a, b):
    """Integer operations on a symbolic window and an exact operand.

    Low bits of a sum depend only on low operand bits, so an add keeps a window at bit 0.
    An OR or AND with an exact value keeps the window's bits below the first bit the value
    changes; constant shifts move it. Anything else leaves the window's bits unknown.
    """
    if b.window is not None and a.window is None and kind in ("add", "or", "and"):
        a, b = b, a
    if kind in ("slt", "sltu"):
        return Bits(MASK ^ 1, 0)
    constant = b.exact
    if b.window is not None or constant is None:
        return UNKNOWN
    symbol, offset, at, width = a.window
    span = ((1 << width) - 1) << at
    outside = Bits(a.mask & ~span & MASK, a.value & ~span & MASK)
    if kind in ("add", "sub"):
        if at:
            return UNKNOWN
        delta = constant if kind == "add" else -constant
        return Bits(window=(symbol, (offset + delta) % (1 << width), 0, width))
    if kind in ("or", "and"):
        inside = (constant & span) >> at
        changed = inside if kind == "or" else ~inside & ((1 << width) - 1)
        kept = min(width, (changed & -changed).bit_length() - 1) if changed else width
        rest = arithmetic(kind, outside, b)
        keep = ((1 << kept) - 1) << at
        return Bits(rest.mask & ~keep & MASK, rest.value & ~keep & MASK, window=_cut_window(a.window, at, kept))
    if kind in ("sll", "srl", "sra"):
        amount = constant & 63
        if not amount:
            return a
        rest = arithmetic(kind, outside, b)
        if kind == "sll":
            moved = _cut_window(a.window, at + amount, min(width, 64 - at - amount))
        else:
            moved = _cut_window(a.window, at - amount, width) if at >= amount else None
        if moved is None:
            return rest
        keep = ((1 << moved[3]) - 1) << moved[2]
        return Bits(rest.mask & ~keep & MASK, rest.value & ~keep & MASK, window=moved)
    return UNKNOWN


def arithmetic(kind, a, b):
    if a.window is not None or b.window is not None:
        return _window_arithmetic(kind, a, b)
    av, bv = a.exact, b.exact
    if kind in ("add", "sub"):
        if bv is not None and a.base:
            delta = signed(bv, 64) * (1 if kind == "add" else -1)
            return Bits(base=a.base, offset=a.offset + delta)
        if av is not None and b.base and kind == "add":
            return Bits(base=b.base, offset=b.offset + signed(av, 64))
        if av is not None and bv is not None:
            return Bits.const(av + bv if kind == "add" else av - bv)
        # Standard known-bits addition with an explicit carry (a - b == a + ~b + 1).
        # A bit is known when both operand bits and its incoming carry are known.
        a_zero, a_one = a.mask & ~a.value & MASK, a.mask & a.value
        b_zero, b_one = b.mask & ~b.value & MASK, b.mask & b.value
        carry = 0
        if kind == "sub":
            b_zero, b_one, carry = b_one, b_zero, 1
        largest = ((~a_zero & MASK) + (~b_zero & MASK) + carry) & MASK
        smallest = (a_one + b_one + carry) & MASK
        carry_zero = ~(largest ^ a_zero ^ b_zero) & MASK
        carry_one = (smallest ^ a_one ^ b_one) & MASK
        known = (a_zero | a_one) & (b_zero | b_one) & (carry_zero | carry_one) & MASK
        return Bits(known, smallest & known)
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


@dataclass(frozen=True)
class _Table:
    """A register value at every point of its loops' joint iteration space.

    ``dims`` is ((header_pc, count), ...); ``values`` enumerates the points in
    row-major order (last header fastest), each reduced modulo 2**64.
    """

    dims: tuple
    values: tuple
