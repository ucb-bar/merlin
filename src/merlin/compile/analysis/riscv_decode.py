"""Decode RV64 scalar instructions and index control-flow regions."""

from __future__ import annotations

from collections import Counter, OrderedDict
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
