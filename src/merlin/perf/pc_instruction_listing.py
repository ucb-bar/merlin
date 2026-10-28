"""Read register-free PC/word listings without claiming retirement or timing.

The provider adapts a structured text format to the shared disassembly and RoCC
readers. Calls delimit listed bodies by their return PC, not by the first `ret`:
static listings can contain return instructions before out-of-line tail blocks.
It never executes scalar instructions or initializes missing architectural state.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.compile.analysis.disassembly import RawInsn, tokenize_text
from merlin.compile.analysis.rocc_commands import audit_decoded, decode_stream


@dataclass(frozen=True)
class ListingRegion:
    region_id: int
    parent: int | None
    entry_pc: int | None
    call_index: int | None
    return_pc: int | None
    indices: tuple[int, ...]
    closed: bool

    def to_dict(self) -> dict[str, Any]:
        runs: list[list[int]] = []
        for index in self.indices:
            if runs and runs[-1][1] == index:
                runs[-1][1] += 1
            else:
                runs.append([index, index + 1])
        return {"region_id": self.region_id, "parent": self.parent,
                "entry_pc": self.entry_pc, "call_index": self.call_index,
                "return_pc": self.return_pc, "index_runs": runs, "closed": self.closed}


def read_listing(lines: Iterable[str]) -> tuple[RawInsn, ...]:
    """Parse every nonblank ``0xPC 0xWORD mnemonic operands`` row or reject it.

    Both columns are hexadecimal and may be zero padded. The existing structured
    tokenizer owns operand splitting. A malformed row is never silently dropped.
    """
    out = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split(None, 2)
        try:
            if len(fields) != 3:
                raise ValueError("missing PC/word/disassembly column")
            pc, word = int(fields[0], 16), int(fields[1], 16)
            if not 0 <= pc < (1 << 64) or pc % 2 or not 0 <= word < (1 << 32):
                raise ValueError("PC or word out of range")
            parsed = tokenize_text(f"{pc:x}: {word:08x} {fields[2]}")
            if len(parsed) != 1:
                raise ValueError("disassembly column is not one instruction")
            riscv_width(parsed[0])  # reject truncation/unsupported lengths
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid PC listing row {line_number}: {exc}") from exc
        out.append(parsed[0])
    return tuple(out)


def riscv_width(insn: RawInsn) -> int:
    """RISC-V instruction-length encoding; only complete 16/32-bit rows supported."""
    word = int(insn.hexcode, 16)
    if word & 3 != 3:
        if word >= 1 << 16:
            raise ValueError("nonzero padding above a compressed instruction")
        return 2
    if word & 0x1F == 0x1F:
        raise ValueError("instruction length exceeds the listing word column")
    return 4


def instruction_bytes(insns: Iterable[RawInsn]) -> bytes:
    return b"".join(int(insn.hexcode, 16).to_bytes(riscv_width(insn), "little") for insn in insns)


def _link_call(insn: RawInsn) -> bool:
    """Standard RV64 JAL/JALR link destinations, independent of accelerator ISA."""
    word = int(insn.hexcode, 16)
    if riscv_width(insn) == 2:
        # C.JALR (C.JAL is RV32-only). Zero rs1 would instead be C.EBREAK.
        return word & 0xF07F == 0x9002 and (word >> 7) & 31 != 0
    return word & 0x7F in (0x6F, 0x67) and (word >> 7) & 31 in (1, 5)


def listing_regions(insns: Sequence[RawInsn]) -> tuple[ListingRegion, ...]:
    """Recover listed call intervals; their order does not establish executions.

    A body closes when the listing resumes at its caller's exact return PC. A
    terminal outer `ret` closes only the outer body. Incomplete captures remain
    explicitly open. No callee name, graph operation or source shape is guessed.
    """
    if not insns:
        return ()
    rows = [{"parent": None, "entry_pc": insns[0].addr, "call_index": None,
             "return_pc": None, "indices": [], "closed": False}]
    stack = [0]
    for index, insn in enumerate(insns):
        while len(stack) > 1 and insn.addr == rows[stack[-1]]["return_pc"]:
            rows[stack.pop()]["closed"] = True
        rows[stack[-1]]["indices"].append(index)
        if _link_call(insn):
            return_pc = insn.addr + riscv_width(insn)
            next_pc = insns[index + 1].addr if index + 1 < len(insns) else None
            included = next_pc is not None and next_pc != return_pc
            rows.append({"parent": stack[-1], "entry_pc": next_pc if included else None,
                         "call_index": index, "return_pc": return_pc,
                         "indices": [], "closed": False})
            if included:
                stack.append(len(rows) - 1)
    if len(stack) == 1 and _is_return(insns[-1]):
        rows[0]["closed"] = True
    return tuple(ListingRegion(region_id=i, **{**row, "indices": tuple(row["indices"])})
                 for i, row in enumerate(rows))


def _is_return(insn: RawInsn) -> bool:
    word = int(insn.hexcode, 16)
    return word in (0x8082, 0x8282, 0x00008067, 0x00028067)


def _direct_jump(insn: RawInsn) -> bool:
    return insn.mnemonic in ("j", "c.j") or (insn.mnemonic == "jal" and not _link_call(insn))


_CONDITIONAL = frozenset(("c.beqz", "c.bnez", "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
                           "blez", "bgez", "bltz", "bgtz", "bgt", "ble", "bgtu", "bleu"))


def listed_backedges(insns: Sequence[RawInsn]) -> tuple[dict[str, Any], ...]:
    """Explicit RV64 branch targets within a body, never inferred trip counts."""
    addresses = {insn.addr for insn in insns}
    out = []
    for insn in insns:
        if not (insn.mnemonic in _CONDITIONAL or _direct_jump(insn)) or not insn.operands:
            continue
        try:
            target = int(insn.operands[-1].split()[0], 16)
        except ValueError:
            continue
        if target <= insn.addr and target in addresses:
            out.append({"from_pc": insn.addr, "to_pc": target,
                        "conditional": insn.mnemonic in _CONDITIONAL,
                        "trip_count": None, "basis": "listed_branch_target"})
    return tuple(out)


def inventory(insns: Sequence[RawInsn], *, table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Structural observations plus optional selected-fact decoder projection.

    The supplied table is a conditional interpretation; the caller must pin it
    and separately establish its identity with the trace producer's target.
    Counts are listed instruction sites, never dynamic commands or elapsed time.
    """
    seen: dict[int, str] = {}
    conflicts, repeats = [], 0
    for index, insn in enumerate(insns):
        if insn.addr in seen:
            repeats += 1
            if seen[insn.addr] != insn.hexcode:
                conflicts.append(index)
        seen[insn.addr] = insn.hexcode
    regions = []
    for region in listing_regions(insns):
        body = [insns[index] for index in region.indices]
        row = region.to_dict()
        row.update({"listed_instructions": len(body), "mnemonics": dict(Counter(i.mnemonic for i in body)),
                    "backedges": list(listed_backedges(body))})
        if table:
            row["selected_decode"] = audit_decoded(decode_stream(body, table), table=table).to_dict()
        regions.append(row)
    return {"schema": "merlin.pc_instruction_listing.v1", "listed_instructions": len(insns),
            "unique_pcs": len(seen), "repeated_pc_rows": repeats, "conflicting_word_rows": conflicts,
            "regions": regions, "count_basis": "listed_instruction_sites",
            "unknown": ["producer_capture_semantics", "dynamic_loop_multiplicity", "register_values",
                        "memory_values", "cycle_values", "exact_source_and_target_binding"]}


def control_flow(insns: Sequence[RawInsn]) -> dict[str, Any]:
    """RV64 listing CFG and dominance-proven natural loops, without trip counts.

    Edges come from explicit decoded branch operands, not listing adjacency at
    call expansion boundaries. Unresolved indirect transfers remain gaps. A
    backedge forms a natural loop only when its header dominates its latch.
    """
    if not insns:
        return {"blocks": [], "natural_loops": [], "unresolved": []}
    if any(not insn.hexcode for insn in insns):
        return {"blocks": [], "natural_loops": [], "unresolved": ["instruction_encodings_missing"]}
    by_pc = {insn.addr: insn for insn in insns}
    if len(by_pc) != len(insns):
        return {"blocks": [], "natural_loops": [], "unresolved": ["repeated_pc_body"]}
    ordered = sorted(by_pc)
    leaders = {ordered[0], insns[0].addr}
    leaders.update(right for left, right in zip(ordered, ordered[1:])
                   if right != left + riscv_width(by_pc[left]))
    edges: dict[int, list[int]] = {}
    unresolved = []
    terminal = set()
    for pc in ordered:
        insn = by_pc[pc]
        following = pc + riscv_width(insn)
        next_edges = [following] if following in by_pc else []
        branch = insn.mnemonic in _CONDITIONAL or _direct_jump(insn)
        if branch:
            try:
                target = int(insn.operands[-1].split()[0], 16)
            except (ValueError, IndexError):
                target = None
            next_edges = next_edges if insn.mnemonic in _CONDITIONAL else []
            if target in by_pc:
                next_edges.append(target)
                leaders.add(target)
            else:
                unresolved.append({"pc": pc, "reason": "branch_target_outside_or_unresolved"})
            terminal.add(pc)
        elif _is_return(insn) or (insn.mnemonic in ("ret", "jr", "c.jr", "jalr") and not _link_call(insn)):
            next_edges = []
            terminal.add(pc)
            if not _is_return(insn):
                unresolved.append({"pc": pc, "reason": "indirect_transfer"})
        if pc in terminal and following in by_pc:
            leaders.add(following)
        edges[pc] = next_edges
    blocks: dict[int, list[int]] = {}
    owner = {}
    current = ordered[0]
    for pc in ordered:
        if pc in leaders or not blocks:
            current = pc
            blocks[current] = []
        blocks[current].append(pc)
        owner[pc] = current
    successors = {key: sorted({owner[target] for target in edges[pcs[-1]]}) for key, pcs in blocks.items()}
    predecessors = {key: [] for key in blocks}
    for source, targets in successors.items():
        for target in targets:
            predecessors[target].append(source)
    # Iterative postorder avoids recursion limits for large generated kernels.
    entry = insns[0].addr
    visited, postorder = set(), []
    pending = [(entry, False)]
    while pending:
        node, expanded = pending.pop()
        if expanded:
            postorder.append(node)
        elif node not in visited:
            visited.add(node)
            pending.append((node, True))
            pending.extend((target, False) for target in reversed(successors[node]) if target not in visited)
    rpo = list(reversed(postorder))
    rank = {key: i for i, key in enumerate(rpo)}
    idom = {entry: entry}

    def intersect(left, right):
        while left != right:
            while rank[left] > rank[right]:
                left = idom[left]
            while rank[right] > rank[left]:
                right = idom[right]
        return left

    changed = True
    while changed:
        changed = False
        for node in rpo[1:]:
            incoming = [pred for pred in predecessors[node] if pred in idom]
            if not incoming:
                continue
            candidate = incoming[0]
            for pred in incoming[1:]:
                candidate = intersect(candidate, pred)
            if idom.get(node) != candidate:
                idom[node] = candidate
                changed = True

    def dominates(header, node):
        while node in idom:
            if node == header:
                return True
            if idom[node] == node:
                return False
            node = idom[node]
        return False

    loops = []
    for latch in rpo:
        for header in successors[latch]:
            if not dominates(header, latch):
                continue
            members, todo = {header, latch}, [latch] if latch != header else []
            while todo:
                node = todo.pop()
                for pred in predecessors[node]:
                    if pred in visited and pred not in members:
                        members.add(pred)
                        todo.append(pred)
            loops.append({"header_pc": header, "latch_pc": latch, "block_pcs": sorted(members),
                          "trip_count": None, "basis": "cfg_header_dominates_latch"})
    return {"blocks": [{"pc": key, "last_pc": pcs[-1], "listed_instructions": len(pcs),
                        "successors": successors[key], "reachable_from_entry": key in visited,
                        "immediate_dominator": idom.get(key)} for key, pcs in blocks.items()],
            "natural_loops": loops, "unresolved": unresolved}


def match_symbol_bodies(insns: Sequence[RawInsn], regions: Sequence[ListingRegion], *,
                        symbols: Mapping[str, Sequence[RawInsn]], candidate_sha256: str) -> tuple[dict[str, Any], ...]:
    """Match complete contiguous body bytes; preserve every identical symbol.

    Candidate symbols/extent and ELF pin must come from the selected ELF reader.
    Names are reported as evidence and never dispatch compilation. A function with
    a different body cannot be promoted by a shared prefix or mnemonic pattern.
    """
    import hashlib

    if (len(candidate_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in candidate_sha256)):
        raise ValueError("candidate requires an exact lowercase SHA256")
    indexed: dict[str, list[dict[str, Any]]] = {}
    for name, body in symbols.items():
        if not body or any(right.addr != left.addr + riscv_width(left) for left, right in zip(body, body[1:])):
            continue
        data = instruction_bytes(body)
        indexed.setdefault(hashlib.sha256(data).hexdigest(), []).append(
            {"symbol": name, "entry_pc": body[0].addr, "size_bytes": len(data),
             "candidate_sha256": candidate_sha256})
    out = []
    for region in regions:
        body = [insns[index] for index in region.indices]
        contiguous = all(right.addr == left.addr + riscv_width(left) for left, right in zip(body, body[1:]))
        body_hash = hashlib.sha256(instruction_bytes(body)).hexdigest()
        out.append({"region_id": region.region_id, "body_sha256": body_hash,
                    "contiguous_body": contiguous,
                    "matches": indexed.get(body_hash, []) if contiguous else [],
                    "basis": "complete_listed_body_bytes", "whole_elf_identity": "unknown"})
    return tuple(out)


def pc_relative_constants(insns: Sequence[RawInsn]) -> tuple[dict[str, Any], ...]:
    """Prove adjacent RV64 AUIPC+ADDI values without replaying missing state.

    These are instruction-local address constants, not loaded data or an ABI
    assignment. Any intervening instruction, register mismatch or other opcode
    declines the pair. Standard RISC-V field layout supplies the arithmetic.
    """
    out = []
    for first, second in zip(insns, insns[1:]):
        a, b = int(first.hexcode, 16), int(second.hexcode, 16)
        rd = (a >> 7) & 31
        if (riscv_width(first) != 4 or riscv_width(second) != 4 or second.addr != first.addr + 4
                or a & 127 != 0x17 or b & 0x707F != 0x13 or rd == 0
                or (b >> 7) & 31 != rd or (b >> 15) & 31 != rd):
            continue
        high, low = a & 0xFFFFF000, b >> 20
        high = high - (1 << 32) if high & (1 << 31) else high
        low = low - (1 << 12) if low & (1 << 11) else low
        out.append({"first_pc": first.addr, "second_pc": second.addr, "register": rd,
                    "value": (first.addr + high + low) % (1 << 64),
                    "basis": "adjacent_rv64_auipc_addi", "pointee_value": None})
    return tuple(out)


def normalized_text(insns: Iterable[RawInsn]) -> str:
    """Existing objdump-tokenizer input, retaining listed words and addresses."""
    return "".join(f"{insn.addr:x}: {insn.hexcode} {insn.mnemonic} {','.join(insn.operands)}\n"
                   for insn in insns)


def lift_schedule(insns: Sequence[RawInsn], *, table: Mapping[str, Any], evidence):
    """Project scoped listings through the established static ScheduleModel path.

    Machine register identifiers remain identifiers; missing values never become
    zero operands. The CCA projection for static disassembly deliberately cannot
    authorize dynamic dispatch counts, reuse, completion or elapsed timing.
    """
    from dataclasses import replace

    from merlin.kernels.kernel_evidence import analyze_disassembly

    model = analyze_disassembly(normalized_text(insns), table=table, evidence=evidence)
    return replace(model, complete=False, unknowns=tuple(sorted({*model.unknowns,
        "listing_multiplicity_not_established", "selected_isa_correspondence_not_established",
        "register_and_memory_values_not_supplied"})))
