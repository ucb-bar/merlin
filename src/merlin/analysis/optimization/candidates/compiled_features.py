"""Bounded emitted sites and linked bytes, without physical cost inference.

The existing typed LLVM observer owns the supported complete body grammar.
Counts describe that IR before native optimization, not executed instructions,
traffic, allocations or time. Unsupported work is never silently discarded.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from dataclasses import asdict, dataclass

from merlin.development.targetgen.grading.elf_lanes import executable_sections
from merlin.target.contract.build.emitted_code import observe_emitted_dataflow

# ---- compiled_static_features ------------------------------------------------------------------

SCHEMA = "merlin.compiled_static_features.v1"


def linked_entry_extent(elf_bytes, entry_symbol):
    """Read one defined function from the existing bounded ELF64 parser.

    This establishes a linked extent only, never execution or semantic coverage.
    Unsupported ELF formats and stripped/ambiguous definitions refuse.
    """

    _, sections = _read(elf_bytes)
    definitions = set()
    for table in sections:
        if table.kind not in (2, 11):
            continue
        if table.entry_size != 24 or len(table.data) % 24 or not 0 <= table.link < len(sections):
            raise ValueError("unsupported ELF function symbol table")
        strings = sections[table.link]
        if strings.kind != 3:
            raise ValueError("ELF function symbols have no string table")
        for offset in range(0, len(table.data), 24):
            name, info, _, owner, address, size = struct.unpack_from("<IBBHQQ", table.data, offset)
            if info & 15 == 2 and 0 < owner < len(sections) and _string(strings.data, name) == entry_symbol:
                definitions.add((address, size, owner))
    if len(definitions) != 1:
        raise ValueError("ELF has no unique retained function definition")
    address, size, owner = next(iter(definitions))
    section = sections[owner]
    if (
        not section.flags & 4
        or not section.name
        or sum(item.name == section.name for item in sections) != 1
        or address < section.address
        or address + size > section.address + section.size
    ):
        raise ValueError("ELF function extent has no unique executable owner section")
    return {
        "symbol": entry_symbol,
        "address": address,
        "size_bytes": size,
        "section": section.name,
        "scope": "static linked definition",
    }


@dataclass(frozen=True)
class StaticFeatureLimits:
    max_source_bytes: int
    max_elf_bytes: int
    max_operations: int

    def verify(self):
        for value, ceiling in zip(asdict(self).values(), (8 << 20, 64 << 20, 100000), strict=True):
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError("static features need explicit bounded source/ELF/operation limits")


def derive_compiled_static_features(*, llvm_mlir, elf_bytes, entry_symbol, pointer_bits, retained_entry, limits):
    """Observe a complete supported emitted body and structural ELF sections.

    ``pointer_bits`` is an explicit observation parameter; this function does
    not establish the compiler's pointer ABI. The caller must independently
    join these bytes and the entry extent to the actual compilation products.
    No shaped operands, references, engine, instruction decoder or cost model
    is loaded. Malformed or unsupported input raises instead of returning zero.
    """
    if type(limits) is not StaticFeatureLimits:
        raise ValueError("static features require exact typed limits")
    limits.verify()
    if (
        type(llvm_mlir) is not str
        or len(llvm_mlir) > limits.max_source_bytes
        or len(llvm_mlir.encode("utf-8")) > limits.max_source_bytes
        or type(elf_bytes) is not bytes
        or len(elf_bytes) > limits.max_elf_bytes
    ):
        raise ValueError("static feature source or ELF exceeds its byte limit")
    dataflow = observe_emitted_dataflow(
        llvm_mlir, entry_symbol=entry_symbol, pointer_bits=pointer_bits, max_operations=limits.max_operations
    )
    sections = executable_sections(elf_bytes)
    if (
        type(retained_entry) is not dict
        or set(retained_entry) != {"symbol", "address", "size_bytes", "scope"}
        or retained_entry["symbol"] != entry_symbol
        or retained_entry["scope"] != "static linked definition"
        or type(retained_entry["address"]) is not int
        or retained_entry["address"] < 0
        or type(retained_entry["size_bytes"]) is not int
        or retained_entry["size_bytes"] <= 0
        or not any(
            address <= retained_entry["address"]
            and retained_entry["address"] + retained_entry["size_bytes"] <= address + size
            for _, _, size, address in sections
        )
    ):
        raise ValueError("static feature entry extent is outside the complete executable section roster")
    memory = {}
    for kind in ("load", "store"):
        actions = [row for row in dataflow.actions if row.kind == kind]
        widths = [dataflow.value(row.results[0] if kind == "load" else row.operands[0]).bits for row in actions]
        memory[kind] = {
            "sites": len(actions),
            "declared_access_bits": sum(widths),
            "width_sites": dict(sorted(Counter(str(width) for width in widths).items())),
            "alignment_sites": dict(
                sorted(
                    Counter("undeclared" if row.alignment is None else str(row.alignment) for row in actions).items()
                )
            ),
            "volatile_sites": sum(row.volatile is True for row in actions),
        }
    return {
        "schema": SCHEMA,
        "limits": asdict(limits),
        "llvm_mlir_sha256": dataflow.source_sha256,
        "pointer_bits_selection": pointer_bits,
        "entry_symbol": entry_symbol,
        "emitted": {
            "value_sites": dict(sorted(Counter(row.kind for row in dataflow.values).items())),
            "action_sites": dict(sorted(Counter(row.kind for row in dataflow.actions).items())),
            "memory": memory,
            "pointer_transform_sites": sum(
                row.kind in {"llvm.ptrtoint", "llvm.inttoptr", "llvm.bitcast"} for row in dataflow.values
            ),
            "opaque_assembly_sites": sum(row.kind == "assembly" for row in dataflow.actions),
        },
        "linked": {
            "executable_sections": [
                {"name": name, "file_offset": offset, "size_bytes": size, "address": address}
                for name, offset, size, address in sections
            ],
            "executable_bytes": sum(size for _, _, size, _ in sections),
            "retained_entry": dict(retained_entry),
        },
        "unknown": [
            "source_equivalence",
            "pointer_abi_and_layout",
            "instruction_semantics_and_count",
            "dynamic_execution_counts",
            "physical_memory_traffic",
            "staging_allocation_and_capacity",
            "overlap_and_lifetime",
            "runtime_and_observer_integrity",
            "hardware_and_timer_domain",
            "cold_warm_costs",
            "calibration_and_held_validation",
        ],
        "authority": "none",
        "scope": "static emitted sites before native optimization and linked code extents only",
    }


# ---- debug_companion ---------------------------------------------------------------------------
# Admit debug-only ELF companions before attributing an existing PC census.
#
# The caller supplies target tools, PC counts and symbolizer output. No execution,
# ISA decoding, compiler invocation or timing inference occurs here. The current
# binary reader deliberately supports ordinary little-endian ELF64 only.


@dataclass(frozen=True)
class _Section:
    name: str
    kind: int
    flags: int
    address: int
    size: int
    link: int
    info: int
    alignment: int
    entry_size: int
    data: bytes


def _slice(data: bytes, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError("ELF range outside file")
    return data[offset : offset + size]


def _string(data: bytes, offset: int) -> str:
    if offset >= len(data) or b"\0" not in data[offset:]:
        raise ValueError("invalid ELF string")
    return data[offset : data.index(b"\0", offset)].decode("utf-8", errors="strict")


def _read(data: bytes):
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("ordinary little-endian ELF64 required")
    header = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    kind, machine, version, entry, _, offset, flags, size, _, _, stride, count, names = header
    if version != 1 or size != 64 or stride != 64 or not count or not 0 < names < count:
        raise ValueError("unsupported ELF header or extended section numbering")
    rows = [struct.unpack("<IIQQQQIIQQ", _slice(data, offset + i * stride, stride)) for i in range(count)]
    table = _slice(data, rows[names][4], rows[names][5])
    sections = []
    for row in rows:
        name, typ, section_flags, address, start, length, link, info, align, entsize = row
        body = b"" if typ == 8 else _slice(data, start, length)
        sections.append(
            _Section(_string(table, name), typ, section_flags, address, length, link, info, align, entsize, body)
        )
    return (kind, machine, entry, flags), sections


def _allocated(sections):
    result = {}
    for section in sections:
        if section.flags & 2:
            if section.name in result:
                raise ValueError("duplicate allocated section name")
            result[section.name] = (
                section.kind,
                section.flags,
                section.address,
                section.size,
                section.alignment,
                section.data,
            )
    if not result:
        raise ValueError("ELF has no allocated sections")
    return result


def _relocations(sections):
    result = []
    for reloc in sections:
        if reloc.kind not in (4, 9):
            continue
        if reloc.info >= len(sections) or reloc.link >= len(sections):
            raise ValueError("invalid relocation section indices")
        destination = sections[reloc.info]
        if not destination.flags & 2:
            continue
        symbols = sections[reloc.link]
        if symbols.kind not in (2, 11) or symbols.entry_size != 24 or symbols.link >= len(sections):
            raise ValueError("invalid relocation symbol table")
        if sections[symbols.link].kind != 3 or len(symbols.data) % 24:
            raise ValueError("invalid relocation symbol strings or entries")
        strings = sections[symbols.link].data
        stride = 24 if reloc.kind == 4 else 16
        if reloc.entry_size != stride or len(reloc.data) % stride:
            raise ValueError("invalid relocation entries")
        for offset in range(0, len(reloc.data), stride):
            place, info = struct.unpack_from("<QQ", reloc.data, offset)
            addend = struct.unpack_from("<q", reloc.data, offset + 16)[0] if stride == 24 else None
            symbol_index, relocation_type = info >> 32, info & 0xFFFFFFFF
            raw = _slice(symbols.data, symbol_index * 24, 24)
            name, symbol_info, other, section_index, value, size = struct.unpack("<IBBHQQ", raw)
            if len(sections) <= section_index < 0xFF00:
                raise ValueError("symbol section index outside section table")
            owner = sections[section_index].name if section_index < len(sections) else section_index
            if section_index == 0xFFFF:
                raise ValueError("extended symbol section indices unsupported")
            symbol = (_string(strings, name), symbol_info, other, owner, value, size)
            result.append((destination.name, place, relocation_type, addend, symbol))
    return Counter(result)


def verify_debug_companion(control: bytes, companion: bytes, *, relocatable: bool = False) -> dict:
    """Require identical allocated data/address/flags and normalized relocations.

    Relocatable objects additionally require ELF ET_REL. Final images require
    ET_EXEC or ET_DYN. This does not establish source correctness, original
    numerical gates or equivalence of an unrelated compiler command.
    """
    if type(relocatable) is not bool:
        raise ValueError("explicit boolean object policy required")
    a_header, a = _read(control)
    b_header, b = _read(companion)
    if a_header != b_header or a_header[0] not in ((1,) if relocatable else (2, 3)):
        raise ValueError("ELF identity differs or wrong file kind")
    if _allocated(a) != _allocated(b):
        raise ValueError("allocated section bytes, address or attributes differ")
    if _relocations(a) != _relocations(b):
        raise ValueError("allocated-section relocations differ")
    return {
        "control_sha256": hashlib.sha256(control).hexdigest(),
        "companion_sha256": hashlib.sha256(companion).hexdigest(),
        "allocated_sections": len(_allocated(a)),
        "relocations": sum(_relocations(a).values()),
        "relocatable": relocatable,
    }


def attribute_symbolized_pcs(counts: dict[int, int], records: list[dict]) -> dict:
    """Aggregate LLVM symbolizer JSON records with complete address coverage.

    Every PC is accounted once, including records without source lines. Inline
    frames are retained as a stack; innermost aggregation is not a call timer.
    The caller must first verify the exact binary companion and census identity.
    """
    if any(type(pc) is not int or pc < 0 or type(n) is not int or n < 0 for pc, n in counts.items()):
        raise ValueError("nonnegative integer PCs and counts required")
    seen = set()
    functions, lines, stacks = Counter(), Counter(), Counter()
    for record in records:
        pc = int(record["Address"], 0)
        if pc not in counts or pc in seen:
            raise ValueError("unexpected or duplicate symbolized PC")
        seen.add(pc)
        frames = record["Symbol"]
        stack = tuple(
            (f.get("FunctionName", ""), f.get("FileName", ""), f.get("Line", 0), f.get("Column", 0)) for f in frames
        )
        if any(
            type(line) is not int or line < 0 or type(column) is not int or column < 0 for _, _, line, column in stack
        ):
            raise ValueError("invalid source coordinate")
        function, filename, line, _ = stack[0] if stack else ("", "", 0, 0)
        functions[function] += counts[pc]
        lines[(filename, line)] += counts[pc]
        stacks[stack] += counts[pc]
    if seen != counts.keys():
        raise ValueError("incomplete symbolizer coverage")
    return {
        "total": sum(counts.values()),
        "functions": dict(functions),
        "lines": dict(lines),
        "inline_stacks": dict(stacks),
    }
