"""Bounded emitted sites and linked bytes, without physical cost inference.

The existing typed LLVM observer owns the supported complete body grammar.
Counts describe that IR before native optimization, not executed instructions,
traffic, allocations or time. Unsupported work is never silently discarded.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import asdict, dataclass

from merlin.targetgen.contract.emitted_dataflow import observe_emitted_dataflow
from merlin.targetgen.elf_lanes import executable_sections

SCHEMA = "merlin.compiled_static_features.v1"


def linked_entry_extent(elf_bytes, entry_symbol):
    """Read one defined function from the existing bounded ELF64 parser.

    This establishes a linked extent only, never execution or semantic coverage.
    Unsupported ELF formats and stripped/ambiguous definitions refuse.
    """
    from merlin.perf.debug_companion import _read, _string

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
                definitions.add((address, size))
    if len(definitions) != 1:
        raise ValueError("ELF has no unique retained function definition")
    address, size = next(iter(definitions))
    return {"symbol": entry_symbol, "address": address, "size_bytes": size, "scope": "static linked definition"}


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
