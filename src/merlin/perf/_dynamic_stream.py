"""Online descriptive Spike commit aggregation over one verified ELF entry.

Only lexical entry PCs are attributed. Outside-entry activity is retained as
unattributed; callee work, numerical behavior and all physical costs are UNKNOWN.
"""

from __future__ import annotations

from collections import Counter

from merlin.perf.compiled_static_features import linked_entry_extent
from merlin.targetgen.elf_lanes import executable_sections
from merlin.targetgen.rocc import semantics


class CensusUnavailable(ValueError):
    """An incomplete or unsupported census must never expose partial counts."""


class CensusCleanupUnknown(BaseException):
    """Abort the enclosing job while its real descendant owner remains live."""


def _hex(value):
    token = value.removeprefix("(").removesuffix(")")
    if not token.startswith("0x"):
        raise CensusUnavailable("unsupported committed hexadecimal value")
    try:
        number = int(token, 16)
    except ValueError as error:
        raise CensusUnavailable("malformed committed hexadecimal value") from error
    if not 0 <= number < 1 << 64:
        raise CensusUnavailable("committed value exceeds supported register width")
    return number


class OnlineCensus:
    def __init__(self, *, elf_bytes, entry_symbol, isa, local_fields, limits):
        limits.verify()
        if len(elf_bytes) > limits.max_elf_bytes:
            raise CensusUnavailable("linked ELF exceeds census byte budget")
        self.entry = linked_entry_extent(elf_bytes, entry_symbol)
        address, size = self.entry["address"], self.entry["size_bytes"]
        if type(size) is not int or size <= 0:
            raise CensusUnavailable("linked entry has no finite retained extent")
        sections = [
            (offset, start)
            for _, offset, length, start in executable_sections(elf_bytes)
            if start <= address and address + size <= start + length
        ]
        if len(sections) != 1:
            raise CensusUnavailable("linked entry is outside its unique executable section")
        offset, start = sections[0]
        self.code = elf_bytes[offset + address - start : offset + address - start + size]
        self.opcode = isa.get("CUSTOM_OPCODE")
        if type(self.opcode) is not int or not 0 <= self.opcode < 128:
            raise CensusUnavailable("selected RTL facts supply no custom transport opcode")
        self.isa, self.local_fields, self.limits = isa, tuple(local_fields), limits
        if len(self.local_fields) > limits.max_bins:
            raise CensusUnavailable("local-address declarations exceed census state budget")
        self.regs, self.retired, self.inside = {}, 0, 0
        self.host, self.commands, self.requests, self.memory, self.local = (Counter() for _ in range(5))
        self.accelerator_memory = Counter()
        self.bins, self.state_bytes = 0, 0

    def _add(self, counter, key):
        if key not in counter:
            self.bins += 1
            # Bound each stored key and the complete serialized census state.
            # Process RSS is separately bounded by the selected process owner.
            self.state_bytes += len(str(key).encode()) + 512
            if self.bins > self.limits.max_bins or self.state_bytes > self.limits.max_state_bytes:
                raise CensusUnavailable("online census exceeds its state budget")
        counter[key] += 1

    def line(self, raw):
        if len(raw.encode("utf-8")) > self.limits.max_line_bytes:
            raise CensusUnavailable("commit line exceeds census line budget")
        parts = raw.split()
        if not parts or parts[0] != "core":
            return
        if len(parts) < 3:
            raise CensusUnavailable("malformed core trace record")
        if parts[2].startswith("0x"):
            return  # The paired disassembly line is not another retirement.
        if len(parts) < 5 or not parts[2].isascii() or not parts[2].isdigit():
            raise CensusUnavailable("unsupported committed core record")
        if not parts[1].endswith(":") or not parts[1][:-1].isascii() or not parts[1][:-1].isdigit():
            raise CensusUnavailable("malformed commit hart identity")
        hart = int(parts[1][:-1])
        if hart not in self.regs:
            if len(self.regs) >= self.limits.max_harts:
                raise CensusUnavailable("trace exceeds selected hart state budget")
            self.regs[hart] = [0, *([None] * 31)]
        registers = self.regs[hart]
        pc, instruction = _hex(parts[3]), _hex(parts[4])
        self.retired += 1
        if self.retired > self.limits.max_retired:
            raise CensusUnavailable("trace exceeds retired-instruction budget")
        at = pc - self.entry["address"]
        inside = 0 <= at < self.entry["size_bytes"]
        custom = False
        if inside:
            # These are the standard RISC-V variable-width transport bits,
            # shared with executed_trace; device opcode values come from ISA.
            width = 4 if instruction & 3 == 3 else 2
            if instruction & 31 == 31 or at % 2 or at + width > len(self.code):
                raise CensusUnavailable("unsupported or out-of-range committed entry instruction")
            if instruction != int.from_bytes(self.code[at : at + width], "little"):
                raise CensusUnavailable("committed entry bytes disagree with selected linked ELF")
            self.inside += 1
            if width == 4 and instruction & 127 == self.opcode:
                custom = True
                first, second = registers[(instruction >> 15) & 31], registers[(instruction >> 20) & 31]
                if first is None or second is None:
                    raise CensusUnavailable("executed command has an unobserved source register")
                funct = instruction >> 25
                kind, decoded = semantics.decode_instruction(
                    funct, {"raw": first, "kind": "const"}, {"raw": second, "kind": "const"}, self.isa
                )
                if type(kind) is not str or not kind or kind == "UNKNOWN" or len(kind.encode()) > 128:
                    raise CensusUnavailable("executed command has unsupported selected ISA semantics")
                self._add(self.commands, kind)
                self._add(self.requests, (kind, funct, first, second))
                for classes, field, rows_field in self.local_fields:
                    if kind not in classes:
                        continue
                    value, rows = decoded.get(field), decoded.get(rows_field) if rows_field else None
                    if type(value) is not int or not 0 <= value < 1 << 64:
                        raise CensusUnavailable("local-address operand could not be decoded")
                    if rows_field and (type(rows) is not int or not 0 <= rows < 1 << 64):
                        raise CensusUnavailable("local-address extent could not be decoded")
                    self._add(self.local, (kind, field, value, rows_field, rows))
            else:
                self._add(self.host, pc)
        writes = iter(parts[5:])
        for name in writes:
            if name == "mem":
                address = _hex(next(writes, ""))
                if inside:
                    self._add(self.accelerator_memory if custom else self.memory, address)
            elif name.startswith("x"):
                number = next(writes, "") if name == "x" else name[1:]
                if not number.isascii() or not number.isdigit() or not 0 <= int(number) < 32:
                    raise CensusUnavailable("unsupported integer-register write")
                value = _hex(next(writes, ""))
                if int(number):
                    registers[int(number)] = value
                elif value:
                    raise CensusUnavailable("committed zero-register write is inconsistent")

    def finish(self):
        if not self.retired or not self.inside:
            raise CensusUnavailable("complete trace did not retire the verified linked entry")
        return {
            "entry": {**self.entry, "instruction_bytes_verified": True},
            "retired_instructions": {
                "whole_trace": self.retired,
                "entry": self.inside,
                "outside_entry": self.retired - self.inside,
            },
            "host": {
                "retired_in_entry": sum(self.host.values()),
                "by_pc": [{"pc": pc, "count": count} for pc, count in sorted(self.host.items())],
                "scope": "verified lexical entry PCs only; harness/setup/readback excluded",
                "outside_entry_callee_work": "UNKNOWN",
            },
            "accelerator": {
                "commands_in_entry": sum(self.commands.values()),
                "by_class": dict(sorted(self.commands.items())),
                "operand_requests": [
                    {"class": kind, "funct": funct, "rs1": first, "rs2": second, "count": count}
                    for (kind, funct, first, second), count in sorted(self.requests.items())
                ],
            },
            "host_memory_requests": {
                "requests_in_entry": sum(self.memory.values()),
                "by_address": [{"address": address, "count": count} for address, count in sorted(self.memory.items())],
                "physical_bytes": None,
            },
            "accelerator_memory_requests": {
                "requests_in_entry": sum(self.accelerator_memory.values()),
                "by_address": [
                    {"address": address, "count": count} for address, count in sorted(self.accelerator_memory.items())
                ],
                "scope": "only requests present in commit records; total and physical traffic UNKNOWN",
            },
            "local_address_activity": {
                "status": "observed" if self.local_fields else "UNKNOWN",
                "requests": [
                    {
                        "class": kind,
                        "address_field": field,
                        "raw_address": value,
                        "rows_field": rows_field,
                        "raw_rows": rows,
                        "count": count,
                    }
                    for (kind, field, value, rows_field, rows), count in sorted(self.local.items())
                ],
                "scope": "decoded requested addresses/extents; allocation and physical traffic UNKNOWN",
            },
        }
