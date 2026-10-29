"""Exact retained ELF copies with their original native link custody.

Relocation issues no new producer, numerical or scope evidence. Original native
invocations and all their inputs remain independently reopened before and after
observation. The destination only grants a bounded execution/output workspace.
"""

from dataclasses import dataclass
from pathlib import Path
import json

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F

from . import census_launch as BOX


@dataclass(frozen=True)
class NativeElfRelocation:
    original: dict
    destination: dict
    producer_records: tuple

    def producer(self, elf, limits):
        source = BOX._producer(Path(self.original["path"]), limits)
        destination = F.pin(elf, max_bytes=limits.max_elf_bytes)
        if source != self.original or destination != self.destination:
            raise ValueError("retained ELF relocation source or destination changed")
        if source["sha256"] != destination["sha256"]:
            raise ValueError("retained ELF relocation changed executable bytes")
        if not self.producer_records:
            raise ValueError("retained ELF relocation lacks original producer pins")
        for pin in self.producer_records:
            if F.pin(Path(pin["path"]), max_bytes=1 << 20) != pin:
                raise ValueError("retained ELF native producer record changed")
            I.verify(Path(pin["path"]))
        return destination

    def record(self):
        return {"schema": "merlin.native_elf_exact_relocation.v1", "original": self.original,
                "destination": self.destination, "new_native_producer": False,
                "original_native_producers": list(self.producer_records)}


def review_native_elf_relocation(original, destination, limits):
    """Review an explicit exact copy; never search for an equivalent producer."""
    source = BOX._producer(F.canonical(original), limits)
    copied = F.pin(destination, max_bytes=limits.max_elf_bytes)
    records = []
    for path in F.members(Path(source["path"]).parent / "invocations", max_files=limits.max_bins):
        if path.name != "invocation.json" or not path.is_file():
            continue
        pin = F.pin(path, max_bytes=1 << 20)
        row = json.loads(path.read_text())
        if (row.get("kind") == "subprocess" and row.get("stage") == "elf"
                and row.get("status") == "completed" and row.get("returncode") == 0
                and source in row.get("outputs", ())):
            I.verify(path)
            records.append(pin)
    result = NativeElfRelocation(source, copied, tuple(records))
    result.producer(destination, limits)
    return result
