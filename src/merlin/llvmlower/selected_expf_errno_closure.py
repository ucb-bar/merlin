"""Additional explicit errno fact for the already selected expf implementation.

This does not select a compiler feature. Every path of the reviewed archive's
expf either leaves errno alone or assigns ERANGE (34), including NaN/infinity
paths. The linked errno getter must separately be proved to return a fixed
pointer without effects. Existing per-region floating-environment, first-call
and ordinary source-side-effect guards remain required.
"""

from __future__ import annotations

import contextlib
import contextvars
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .selected_expf_equivalence import (
    ExpfEquivalenceError,
    ExpfUnderflowEquivalence,
    current_expf_underflow_equivalence,
    verify_expf_underflow_link,
)

PROTOCOL = "merlin.selected_expf_errno34_closure.v1"


@dataclass(frozen=True)
class ExpfErrnoClosure:
    """All original selected expf calls preserve an already established errno34.

    This fact is conditional on the final link verifier's pure fixed-pointer
    getter check. It permits no omission of an original first underflow call,
    no environment reset, and no intervening arbitrary call or errno write.
    """

    underflow: ExpfUnderflowEquivalence

    @property
    def preserved_errno(self) -> int:
        return 34

    def revalidate(self) -> None:
        if type(self) is not ExpfErrnoClosure or type(self.underflow) is not ExpfUnderflowEquivalence:
            raise ExpfEquivalenceError("expected the original typed expf errno closure")
        self.underflow.revalidate()


def select_expf_errno_closure(grant: ExpfUnderflowEquivalence) -> ExpfErrnoClosure:
    """Explicit extra permission; existing underflow selection alone is unchanged."""
    closure = ExpfErrnoClosure(grant)
    closure.revalidate()
    return closure


_CURRENT: contextvars.ContextVar[ExpfErrnoClosure | None] = contextvars.ContextVar(
    "merlin_selected_expf_errno_closure", default=None
)


@contextlib.contextmanager
def selected_expf_errno_closure_scope(closure: ExpfErrnoClosure) -> Iterator[None]:
    """Explicit compiler scope nested inside its exact original underflow scope."""
    if type(closure) is not ExpfErrnoClosure:
        raise ExpfEquivalenceError("expected the original typed expf errno closure")
    closure.revalidate()
    if current_expf_underflow_equivalence() is not closure.underflow:
        raise ExpfEquivalenceError("errno closure requires its exact live underflow scope")
    token = _CURRENT.set(closure)
    try:
        yield
        closure.revalidate()
    finally:
        _CURRENT.reset(token)


def current_expf_errno_closure() -> ExpfErrnoClosure | None:
    """Reopen the explicit additional permission at pipeline construction."""
    closure = _CURRENT.get()
    if closure is not None:
        closure.revalidate()
        if current_expf_underflow_equivalence() is not closure.underflow:
            raise ExpfEquivalenceError("errno closure lost its exact live underflow scope")
    return closure


def _sections(text: str, name: str) -> list[tuple[int, int, str]]:
    rows = []
    pending = False
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0].startswith(".text"):
            pending = words[0] == name
            words = words[1:]
        if pending and len(words) >= 3 and words[0].startswith("0x") and words[1].startswith("0x"):
            address, size = int(words[0], 16), int(words[1], 16)
            if address and size:
                rows.append((address, size, " ".join(words[2:])))
            pending = False
    return rows


def _virtual_bytes(data: bytes, address: int, size: int) -> bytes:
    """Read a small range from a file-backed ELF64 little-endian LOAD segment."""
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01" or data[18:20] != b"\xf3\x00":
        raise ExpfEquivalenceError("errno getter requires a complete ELF64 RISC-V output")
    offset = struct.unpack_from("<Q", data, 32)[0]
    entry_size, count = struct.unpack_from("<HH", data, 54)
    if entry_size != 56 or not 0 < count <= 256 or offset + entry_size * count > len(data):
        raise ExpfEquivalenceError("unsupported errno-getter ELF program headers")
    matches = []
    for i in range(count):
        kind, _, file_offset, virtual, _, file_size, _, _ = struct.unpack_from("<IIQQQQQQ", data, offset + 56 * i)
        if kind == 1 and virtual <= address and address + size <= virtual + file_size:
            start = file_offset + address - virtual
            if start + size > len(data):
                raise ExpfEquivalenceError("truncated errno-getter ELF LOAD segment")
            matches.append(data[start : start + size])
    if len(matches) != 1:
        raise ExpfEquivalenceError("errno getter lacks one exact file-backed LOAD range")
    return matches[0]


def _verify_getter(code: bytes) -> None:
    # AUIPC a0, imm20; ADDI a0, a0, imm12; C.JR ra. Immediates only
    # select the fixed address. No memory/FP/CSR access, branches or calls.
    if len(code) != 10:
        raise ExpfEquivalenceError("unsupported errno getter; expected a pure fixed-pointer return")
    first, second, last = struct.unpack("<IIH", code)
    if first & 0xFFF != 0x517 or second & 0xFFFFF != 0x50513 or last != 0x8082:
        raise ExpfEquivalenceError("unsupported errno getter; expected a pure fixed-pointer return")


def verify_expf_errno_closure_link(
    closure: ExpfErrnoClosure, *, elf: Path, map_path: Path
) -> dict[str, object]:
    """Reopen the actual link, all selected callees and its effect-free getter.

    Strictly unsupported getter encodings retain the ordinary errno-checked
    implementation. No serialized receipt can replace this live check.
    """
    if type(closure) is not ExpfErrnoClosure:
        raise ExpfEquivalenceError("expected the original typed expf errno closure")
    closure.revalidate()
    receipt = verify_expf_underflow_link(closure.underflow, elf=elf, map_path=map_path)
    text = Path(map_path).read_text()
    overflow = _sections(text, ".text.__math_oflowf")
    expected = f"{closure.underflow.library_path}(libm_a-math_errf.o)"
    if len(overflow) != 1 or overflow[0][2] != expected:
        raise ExpfEquivalenceError("selected expf overflow path has missing/foreign owner")
    getters = _sections(text, ".text.__errno")
    if len(getters) != 1:
        raise ExpfEquivalenceError("errno closure requires one linked getter section")
    address, size, owner = getters[0]
    code = _virtual_bytes(Path(elf).read_bytes(), address, size)
    _verify_getter(code)
    closure.revalidate()
    return {
        **receipt,
        "errno_closure_protocol": PROTOCOL,
        "errno_preserved_from": 34,
        "all_input_errno_store_values": [34],
        "overflow_owner": overflow[0][2],
        "getter_owner": owner,
        "getter_encoding": code.hex(),
        "getter_effects": "fixed-pointer return; no loads/stores/calls/FP/CSR effects",
    }
