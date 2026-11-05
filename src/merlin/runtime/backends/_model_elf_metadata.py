"""Read hart, memory and ISA declarations from bare-metal ELF metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path
from ...llvmlower import toolchain


DRAM_BASE_SYMBOL = "MERLIN_DRAM_BASE"

DRAM_SPAN_SYMBOL = "MERLIN_DRAM_SPAN"

HART_COUNT_SYMBOL = "MERLIN_HART_COUNT"


def declared_harts(elf: str | Path) -> int | None:
    """The hart count an image states with :data:`HART_COUNT_SYMBOL`; ``None`` when it states none."""
    listing = subprocess.run(
        [str(toolchain.nm()), "--defined-only", "--radix=d", str(elf)], capture_output=True, text=True
    )
    for line in listing.stdout.splitlines() if listing.returncode == 0 else ():
        parts = line.split()
        if len(parts) == 3 and parts[2] == HART_COUNT_SYMBOL and parts[0].isdigit():
            return int(parts[0])
    return None


def _extension_name(token: str) -> str:
    """``zvl128b1p0`` -> ``zvl128b``, ``m2p0`` -> ``m``: an ELF arch token without its version."""
    end = len(token)
    while end and token[end - 1].isdigit():
        end -= 1
    if end and token[end - 1] == "p" and end < len(token):
        cut = end - 1
        while cut and token[cut - 1].isdigit():
            cut -= 1
        return token[:cut]
    return token


def arch_extensions(path: str | Path) -> list[str]:
    """The ISA an ELF (an image or one object) records it was compiled for (``Tag_RISCV_arch``), as
    ``[base+first letter, extension, ...]`` without versions (``["rv64i", "m", ..., "v", "zvl128b"]``);
    empty when it records none."""
    listing = subprocess.run([str(toolchain.readelf()), "-A", str(path)], capture_output=True, text=True)
    arch, named = None, False
    for line in listing.stdout.splitlines() if listing.returncode == 0 else ():
        key, _, value = line.strip().partition(":")
        if key == "TagName":
            named = value.strip() == "arch"
        elif key == "Value" and named:
            arch, named = value.strip(), False
    return [_extension_name(t) for t in (arch or "").split("_") if t]


def declared_memory(elf: str | Path) -> tuple[int, int] | None:
    """``(base, bytes)`` the image states with :data:`DRAM_BASE_SYMBOL` / :data:`DRAM_SPAN_SYMBOL`,
    read off its symbol table; ``None`` for an image that states none (the simulator's default span)."""
    listing = subprocess.run(
        [str(toolchain.nm()), "--defined-only", "--radix=d", str(elf)], capture_output=True, text=True
    )
    if listing.returncode != 0:
        return None
    found: dict[str, int] = {}
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in (DRAM_BASE_SYMBOL, DRAM_SPAN_SYMBOL) and parts[0].isdigit():
            found[parts[2]] = int(parts[0])
    if len(found) != 2:
        return None
    return found[DRAM_BASE_SYMBOL], found[DRAM_SPAN_SYMBOL]


def _supplemental_object_digest(objects):
    """Identity of actual linked device/matrix bytes, independent of work paths."""
    import hashlib

    digest = hashlib.sha256(b"merlin-supplemental-objects-v1")
    for path in objects:
        content = Path(path).read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.digest()
