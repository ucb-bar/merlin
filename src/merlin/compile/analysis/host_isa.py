"""Select the host ISA a disassembler decodes with, as recorded data with provenance.

An ELF without a ``.riscv.attributes`` section tells LLVM nothing about the extensions it was
built for, so ``llvm-objdump`` decodes only the base integer and compressed encodings and prints
every M/F/D/... word as ``<unknown>``. Those words are ordinary host instructions, not accelerator
commands, but a consumer that reads ``<unknown>`` as "undecodable" cannot tell them apart.

The decoder ISA is therefore selected, in order, from:

1. the ELF's own arch attribute (``Tag_RISCV_arch``), when present;
2. the selected target contract's declared host ISA: the functional simulator's ``--isa`` for
   the selected runner family, else that family's toolchain ``-march``;
3. otherwise nothing: the ISA stays ``unknown`` and no feature option is passed.

The ELF header's float-ABI/RVC flags only corroborate a selection; they never add an extension.
The ISA string is translated to ``--mattr`` structurally (``str`` operations, no regex), and the
whole selection is returned as a record so a reviewer can see where the value came from.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from typing import Any

__all__ = [
    "contract_host_isa",
    "elf_arch_attribute",
    "elf_flag_corroboration",
    "isa_features",
    "select_decoder_isa",
]

# ELF psABI format constants (the RISC-V ELF container, not any target's facts).
_SHT_RISCV_ATTRIBUTES = 0x70000003  # derived-ok: psABI section type of the build-attribute section
_ATTRIBUTE_FORMAT = ord("A")  # derived-ok: psABI attribute-section format version byte
_ATTRIBUTE_VENDOR = b"riscv"  # derived-ok: psABI vendor subsection name
_TAG_FILE = 1  # derived-ok: psABI Tag_file
_TAG_ARCH = 5  # derived-ok: psABI Tag_RISCV_arch (NTBS)
_EF_RVC = 0x1  # derived-ok: psABI e_flags compressed bit
_EF_FLOAT_ABI_SHIFT, _EF_FLOAT_ABI_MASK = 1, 0x3  # derived-ok: psABI e_flags float-ABI field
_FLOAT_ABI = {0: ("soft", None), 1: ("single", "f"), 2: ("double", "d"), 3: ("quad", "q")}
# The ISA manual's definition of the ``G`` shorthand, not a target property.
_G_EXPANSION = ("i", "m", "a", "f", "d", "zicsr", "zifencei")
_MULTI_LETTER_PREFIXES = ("z", "s", "x")


def _uleb(data: bytes, at: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        if at >= len(data) or shift > 63:
            raise ValueError("elf_arch_attribute_invalid")
        byte = data[at]
        at += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, at


def _ntbs(data: bytes, at: int, stop: int) -> tuple[str, int]:
    end = data.find(b"\0", at, stop)
    if end < 0:
        raise ValueError("elf_arch_attribute_invalid")
    return data[at:end].decode("ascii"), end + 1


def parse_attribute_section(data: bytes) -> str | None:
    """``Tag_RISCV_arch`` from the bytes of a ``.riscv.attributes`` section, or ``None``."""
    if not data or data[0] != _ATTRIBUTE_FORMAT:
        raise ValueError("elf_arch_attribute_invalid")
    at = 1
    while at < len(data):
        if at + 4 > len(data):
            raise ValueError("elf_arch_attribute_invalid")
        (length,) = struct.unpack_from("<I", data, at)
        end = at + length
        if length < 5 or end > len(data):
            raise ValueError("elf_arch_attribute_invalid")
        vendor, cursor = _ntbs(data, at + 4, end)
        if vendor.encode("ascii") == _ATTRIBUTE_VENDOR:
            while cursor < end:
                tag, body = _uleb(data, cursor)
                if body + 4 > end:
                    raise ValueError("elf_arch_attribute_invalid")
                (size,) = struct.unpack_from("<I", data, body)
                stop = cursor + size
                if size < 5 or stop > end:
                    raise ValueError("elf_arch_attribute_invalid")
                if tag == _TAG_FILE:
                    cursor = body + 4
                    while cursor < stop:
                        attribute, cursor = _uleb(data, cursor)
                        # psABI: unknown odd tags are NTBS, even tags ULEB128.
                        if attribute % 2:
                            value, cursor = _ntbs(data, cursor, stop)
                            if attribute == _TAG_ARCH:
                                return value
                        else:
                            _, cursor = _uleb(data, cursor)
                cursor = stop
        at = end
    return None


def elf_arch_attribute(headers, read_at) -> dict[str, Any] | None:
    """The ELF's own arch attribute, from section headers ``(name, type, flags, addr, off, size, ...)``."""
    found = [h for h in headers if h[1] == _SHT_RISCV_ATTRIBUTES]
    if not found:
        return None
    if len(found) != 1:
        return dict(value=None, source="elf:.riscv.attributes:Tag_RISCV_arch", error="elf_arch_attribute_ambiguous")
    try:
        value = parse_attribute_section(read_at(found[0][4], found[0][5]))
    except (ValueError, UnicodeDecodeError):
        # A present but unreadable attribute is not absence: the contract must not replace it.
        return dict(value=None, source="elf:.riscv.attributes:Tag_RISCV_arch", error="elf_arch_attribute_invalid")
    if not value:
        return None
    return dict(value=value, source="elf:.riscv.attributes:Tag_RISCV_arch")


def _strip_version(token: str) -> str:
    """``zicsr2p0`` -> ``zicsr``; ``zvl128b1p0`` -> ``zvl128b``; ``zba`` unchanged."""
    end = len(token)
    while end and token[end - 1].isdigit():
        end -= 1
    if end < len(token) and end >= 2 and token[end - 1] == "p" and token[end - 2].isdigit():
        end -= 1
        while end and token[end - 1].isdigit():
            end -= 1
    return token[:end]


def _single_letters(token: str) -> list[str]:
    letters, at = [], 0
    while at < len(token):
        char = token[at]
        if not char.isalpha():
            raise ValueError("decoder_isa_string_unparseable")
        letters.append(char)
        at += 1
        # Skip an optional ``<major>[p<minor>]`` version.
        while at < len(token) and token[at].isdigit():
            at += 1
        if at + 1 < len(token) and token[at] == "p" and token[at + 1].isdigit():
            at += 1
            while at < len(token) and token[at].isdigit():
                at += 1
    return letters


def isa_features(isa: str) -> tuple[int, tuple[str, ...]]:
    """Split a RISC-V ISA string into ``(xlen, extension names)`` without a regex."""
    text = isa.strip().lower()
    if not text.startswith("rv"):
        raise ValueError("decoder_isa_string_unparseable")
    rest = text[2:]
    digits = 0
    while digits < len(rest) and rest[digits].isdigit():
        digits += 1
    if not digits:
        raise ValueError("decoder_isa_string_unparseable")
    xlen, rest = int(rest[:digits]), rest[digits:]
    features: list[str] = []
    for index, chunk in enumerate(part for part in rest.split("_") if part):
        if chunk.startswith(_MULTI_LETTER_PREFIXES) and index:
            names = [_strip_version(chunk)]
        else:
            names = _single_letters(chunk)
        for name in names:
            for expanded in _G_EXPANSION if name == "g" else (name,):
                if expanded not in features:
                    features.append(expanded)
    if not features:
        raise ValueError("decoder_isa_string_unparseable")
    return xlen, tuple(features)


def elf_flag_corroboration(e_flags: int, features) -> dict[str, Any]:
    """Compare the ELF header's ABI flags with a selection; never adds an extension."""
    abi, needs = _FLOAT_ABI[(e_flags >> _EF_FLOAT_ABI_SHIFT) & _EF_FLOAT_ABI_MASK]
    expected = [needs] if needs else []
    if e_flags & _EF_RVC:
        expected.append("c")
    missing = [] if features is None else [name for name in expected if name not in features]
    return dict(e_flags=e_flags, float_abi=abi, rvc=bool(e_flags & _EF_RVC), implies=expected,
                consistent=None if features is None else not missing, missing=missing,
                scope="corroboration only; header flags add no extension")


def contract_host_isa(contract: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The host ISA the selected contract declares for its selected runner family, or ``None``.

    The functional simulator's ``--isa`` describes the executing hart and is preferred; the
    toolchain ``-march`` is the fallback. Conflicting declarations select nothing.
    """
    if not isinstance(contract, Mapping):
        return None
    runner = contract.get("runner")
    if not isinstance(runner, Mapping) or not isinstance(runner.get("backend"), str):
        return None
    family_name = runner["backend"]
    family = runner.get(family_name)
    if not isinstance(family, Mapping):
        return None
    base = "selected_contract:runner." + family_name

    def declared(flags, prefix):
        return [flag[len(prefix):] for flag in flags if isinstance(flag, str) and flag.startswith(prefix)]

    simulator = []
    for engine, block in sorted((family.get("engines") or {}).items()):
        if isinstance(block, Mapping) and isinstance(block.get("argv"), list):
            source = f"{base}.engines.{engine}.argv:--isa"
            simulator.extend((value, source) for value in declared(block["argv"], "--isa="))
    toolchain = family.get("toolchain")
    compiler = []
    if isinstance(toolchain, Mapping) and isinstance(toolchain.get("cflags"), list):
        compiler = [(value, base + ".toolchain.cflags:-march") for value in declared(toolchain["cflags"], "-march=")]
    for candidates in (simulator, compiler):
        values = {value for value, _ in candidates if value}
        if len(values) == 1:
            value = values.pop()
            return dict(value=value, source=next(source for v, source in candidates if v == value))
        if len(values) > 1:
            return None
    return None


def select_decoder_isa(*, elf_attribute, declared, e_flags: int, elf_xlen: int) -> dict[str, Any]:
    """The decoder ISA record: value, source, llvm-objdump options and corroboration."""
    if elf_attribute and elf_attribute.get("error"):
        return dict(value=None, source=elf_attribute.get("source"), status="unknown", reason=elf_attribute["error"],
                    options=[], corroboration=elf_flag_corroboration(e_flags, None))
    for candidate in (elf_attribute, declared):
        if not candidate or not candidate.get("value"):
            continue
        try:
            xlen, features = isa_features(str(candidate["value"]))
        except ValueError:
            return dict(value=str(candidate["value"]), source=candidate.get("source"), status="unknown",
                        reason="decoder_isa_string_unparseable", options=[],
                        corroboration=elf_flag_corroboration(e_flags, None))
        if xlen != elf_xlen:
            return dict(value=str(candidate["value"]), source=candidate.get("source"), status="unknown",
                        reason="decoder_isa_xlen_differs_from_elf_class", options=[],
                        corroboration=elf_flag_corroboration(e_flags, None))
        return dict(value=str(candidate["value"]), source=candidate.get("source"), status="selected",
                    features=list(features), options=["--mattr=" + ",".join("+" + name for name in features)],
                    corroboration=elf_flag_corroboration(e_flags, features))
    return dict(value=None, source=None, status="unknown", options=[],
                reason="no ELF arch attribute and no selected contract host ISA",
                corroboration=elf_flag_corroboration(e_flags, None))


def unrecognized_features(stderr: str) -> list[str]:
    """Feature names the disassembler reported ignoring (``'+x' is not a recognized feature``)."""
    found = []
    for line in stderr.splitlines():
        if "is not a recognized feature" in line:
            name = line.partition("'")[2].partition("'")[0]
            if name and name not in found:
                found.append(name)
    return found
