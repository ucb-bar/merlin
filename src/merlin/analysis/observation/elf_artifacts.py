"""Bounded, pinned ELF access without reading model weights into Python memory."""

from __future__ import annotations

import hashlib
import os
import resource
import stat
import struct
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

# ---- artifact_reader ---------------------------------------------------------------------------

MAX_ELF_BYTES = 4 << 30
MAX_CODE_BYTES = 32 << 20
MAX_TEXT_BYTES = 32 << 20
MAX_INSTRUCTION_ROWS = 2_000_000
_HASHES = OrderedDict()
_DISASSEMBLIES = OrderedDict()


def identity(info):
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def open_ordinary(path):
    """Open by directory FDs so ancestor replacement cannot redirect the read."""
    path = path.absolute()
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = following
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ValueError("input_not_a_bounded_ordinary_file")
    return fd


class PinnedELF:
    """Own one checked descriptor and bounded metadata until analysis completes.

    Positional reads also make concurrent truncation a refusal rather than a
    process-killing SIGBUS from a mapped candidate file.
    """

    def __init__(self, fd, label, *, declared_isa=None):
        self.fd = fd
        try:
            self.before = os.fstat(fd)
            if not 0 < self.before.st_size <= MAX_ELF_BYTES:
                raise ValueError("analysis_elf_file_exceeds_bound")
            header = self.read_at(0, 64)
            if header[:4] != b"\x7fELF":
                raise ValueError("analysis_elf_invalid")
            if header[:7] != b"\x7fELF\x02\x01\x01" or struct.unpack_from("<H", header, 18)[0] != 243:
                raise ValueError("analysis_cpu_isa_unsupported")
            shoff = struct.unpack_from("<Q", header, 40)[0]
            self.e_flags = struct.unpack_from("<I", header, 48)[0]
            stride, count, names_index = struct.unpack_from("<HHH", header, 58)
            if not count or stride < 64 or count * stride > 16 << 20 or names_index >= count:
                raise ValueError("analysis_elf_invalid")
            metadata = self.read_at(shoff, count * stride)
            self.headers = [struct.unpack_from("<IIQQQQIIQQ", metadata, i * stride) for i in range(count)]
            name_header = self.headers[names_index]
            if name_header[1] != 3:
                raise ValueError("analysis_elf_invalid")
            names = self.read_at(name_header[4], name_header[5])
            self.sections = []
            for entry in self.headers:
                name, kind, flags, address, offset, size, *_ = entry
                if kind != 1 or not flags & 4 or not size:
                    continue
                stop = names.find(b"\0", name, name + 1025)
                if name >= len(names) or stop < 0 or offset + size > len(self):
                    raise ValueError("analysis_elf_invalid")
                self.sections.append((names[name:stop].decode("utf-8"), offset, size, address))
            if len({s[0] for s in self.sections}) != len(self.sections):
                raise ValueError("analysis_elf_invalid")
            if not self.sections or sum(s[2] for s in self.sections) > MAX_CODE_BYTES:
                raise ValueError("analysis_elf_code_exceeds_bound")
            key = identity(self.before)
            digest = _HASHES.get(key)
            if digest is None:
                digest = hashlib.sha256()
                cursor = 0
                while cursor < self.before.st_size:
                    chunk = os.pread(fd, min(1 << 20, self.before.st_size - cursor), cursor)
                    if not chunk:
                        raise ValueError("analysis_input_changed")
                    digest.update(chunk)
                    cursor += len(chunk)
                digest = digest.hexdigest()
                _HASHES[key] = digest
                while len(_HASHES) > 64:
                    _HASHES.popitem(last=False)
            self.pin = dict(path=label, sha256=digest, size_bytes=self.before.st_size)

            self.arch_attribute = elf_arch_attribute(self.headers, self.read_at)
            self.declare_isa(declared_isa)
            self.check()
        except BaseException:
            self.close(check=False)
            raise

    def declare_isa(self, declared):
        """Select the decoder ISA: the ELF's own attribute, else ``declared`` (``{value, source}``).

        ``declared`` is the selected contract's host ISA. Neither present leaves the ISA unknown
        and passes no feature option; the record is kept in :attr:`decoder_isa`.
        """

        self.decoder_isa = select_decoder_isa(
            elf_attribute=self.arch_attribute, declared=declared, e_flags=self.e_flags, elf_xlen=64
        )
        return self.decoder_isa

    def __len__(self):
        return self.before.st_size

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise ValueError("analysis_elf_invalid")
        start, stop, _ = key.indices(len(self))
        return self.read_at(start, stop - start)

    def read_at(self, offset, size):
        if not 0 <= size <= 16 << 20 or offset < 0 or offset + size > len(self):
            raise ValueError("analysis_elf_invalid")
        data = os.pread(self.fd, size, offset)
        if len(data) != size:
            raise ValueError("analysis_input_changed")
        return data

    def check(self):
        if identity(self.before) != identity(os.fstat(self.fd)):
            raise ValueError("analysis_input_changed")

    def function_extent(self, symbol):
        """Read only ELF64 symbol metadata, never materialize allocated data."""
        headers, count = self.headers, len(self.headers)
        definitions = set()
        for header in headers:
            _, kind, _, _, offset, size, link, _, _, entry_size = header
            if kind not in (2, 11):
                continue
            if entry_size != 24 or size % 24 or link >= count:
                raise ValueError("analysis_elf_invalid")
            strings = headers[link]
            if strings[1] != 3:
                raise ValueError("analysis_elf_invalid")
            table = self.read_at(strings[4], strings[5])
            symbols = self.read_at(offset, size)
            for at in range(0, size, 24):
                name, info, _, owner, address, length = struct.unpack_from("<IBBHQQ", symbols, at)
                if info & 15 != 2 or not 0 < owner < count or name >= len(table):
                    continue
                stop = table.find(b"\0", name, name + 1025)
                if stop >= 0 and table[name:stop].decode("utf-8") == symbol:
                    definitions.add((address, length, owner))
        if len(definitions) != 1:
            raise ValueError("analysis_elf_symbol_boundary_unavailable")
        address, length, owner = next(iter(definitions))
        section = next(
            (
                s
                for s in self.sections
                if s[1:3] == (headers[owner][4], headers[owner][5]) and s[3] == headers[owner][3]
            ),
            None,
        )
        if not length or section is None or not section[3] <= address < address + length <= section[3] + section[2]:
            raise ValueError("analysis_elf_symbol_boundary_unavailable")
        return dict(
            symbol=symbol, address=address, size_bytes=length, section=section[0], scope="static linked definition"
        )

    def close(self, *, check=True):
        try:
            if check and self.fd is not None:
                self.check()
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def instructions(self, *, section=None):
        from merlin.analysis.observation.disassembly import objdump_bin, tokenize_text

        tool = objdump_bin()
        tool_info = os.stat(tool) if os.path.isabs(tool) else None
        key = (
            self.pin["sha256"],
            tool,
            identity(tool_info) if tool_info else None,
            section,
            tuple(self.decoder_isa["options"]),
        )
        cached = _DISASSEMBLIES.get(key)
        if cached is not None:
            self.check()
            _DISASSEMBLIES.move_to_end(key)
            return cached
        text = self._disassembly_text(section=section)
        rows = tuple(tokenize_text(text))
        if not rows or len(rows) > MAX_INSTRUCTION_ROWS:
            raise ValueError("instruction_view_empty_or_exceeds_bound")
        _DISASSEMBLIES[key] = rows
        while len(_DISASSEMBLIES) > 3:
            _DISASSEMBLIES.popitem(last=False)
        return rows

    def _disassembly_text(self, *, section=None, address_range=None):
        from merlin.analysis.observation.disassembly import objdump_bin

        temporary_root = os.environ.get("TMPDIR")
        if not temporary_root or not os.path.isabs(temporary_root):
            raise ValueError("analysis_requires_owner_task_temporary_root")
        # --disassemble-symbols stops at local relocation labels in LLVM.
        # Select the ELF-owned section and byte interval instead. Zero padding
        # is retained so that the entire STT_FUNC extent can be checked.
        command = [objdump_bin(), "-d", "-r", "--disassemble-zeroes", "-M", "no-aliases", *self.decoder_isa["options"]]
        sections = [s[0] for s in self.sections] if section is None else [section]
        if not set(sections) <= {s[0] for s in self.sections}:
            raise ValueError("analysis_elf_section_unavailable")
        command.extend("--section=" + name for name in sections)
        if address_range is not None:
            command.extend(("--start-address=" + str(address_range[0]), "--stop-address=" + str(address_range[1])))

        def bounded_output():
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_TEXT_BYTES, MAX_TEXT_BYTES))

        with tempfile.TemporaryFile(dir=temporary_root) as output, tempfile.TemporaryFile(dir=temporary_root) as errors:
            process = subprocess.run(
                [*command, f"/proc/self/fd/{self.fd}"],
                pass_fds=(self.fd,),
                stdout=output,
                stderr=errors,
                timeout=60,
                preexec_fn=bounded_output,
            )
            output.seek(0)
            text = output.read(MAX_TEXT_BYTES + 1)
            errors.seek(0)
            diagnostics = errors.read(1 << 16).decode("utf-8", "replace")
        self.check()
        if self.decoder_isa["options"]:
            self.decoder_isa["unrecognized_features"] = unrecognized_features(diagnostics)
        if process.returncode or len(text) > MAX_TEXT_BYTES:
            raise ValueError("selected_disassembler_could_not_read_input")
        return text.decode("utf-8")

    def function_disassembly(self, symbol):
        """Retain one complete original STT_FUNC, including local labels/padding.

        Coverage is byte-for-byte against this pinned ELF, not inferred from
        the last listed PC or a return mnemonic. Unsupported/data encodings
        refuse coverage instead of silently yielding a prefix-only reference.
        This is static extent coverage, not an execution or semantic proof.
        """
        from merlin.analysis.observation.disassembly import tokenize_text

        extent = self.function_extent(symbol)
        start, end = extent["address"], extent["address"] + extent["size_bytes"]
        text = self._disassembly_text(section=extent["section"], address_range=(start, end))
        rows = tuple(tokenize_text(text))
        section = next(s for s in self.sections if s[0] == extent["section"])
        code = self.read_at(section[1] + start - section[3], end - start)
        coverage = function_listing_coverage(rows, extent, code)
        self.check()
        return dict(
            schema="merlin.elf_function_disassembly.v1",
            artifact=dict(self.pin),
            extent=extent,
            coverage=coverage,
            decoder_isa=dict(self.decoder_isa),
            text=text,
        )


def function_listing_coverage(rows, extent, code):
    """Check exact ordered instruction bytes cover a selected original function."""
    start, size = extent["address"], extent["size_bytes"]
    if not 0 < size <= MAX_CODE_BYTES or len(code) != size or not 0 < len(rows) <= MAX_INSTRUCTION_ROWS:
        raise ValueError("analysis_function_listing_incomplete")
    cursor = 0
    for row in rows:
        if row.section_name != extent["section"] or row.addr != start + cursor or len(row.hexcode) not in (4, 8):
            raise ValueError("analysis_function_listing_incomplete")
        try:
            encoded = int(row.hexcode, 16).to_bytes(len(row.hexcode) // 2, "little")
        except (ValueError, OverflowError) as error:
            raise ValueError("analysis_function_listing_encoding_invalid") from error
        if code[cursor : cursor + len(encoded)] != encoded:
            raise ValueError("analysis_function_listing_bytes_differ")
        cursor += len(encoded)
    if cursor != size:
        raise ValueError("analysis_function_listing_incomplete")
    return dict(
        status="complete_exact_function_bytes",
        start=start,
        end=start + size,
        covered_bytes=cursor,
        instructions=len(rows),
        original_code_sha256=hashlib.sha256(code).hexdigest(),
        scope="static original ELF function extent, not dynamic execution or timing",
    )


# ---- host_isa ----------------------------------------------------------------------------------
# Select the host ISA a disassembler decodes with, as recorded data with provenance.
#
# An ELF without a ``.riscv.attributes`` section tells LLVM nothing about the extensions it was
# built for, so ``llvm-objdump`` decodes only the base integer and compressed encodings and prints
# every M/F/D/... word as ``<unknown>``. Those words are ordinary host instructions, not accelerator
# commands, but a consumer that reads ``<unknown>`` as "undecodable" cannot tell them apart.
#
# The decoder ISA is therefore selected, in order, from:
#
# 1. the ELF's own arch attribute (``Tag_RISCV_arch``), when present;
# 2. the selected target contract's declared host ISA: the functional simulator's ``--isa`` for
#    the selected runner family, else that family's toolchain ``-march``;
# 3. otherwise nothing: the ISA stays ``unknown`` and no feature option is passed.
#
# The ELF header's float-ABI/RVC flags only corroborate a selection; they never add an extension.
# The ISA string is translated to ``--mattr`` structurally (``str`` operations, no regex), and the
# whole selection is returned as a record so a reviewer can see where the value came from.

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
    return dict(
        e_flags=e_flags,
        float_abi=abi,
        rvc=bool(e_flags & _EF_RVC),
        implies=expected,
        consistent=None if features is None else not missing,
        missing=missing,
        scope="corroboration only; header flags add no extension",
    )


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
        return [flag[len(prefix) :] for flag in flags if isinstance(flag, str) and flag.startswith(prefix)]

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
        return dict(
            value=None,
            source=elf_attribute.get("source"),
            status="unknown",
            reason=elf_attribute["error"],
            options=[],
            corroboration=elf_flag_corroboration(e_flags, None),
        )
    for candidate in (elf_attribute, declared):
        if not candidate or not candidate.get("value"):
            continue
        try:
            xlen, features = isa_features(str(candidate["value"]))
        except ValueError:
            return dict(
                value=str(candidate["value"]),
                source=candidate.get("source"),
                status="unknown",
                reason="decoder_isa_string_unparseable",
                options=[],
                corroboration=elf_flag_corroboration(e_flags, None),
            )
        if xlen != elf_xlen:
            return dict(
                value=str(candidate["value"]),
                source=candidate.get("source"),
                status="unknown",
                reason="decoder_isa_xlen_differs_from_elf_class",
                options=[],
                corroboration=elf_flag_corroboration(e_flags, None),
            )
        return dict(
            value=str(candidate["value"]),
            source=candidate.get("source"),
            status="selected",
            features=list(features),
            options=["--mattr=" + ",".join("+" + name for name in features)],
            corroboration=elf_flag_corroboration(e_flags, features),
        )
    return dict(
        value=None,
        source=None,
        status="unknown",
        options=[],
        reason="no ELF arch attribute and no selected contract host ISA",
        corroboration=elf_flag_corroboration(e_flags, None),
    )


def unrecognized_features(stderr: str) -> list[str]:
    """Feature names the disassembler reported ignoring (``'+x' is not a recognized feature``)."""
    found = []
    for line in stderr.splitlines():
        if "is not a recognized feature" in line:
            name = line.partition("'")[2].partition("'")[0]
            if name and name not in found:
                found.append(name)
    return found


__all__ = [
    "contract_host_isa",
    "elf_arch_attribute",
    "elf_flag_corroboration",
    "isa_features",
    "select_decoder_isa",
]
