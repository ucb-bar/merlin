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

    def __init__(self, fd, label):
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
            self.check()
        except BaseException:
            self.close(check=False)
            raise

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
        from merlin.compile.analysis.disassembly import objdump_bin, tokenize_text

        tool = objdump_bin()
        tool_info = os.stat(tool) if os.path.isabs(tool) else None
        key = (self.pin["sha256"], tool, identity(tool_info) if tool_info else None, section)
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
        from merlin.compile.analysis.disassembly import objdump_bin

        temporary_root = os.environ.get("TMPDIR")
        if not temporary_root or not os.path.isabs(temporary_root):
            raise ValueError("analysis_requires_owner_task_temporary_root")
        # --disassemble-symbols stops at local relocation labels in LLVM.
        # Select the ELF-owned section and byte interval instead. Zero padding
        # is retained so that the entire STT_FUNC extent can be checked.
        command = [objdump_bin(), "-d", "-r", "--disassemble-zeroes", "-M", "no-aliases"]
        sections = [s[0] for s in self.sections] if section is None else [section]
        if not set(sections) <= {s[0] for s in self.sections}:
            raise ValueError("analysis_elf_section_unavailable")
        command.extend("--section=" + name for name in sections)
        if address_range is not None:
            command.extend(("--start-address=" + str(address_range[0]),
                            "--stop-address=" + str(address_range[1])))

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
        self.check()
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
        from merlin.compile.analysis.disassembly import tokenize_text

        extent = self.function_extent(symbol)
        start, end = extent["address"], extent["address"] + extent["size_bytes"]
        text = self._disassembly_text(section=extent["section"], address_range=(start, end))
        rows = tuple(tokenize_text(text))
        section = next(s for s in self.sections if s[0] == extent["section"])
        code = self.read_at(section[1] + start - section[3], end - start)
        coverage = function_listing_coverage(rows, extent, code)
        self.check()
        return dict(schema="merlin.elf_function_disassembly.v1", artifact=dict(self.pin),
                    extent=extent, coverage=coverage, text=text)


def function_listing_coverage(rows, extent, code):
    """Check exact ordered instruction bytes cover a selected original function."""
    start, size = extent["address"], extent["size_bytes"]
    if not 0 < size <= MAX_CODE_BYTES or len(code) != size or not 0 < len(rows) <= MAX_INSTRUCTION_ROWS:
        raise ValueError("analysis_function_listing_incomplete")
    cursor = 0
    for row in rows:
        if (row.section_name != extent["section"] or row.addr != start + cursor
                or len(row.hexcode) not in (4, 8)):
            raise ValueError("analysis_function_listing_incomplete")
        try:
            encoded = int(row.hexcode, 16).to_bytes(len(row.hexcode) // 2, "little")
        except (ValueError, OverflowError) as error:
            raise ValueError("analysis_function_listing_encoding_invalid") from error
        if code[cursor:cursor + len(encoded)] != encoded:
            raise ValueError("analysis_function_listing_bytes_differ")
        cursor += len(encoded)
    if cursor != size:
        raise ValueError("analysis_function_listing_incomplete")
    return dict(status="complete_exact_function_bytes", start=start, end=start + size,
                covered_bytes=cursor, instructions=len(rows),
                original_code_sha256=hashlib.sha256(code).hexdigest(),
                scope="static original ELF function extent, not dynamic execution or timing")
