"""Portable refusal controls for the frozen bottleneck host builder."""

from pathlib import Path
import struct
import tempfile
import unittest

import atlas_host_build as build
import kernel_library  # noqa: E402,F401
from atlas_tooling import Admission, sha


def synthetic_elf():
    raw = bytearray(0xC00)
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<16sHHIQQQIHHHHHH", raw, 0, ident, 2, 243, 1, 0x80000000, 64, 0x900, 5, 64, 56, 4, 64, 9, 8)
    asset = b"bound immutable"
    phdrs = [(1, 5, 0x200, 0x80000000, 0x80000000, 16, 16, 1),
        (1, 4, 0x300, 0x80001000, 0x80001000, len(asset), len(asset), 1),
        (1, 6, 0x400, 0x80002000, 0x80002000, 128, 0x11000, 1), (1, 6, 0, 1 << 32, 1 << 32, 0, 4096, 1)]
    for i, row in enumerate(phdrs):
        struct.pack_into("<IIQQQQQQ", raw, 64 + 56 * i, *row)
    raw[0x300 : 0x300 + len(asset)] = asset
    names = ["", ".text", ".rodata", ".data", ".atlas_dram", ".stack", ".symtab", ".strtab", ".shstrtab"]
    shstrings = b"\0"
    nameoffset = {}
    for name in names[1:]:
        nameoffset[name] = len(shstrings)
        shstrings += name.encode() + b"\0"
    raw[0x800 : 0x800 + len(shstrings)] = shstrings
    symbols = [("", 0, 0, 0), ("asset", 0x80001000, len(asset), 2), ("__stack_bottom", 0x80003000, 0, 5),
        ("__stack_top", 0x80013000, 0, 5), ("atlas_workload_mailbox", 0x80002000, 64, 3),
        ("atlas_resnet_state", 0x80002100, 64, 3), ("tohost", 0x80002040, 8, 3), ("fromhost", 0x80002080, 8, 3)]
    strings = b"\0"
    for i, (name, value, size, index) in enumerate(symbols):
        offset = len(strings) if name else 0
        if name:
            strings += name.encode() + b"\0"
        struct.pack_into("<IBBHQQ", raw, 0x600 + 24 * i, offset, 0x11, 0, index, value, size)
    raw[0x500 : 0x500 + len(strings)] = strings
    sections = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (nameoffset[".text"], 1, 6, 0x80000000, 0x200, 16, 0, 0, 4, 0),
        (nameoffset[".rodata"], 1, 2, 0x80001000, 0x300, len(asset), 0, 0, 64, 0),
        (nameoffset[".data"], 1, 3, 0x80002000, 0x400, 128, 0, 0, 64, 0),
        (nameoffset[".atlas_dram"], 8, 3, 1 << 32, 0, 4096, 0, 0, 64, 0),
        (nameoffset[".stack"], 8, 3, 0x80003000, 0, 65536, 0, 0, 64, 0),
        (nameoffset[".symtab"], 2, 0, 0, 0x600, len(symbols) * 24, 7, 0, 8, 24),
        (nameoffset[".strtab"], 3, 0, 0, 0x500, len(strings), 0, 0, 1, 0),
        (nameoffset[".shstrtab"], 3, 0, 0, 0x800, len(shstrings), 0, 0, 1, 0)]
    for i, row in enumerate(sections):
        struct.pack_into("<IIQQQQIIQQ", raw, 0x900 + 64 * i, *row)
    return raw, {"asset": asset}


class ResnetBlockHostTest(unittest.TestCase):
    def test_marker_tail_exact_and_read_only_mailbox_allowed(self):
        raw = struct.pack("<3I", 0xC1102FF3, 0x13, 0x73)
        loaded = build.wrap(raw)
        self.assertEqual(loaded[:8], raw[:8])
        self.assertEqual(loaded[8:], struct.pack("<6I", 0x13, 0xC1102FF3, 0x13, 0xC10F9073, 0x13, 0x73))

    def test_program_shape_and_trap_refusals(self):
        for raw in (b"", b"x", struct.pack("<I", 0x13), struct.pack("<2I", 0x73, 0x73), struct.pack("<2I", 0x00100073, 0x73),
            struct.pack("<I", 0x13) * (32768 - 5) + struct.pack("<I", 0x73)):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                build.wrap(raw)

    def test_reserved_mailbox_write_refusals(self):
        for word in (0xC1001073, 0xC1101073, 0xC1005073, 0xC110D073, 0xC1012073):
            with self.subTest(word=word), self.assertRaises(ValueError):
                build.wrap(struct.pack("<2I", word, 0x73))

    def test_admission_digest_bookends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"admitted")
            admission = Admission()
            self.assertEqual(admission.read(path, sha(b"admitted")), b"admitted")
            self.assertTrue(admission.recheck())
            with self.assertRaises(ValueError):
                Admission().read(path, sha(b"old"))
            path.write_bytes(b"changed")
            self.assertFalse(admission.recheck())
            with self.assertRaises(ValueError):
                admission.read(path)

    def test_elf_accepts_exact_ro_assets_stack_and_high_dram(self):
        raw, assets = synthetic_elf()
        audit = build.audit_elf(raw, assets, 4096)
        self.assertTrue(audit["embedded_assets_exact"])
        self.assertEqual(audit["stack_bytes"], 65536)
        self.assertEqual(audit["entry"], 0x80000000)

    def test_elf_refuses_changed_embedded_asset(self):
        raw, assets = synthetic_elf()
        raw[0x300] ^= 1
        with self.assertRaises(ValueError):
            build.audit_elf(raw, assets, 4096)

    def test_elf_refuses_writable_asset_section(self):
        raw, assets = synthetic_elf()
        struct.pack_into("<Q", raw, 0x900 + 2 * 64 + 8, 3)
        with self.assertRaises(ValueError):
            build.audit_elf(raw, assets, 4096)

    def test_elf_refuses_writable_asset_load_segment(self):
        raw, assets = synthetic_elf()
        struct.pack_into("<I", raw, 64 + 56 + 4, 6)
        with self.assertRaises(ValueError):
            build.audit_elf(raw, assets, 4096)

    def test_elf_refuses_rwx_text(self):
        raw, assets = synthetic_elf()
        struct.pack_into("<I", raw, 64 + 4, 7)
        with self.assertRaises(ValueError):
            build.audit_elf(raw, assets, 4096)

    def test_elf_refuses_bad_entry_or_abi(self):
        for offset, fmt, value in ((24, "<Q", 0x80000004), (48, "<I", 1), (18, "<H", 62)):
            raw, assets = synthetic_elf()
            struct.pack_into(fmt, raw, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                build.audit_elf(raw, assets, 4096)

    def test_elf_refuses_short_stack_or_misaligned_mailbox(self):
        for symbol, value in ((3, 0x80012FF0), (4, 0x80002004)):
            raw, assets = synthetic_elf()
            struct.pack_into("<Q", raw, 0x600 + symbol * 24 + 8, value)
            with self.subTest(symbol=symbol), self.assertRaises(ValueError):
                build.audit_elf(raw, assets, 4096)

    def test_elf_refuses_low32_alias_or_loaded_high_dram(self):
        for offset, value in ((0x900 + 4 * 64 + 16, 0), (64 + 3 * 56 + 32, 1)):
            raw, assets = synthetic_elf()
            struct.pack_into("<Q", raw, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                build.audit_elf(raw, assets, 4096)

    def test_elf_refuses_unmapped_or_overlapping_loads(self):
        raw, assets = synthetic_elf()
        struct.pack_into("<QQ", raw, 64 + 56 + 16, 0x40000, 0x40000)
        with self.assertRaises(ValueError):
            build.audit_elf(raw, assets, 4096)
        raw, assets = synthetic_elf()
        struct.pack_into("<QQ", raw, 64 + 56 + 16, 0x80000000, 0x80000000)
        with self.assertRaises(ValueError):
            build.audit_elf(raw, assets, 4096)


if __name__ == "__main__":
    unittest.main()
