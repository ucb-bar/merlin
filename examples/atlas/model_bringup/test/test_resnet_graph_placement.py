"""Full ResNet placement controls without device or model execution."""

from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_resnet_graph_host as build
from atlas_host_build import GRAPH_PROGRAM_WORDS, LINK
from atlas_host_layout import ATLAS_DRAM_BASE


def program():
    # The forward branch skips transfers and the rescale word before ECALL.
    return struct.pack("<7I", 0x00100f13, 30 << 15 | 0x7f, 0x463,
        3 << 20 | 2 << 15 | 1 << 7 | 0x7b, 127 << 23 | 0x25f,
        1 << 25 | 3 << 20 | 2 << 15 | 1 << 7 | 0x7b, 0x73)


class ResNetPlacementTest(unittest.TestCase):
    def test_default_program_and_patch_are_unchanged(self):
        raw = program()
        self.assertEqual(build.relocate_program(raw, 4, ATLAS_DRAM_BASE), (raw, 4))

    def test_runtime_patch_matches_relocation_at_all_admitted_exponents(self):
        raw = program()
        for base in (0x90000000, 0x190000000, 0x200000000):
            moved, patch = build.relocate_program(raw, 4, base)
            self.assertGreater(patch, 4)
            for exponent in range(-126, 128):
                original, actual = bytearray(raw), bytearray(moved)
                word = ((exponent + 127) << 23) | 0x25f
                struct.pack_into("<I", original, 16, word)
                struct.pack_into("<I", actual, patch * 4, word)
                self.assertEqual(bytes(actual), build.arena.relocate(bytes(original), ATLAS_DRAM_BASE, base))

    def test_patch_and_expanded_program_bounds(self):
        for patch in (-1, 7, True, 3):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                build.relocate_program(program(), patch, 0x90000000)
        raw = program()[:-4] + struct.pack("<I", 0x13) * (GRAPH_PROGRAM_WORDS - 7) + struct.pack("<I", 0x73)
        with self.assertRaisesRegex(ValueError, "program"):
            build.relocate_program(raw, 4, 0x90000000)

    def test_link_and_runtime_constants_agree_and_default_is_retained(self):
        header = (ROOT / "runtime/atlas_host_layout.h").read_bytes()
        self.assertEqual(build.placed_layout(header, ATLAS_DRAM_BASE), header)
        self.assertEqual(build.placed_link(4096, ATLAS_DRAM_BASE), LINK.replace("@DRAM_BYTES@", "4096"))
        for base in (0x90000000, 0x190000000, 0x200000000):
            link = build.placed_link(4096, base)
            self.assertIn(".atlas_dram " + hex(base) + " (NOLOAD)", link)
            self.assertIn("__atlas_dram_end <= " + hex(((base >> 32) + 1) << 32), link)
            self.assertIn(("#define ATLAS_DRAM_BASE " + hex(base)).encode(), build.placed_layout(header, base))
        for base, size in ((0x90000001, 4096), (0x1ffffffc0, 65)):
            with self.subTest(base=base), self.assertRaises(ValueError):
                build.placed_link(size, base)
        with self.assertRaises(ValueError):
            build.placed_layout(header + header, 0x90000000)


if __name__ == "__main__":
    unittest.main()
