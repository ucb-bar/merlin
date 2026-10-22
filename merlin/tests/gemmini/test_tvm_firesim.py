"""Reject incompatible guest binaries and misleading functional completion."""

import importlib.util
import struct
import unittest

from merlin.common.paths import repo_root


class FireSimPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = repo_root() / "examples/gemmini/comparisons/tvm/prepare_firesim.py"
        spec = importlib.util.spec_from_file_location("tvm_firesim_preparation", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def guest(self, *, machine=243, flags=5, entry=0x80000000, physical=0x80000000, memory=32, interpreter=False):
        raw = bytearray(256)
        raw[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into("<HHIQQQIHHHHHH", raw, 16, 2, machine, 1, entry, 64, 0, flags, 64, 56, 2 if interpreter else 1, 0, 0, 0)
        struct.pack_into("<IIQQQQQQ", raw, 64, 1, 5, 240, 0x80000000, physical, 16, memory, 16)
        if interpreter:
            struct.pack_into("<IIQQQQQQ", raw, 120, 3, 4, 240, 0, 0, 4, 4, 1)
        return raw

    def test_enabled_memory_must_cover_BSS_and_stack_as_well_as_file(self):
        ranges = [{"base": 0x80000000, "size": 64}]
        result = self.module.elf_layout(self.guest(), ranges)
        self.assertEqual(result["load_span_including_BSS_and_stack"], 32)
        with self.assertRaises(ValueError):
            self.module.elf_layout(self.guest(memory=65), ranges)

    def test_wrong_arch_ABI_entry_or_relocation_is_refused(self):
        ranges = [{"base": 0x80000000, "size": 64}]
        for options in ({"machine": 62}, {"flags": 1}, {"entry": 0x80000010},
                        {"physical": 0x80000008}, {"interpreter": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module.elf_layout(self.guest(**options), ranges)

    def test_load_identity_ignores_nonloaded_metadata_but_detects_code_changes(self):
        ranges = [{"base": 0x80000000, "size": 64}]
        raw = self.guest()
        baseline = self.module.elf_layout(raw, ranges)
        raw[224] = 1
        self.assertEqual(self.module.elf_layout(raw, ranges), baseline)
        raw[240] = 1
        self.assertNotEqual(self.module.elf_layout(raw, ranges), baseline)

    def test_missing_or_incomplete_positive_completion_is_refused(self):
        check = self.module.spike_passed
        good = "TVM_BLOCK_GRAPH_PASS calls=4 integer_exact=1\n"
        self.assertTrue(check(good, 0, 4))
        for text, code in ((good, 1), (good.replace("calls=4", "calls=1"), 0), ("", 0),
                           (good + "EXPORTED_GRAPH_NUMERICAL_FAILURE", 0)):
            self.assertFalse(check(text, code, 4))

    def test_arbitrary_simulator_failure_does_not_pass_negative_control(self):
        check = self.module.spike_passed
        marker = "EXPORTED_GRAPH_NUMERICAL_FAILURE"
        self.assertTrue(check(marker, 22, 4, True))
        self.assertFalse(check(marker, 1, 4, True))
        self.assertFalse(check("trap or missing file", 22, 4, True))
        self.assertFalse(check(marker + " TVM_BLOCK_GRAPH_PASS", 22, 4, True))


if __name__ == "__main__":
    unittest.main()
