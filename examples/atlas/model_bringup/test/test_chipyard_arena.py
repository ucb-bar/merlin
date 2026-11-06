"""Regression controls for DMA relocation and static physical ELF admission."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

TOOLS = Path(__file__).absolute().parents[1] / 'tools'
for directory in (TOOLS, TOOLS / 'host_iss', TOOLS / 'chipyard'):
    sys.path.insert(0, str(directory))
import arena
from firesim_target import package

BASE = 0x80000000


def elf():
    raw = bytearray(1024)
    struct.pack_into('<16sHHIQQQIHHHHHH', raw, 0, b'\x7fELF\x02\x01\x01', 2, 243, 1,
                     BASE, 64, 256, 4, 64, 56, 1, 64, 4, 0)
    struct.pack_into('<IIQQQQQQ', raw, 64, 1, 7, 0, BASE, BASE, 128, 128, 8)
    struct.pack_into('<IIQQQQIIQQ', raw, 320, 0, 8, 3, BASE + 16, 0, 32, 0, 0, 8, 0)
    names = ['tohost', 'fromhost', '__stack_start', '__stack_size', '_end', '__heap_end']
    table = b'\0' + b''.join(name.encode() + b'\0' for name in names)
    raw[512:512 + len(table)] = table
    struct.pack_into('<IIQQQQIIQQ', raw, 384, 0, 3, 0, 0, 512, len(table), 0, 0, 1, 0)
    struct.pack_into('<IIQQQQIIQQ', raw, 448, 0, 2, 0, 0, 640, 7 * 24, 2, 0, 8, 24)
    for i, value in enumerate([BASE + 64, BASE + 72, BASE + 256, 64, BASE + 384, BASE + 512], 1):
        struct.pack_into('<IBBHQQ', raw, 640 + i * 24, table.index(names[i - 1].encode()), 0, 0, 1, value, 8)
    return raw


def assemble(source):
    from npu_model.configs import isa_definition  # Register assembler metadata.
    from npu_model.util.converter import input_to_program
    return b''.join(struct.pack('<I', word & 0xffffffff) for word in input_to_program(io.StringIO(source)).assemble())


class ArenaTest(unittest.TestCase):
    def test_window_alignment_extent_and_low32_boundary(self):
        arena.window(0x1ffffffc0, 64)
        for base, size in [(True, 64), (1, 64), (0, 0), (0x1ffffffc0, 65), (1 << 64, 64)]:
            with self.subTest(base=base, size=size), self.assertRaises(ValueError):
                arena.window(base, size)

    def test_relocation_rejects_zero_and_aliased_dram_operands(self):
        for load in (True, False):
            for rd, rs1, rs2 in ([(1, 0, 3), (1, 1, 3), (1, 3, 3)] if load else
                                 [(0, 2, 3), (2, 2, 3), (3, 2, 3)]):
                source = f"li x30, 1\ndma.{'load' if load else 'store'} x{rd}, x{rs1}, x{rs2}, ch0\necall\n"
                dma = (0 if load else 1) << 25 | rs2 << 20 | rs1 << 15 | rd << 7 | 0x7b
                raw = struct.pack('<4I', 0x00100f13, 30 << 15 | 0x7f, dma, 0x73)
                with self.subTest(load=load, registers=(rd, rs1, rs2)):
                    with self.assertRaises(ValueError): arena.relocate(raw, arena.DEFAULT_ARENA, BASE)
                    with self.assertRaises(ValueError): arena.relocate_assembly(source, arena.DEFAULT_ARENA, BASE)

    def test_raw_relocation_rejects_x31_in_scalar_formats(self):
        # Immediate bits and tensor register fields are not scalar operands.
        formats = [(0x2003, (7, 15)), (0x2023, (15, 20)), (0x07, (15,)),
                   (0x67 | 1 << 12, (7, 15)), (0x13, (7, 15)),
                   (0x33, (7, 15, 20)), (0x37, (7,)), (0x63, (15, 20)),
                   (0x6f, (7,)), (0x73, (7, 15)), (0x7b, (7, 15, 20)),
                   (0x7f, (15,)), (0x0b, (7, 15, 20))]
        for opcode, shifts in formats:
            for shift in shifts:
                word = opcode | 31 << shift
                raw = struct.pack('<4I', 0x00100f13, word, 2 << 20 | 1 << 15 | 3 << 7 | 0x7b, 0x73)
                with self.subTest(opcode=hex(opcode), shift=shift), self.assertRaisesRegex(ValueError, 'unused x31'):
                    arena.relocate(raw, arena.DEFAULT_ARENA, BASE)

    def test_assembly_requires_one_exact_high_half_setup(self):
        transfer = 'dma.load x1, x2, x3, ch0\n'
        for setup in ('li x30, 10', 'li x30, 1 # comment', ' li x30, 1',
                      'li x30, 1\nli x30, 1'):
            source = setup + '\n' + transfer
            with self.subTest(setup=setup), self.assertRaisesRegex(ValueError, 'one retained'):
                arena.relocate_assembly(source, arena.DEFAULT_ARENA, BASE)
        source = 'li x30, 1\nli x30, 10\n' + transfer
        result = arena.relocate_assembly(source, arena.DEFAULT_ARENA, BASE)
        self.assertIn('li x30, 0\nli x30, 10\n', result)

    def test_assembly_rejects_unparsed_dma_and_wrong_configuration(self):
        for line in ('dma.load x1, x2, x3, ch8', 'dma.store x1, x2, x3, ch0 # comment',
                     'dma.load x1, x2', ' dma.load x1, x2, x3, ch0',
                     'dma.config x29, ch0', 'dma.config x30, ch8',
                     'loop: dma.load x1, x2, x3, ch0', 'loop: dma.config x29, ch0'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                arena.relocate_assembly('li x30, 1\n' + line + '\n', arena.DEFAULT_ARENA, BASE)

    def test_smol_patches_require_exactly_one_target(self):
        from smol_layer import replace_once
        for old in ('#define ATLAS_DRAM_BASE 0x100000000', '.atlas_dram 0x100000000',
                    '__atlas_dram_end <= 0x200000000'):
            self.assertEqual(replace_once('prefix ' + old + ' suffix', old, 'new'), 'prefix new suffix')
            for source in ('missing', old + '\n' + old):
                with self.subTest(old=old, source=source), self.assertRaises(ValueError):
                    replace_once(source, old, 'new')

    def test_raw_relocation_matches_generated_assembly_with_branches(self):
        import kernel_library  # noqa: E402,F401
        from atlas_program import assembly
        source = assembly(32, 2048, 256, 32 * 2048 + 2048 * 256 + 256, 0, traffic_reuse=True)
        raw = assemble(source)
        self.assertTrue(any(word & 127 in (0x63, 0x6f) for word, in struct.iter_unpack('<I', raw)))
        for base in (0x90000000, 0x190000000):
            with self.subTest(base=base):
                self.assertEqual(arena.relocate(raw, arena.DEFAULT_ARENA, base),
                                 assemble(arena.relocate_assembly(source, arena.DEFAULT_ARENA, base)))
        forced = assemble(arena.relocate_assembly(source, arena.DEFAULT_ARENA, arena.DEFAULT_ARENA, force=True))
        self.assertGreater(len(forced), len(raw))
        self.assertEqual(arena.relocate(raw, arena.DEFAULT_ARENA, arena.DEFAULT_ARENA), raw)

    def test_physical_dram_checks_memsz_sections_stack_and_heap(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'host.elf').write_bytes(elf())
            self.assertEqual(arena.audit(out, BASE, 1024)['reserved'],
                             [('stack', BASE + 256, BASE + 320), ('heap', BASE + 384, BASE + 512)])
            for offset, value, message in [(104, 1025, 'PT_LOAD'), (352, 1025, 'allocated section'),
                                            (640 + 3 * 24 + 8, BASE + 1000, 'stack'),
                                            (640 + 6 * 24 + 8, BASE + 1025, 'heap')]:
                raw = elf(); struct.pack_into('<Q', raw, offset, value)
                (out / 'host.elf').write_bytes(raw)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    arena.audit(out, BASE, 1024)

    def test_package_binds_selected_dram_and_rejects_abi_and_htif(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'host.elf').write_bytes(elf())
            contract = package(out, 'fixture', BASE, 1024)
            self.assertEqual((contract['dram_base'], contract['dram_bytes']), (BASE, 1024))
            self.assertFalse(contract['hardware_execution'])
            for key in ('cpu_mhz', 'atlas_mhz', 'vmem_bytes', 'imem_bytes', 'dma'):
                self.assertNotIn(key, contract)
            with self.assertRaises(TypeError):
                package(out, 'fixture')
            self.assertEqual(json.loads((out / 'marshal.json').read_text())['bin'], str(out / 'host.elf'))
            for offset, value, fmt, message in [(48, 0, '<I', 'ABI'), (672, BASE + 65, '<Q', 'HTIF'),
                                                (696, BASE + 64, '<Q', 'aliased HTIF')]:
                raw = elf(); struct.pack_into(fmt, raw, offset, value)
                (out / 'host.elf').write_bytes(raw)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    package(out, 'fixture', BASE, 1024)


if __name__ == '__main__':
    unittest.main()
