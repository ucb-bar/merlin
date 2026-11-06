"""Capture/export staging contracts for Tiny residual and activation fusions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
from tiny_full.fusion import prepare_projection, pack_bf16_tiles, unpack_bf16_tiles, pack_fp8_tiles, unpack_fp8_tiles, pack_fp8_weights


class TinyFusionTest(unittest.TestCase):
    def stage(self, role, n=2048, column=0, width=5632):
        return prepare_projection(128, 2048, n, 0, role, column, width, bytes(128*2048), bytes(2048*n),
                                  bytes(2*128*n) if role in (3, 5, 6) else None, output_scale=-4 if role == 5 else 0)

    def test_strided_extent_and_extras(self):
        for n, column in ((2048, 0), (2048, 2048), (1536, 4096)):
            item = self.stage(5, n, column)
            self.assertEqual(item['dma_store_bytes'], 128*n)
            self.assertEqual(sum(s['bytes'] for s in item['written_spans']), 128*n)
            self.assertEqual(item['output_bytes'], 3*32*5632+column*32+n*32)
            self.assertEqual(item['extras'][0]['offset'], item['output_offset']+item['output_bytes']+256)
            self.assertEqual(item['initial'][item['output_offset']:item['output_offset']+item['output_bytes']], b'\xcd'*item['output_bytes'])
            self.assertEqual(item['epilogue']['scratch_registers'], (10, 11))
            self.assertIn('li x31, 2415919104', item['source'])

    def test_residual_and_silu(self):
        for role, op in ((3, 'add'), (4, 'silu'), (6, 'add')):
            item = self.stage(role)
            self.assertEqual(item['epilogue']['steps'][0]['op'], op)
            self.assertEqual(item['epilogue']['scratch_registers'], (10, 11))
            self.assertEqual(item['dma_store_bytes'], 2*128*2048)
            self.assertEqual(item['initial'][item['output_offset']-256:item['output_offset']], b'\xa5'*256)

    def test_pack_roundtrip(self):
        import numpy as np
        for m, n in ((32, 64), (128, 1536)):
            value = np.arange(m*n, dtype=np.uint16).reshape(m, n)
            self.assertTrue(np.array_equal(value, unpack_bf16_tiles(pack_bf16_tiles(value), m, n)))
            value = value.astype(np.uint8)
            self.assertTrue(np.array_equal(value, unpack_fp8_tiles(pack_fp8_tiles(value), m, n)))

    def test_physical_operand_tile_order(self):
        import numpy as np
        value = np.arange(64*96, dtype=np.uint16).reshape(64, 96)
        expected_bf16 = b''.join(value[r:r+32, c+half:c+half+16].astype('<u2').tobytes()
                                 for r in (0, 32) for c in (0, 32, 64) for half in (0, 16))
        self.assertEqual(pack_bf16_tiles(value), expected_bf16)
        value = value.astype(np.uint8)
        expected_a = b''.join(value[r:r+32, c:c+32].tobytes() for r in (0, 32) for c in (0, 32, 64))
        expected_b = b''.join(value[r:r+32, c:c+32].tobytes() for c in (0, 32, 64) for r in (0, 32))
        self.assertEqual(pack_fp8_tiles(value), expected_a)
        self.assertEqual(pack_fp8_weights(value), expected_b)

    def test_reject_missing_operand(self):
        with self.assertRaises(ValueError):
            prepare_projection(128, 2048, 2048, 0, 3, 0, 2048, bytes(128*2048), bytes(2048*2048))


if __name__ == '__main__':
    unittest.main()
