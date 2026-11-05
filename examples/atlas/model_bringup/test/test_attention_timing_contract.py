import copy
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from attention_timing_contract import metadata_contract, validate_contract


class AttentionTimingContractTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(query_rows=33, keys=33, query_heads=1, kv_heads=1,
                           head_dim=64, packed_decode=False, output_fp8=False)
        self.offsets = dict(q=0, k=8192, v=16384, descriptors=24576,
                            descriptor_bytes=32, key_counts=24608, key_count_bytes=32,
                            masks=24640, mask_bytes=2048, output=26944, output_bytes=8192)
        self.initial = bytearray(35392)
        struct.pack_into('<8I', self.initial, 24576, 0, 0, 0, 1, 0x4000, 0, 0, 1)
        struct.pack_into('<2I', self.initial, 24608, 1, 1)

    def contract(self):
        return metadata_contract(self.config, self.offsets, self.initial)

    def test_exact_image_and_optional_record(self):
        value = self.contract()
        self.assertEqual(value['name'], 'attention-metadata-v1')
        self.assertEqual(value['segments'][0]['address'], 0x90006000)
        self.assertEqual(bytes.fromhex(value['segments'][0]['bytes_hex']), self.initial[24576:24608])
        self.assertEqual(validate_contract(self.config, self.offsets, self.initial, 0x90000000, value), value)

    def test_mask_base_outside_or_unaligned(self):
        for address in (0x3fff, 0x4001, 0x4200, 0xffffffff):
            with self.subTest(address=address):
                struct.pack_into('<I', self.initial, 24592, address)
                with self.assertRaisesRegex(ValueError, 'mask base'):
                    self.contract()

    def test_count_cannot_exceed_geometry_or_visible_tiles(self):
        for count in (0, 2, 3, 0xffffffff):
            with self.subTest(count=count):
                struct.pack_into('<I', self.initial, 24608, count)
                with self.assertRaisesRegex(ValueError, 'key count'):
                    self.contract()

    def test_empty_query_tile_has_zero_bound(self):
        struct.pack_into('<8I', self.initial, 24576, 0, 1, 0, 1, 0, 1, 0, 1)
        struct.pack_into('<2I', self.initial, 24608, 0, 0)
        self.contract()

    def test_skip_and_padding(self):
        struct.pack_into('<I', self.initial, 24580, 2)
        with self.assertRaisesRegex(ValueError, 'skip flag'):
            self.contract()
        struct.pack_into('<I', self.initial, 24580, 0)
        self.initial[24616] = 1
        with self.assertRaisesRegex(ValueError, 'padding'):
            self.contract()

    def test_contract_changed_image_or_address(self):
        value = self.contract()
        changed = copy.deepcopy(value)
        changed['segments'][0]['address'] += 4
        with self.assertRaisesRegex(ValueError, 'differs from launch image'):
            validate_contract(self.config, self.offsets, self.initial, 0x90000000, changed)
        struct.pack_into('<I', self.initial, 24576, 0x4000)
        with self.assertRaisesRegex(ValueError, 'differs from launch image'):
            validate_contract(self.config, self.offsets, self.initial, 0x90000000, value)

    def test_truncated_image_and_offset_overlap(self):
        with self.assertRaisesRegex(ValueError, 'extent'):
            metadata_contract(self.config, self.offsets, self.initial[:-1])
        self.offsets['key_counts'] -= 4
        with self.assertRaisesRegex(ValueError, 'layout'):
            self.contract()


if __name__ == '__main__':
    unittest.main()
