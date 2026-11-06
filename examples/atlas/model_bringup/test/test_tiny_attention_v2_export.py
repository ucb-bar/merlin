"""Typed attention-v2 export, static tail, and sparse preservation contracts."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'tools/tiny_full'), str(ROOT/'tools/host_iss')]
import numpy as np
import kernel_library  # noqa: E402,F401
from atlas_attention_v2_program import AttentionV2Config, stage_v2
from attention_bundle import validate_attention_v2
from attention_v2_export import SCHEMA, attention_kernel_v2, config_from_dict, companion_c, output_spans


def fixture(rows, keys):
    config = AttentionV2Config(rows, keys, 32, 4, mask='causal', query_start=keys-rows,
        output_fp8=True, o_exponent=-13, packed_decode=True)
    initial, offsets = stage_v2(config, np.zeros((32, rows, 64), np.uint16),
        np.zeros((4, keys, 64), np.uint16), np.zeros((4, keys, 64), np.uint16))
    extent = offsets['output_bytes']
    spans = output_spans(config, extent)
    output = bytearray(extent)
    for span in spans:
        output[span['offset']:span['offset']+span['bytes']] = b'\x38'*span['bytes']
    final = initial[:offsets['output']]+output+initial[offsets['output']+extent:]
    item = dict(attention_version=2, kernel_schema='atlas.tiny.attention_launch.v2',
        config=asdict(config), offsets=offsets, output_offset=offsets['output'], output_extent_bytes=extent,
        total_dram_bytes=len(initial), output_dtype='fp8', output_layout='fp8_tiled', output_scale=-13,
        input_layout='bf16_tiled_packed_decode' if rows == 1 else 'bf16_tiled',
        packed_decode_effective=rows == 1, written_spans=spans,
        requested_dma=dict(load_bytes=offsets['output']-256, store_bytes=sum(x['bytes'] for x in spans)))
    return item, initial, bytes(output), bytes(final)


class AttentionV2ExportTests(unittest.TestCase):
    def test_admission_rejects_incomplete_and_patched_execution(self):
        record = dict(schema=SCHEMA, kind='attention', model_commit='47b8599d8ffbe5ddd96059ffe2b5be2427a410f5',
            numerics='rtl', accelerated=False, native_dispatch_enabled=False, dram_base=0x90000000,
            status='PASS', runtime_patched=False, mode='timed', source_and_binary_exact=True,
            source_bookends_verified=True, halt_exact=True, outside_written_spans_preserved=True,
            engine_drain_verified=True, halt_and_drain_exact=True)
        for key, value in (('status', 'PREPARED_NO_EXECUTION'), ('runtime_patched', True),
                ('mode', 'unknown'), ('source_bookends_verified', False), ('halt_exact', False),
                ('engine_drain_verified', False), ('halt_and_drain_exact', False),
                ('outside_written_spans_preserved', False), ('accelerated', True)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'original47 RTL instruction execution'):
                    attention_kernel_v2(None, dict(record, **{key: value}), None, None, 0x90000000)

    def test_independent_tail_and_sparse_extent_all_tiny_shapes(self):
        for rows, keys, extent, load, store in ((128, 128, 262144, 657568, 262144),
                (1, 129, 65536, 184416, 2048), (1, 130, 65536, 184416, 2048)):
            with self.subTest(rows=rows, keys=keys):
                item, initial, output, final = fixture(rows, keys)
                self.assertEqual(item['output_extent_bytes'], extent)
                self.assertEqual(item['requested_dma'], dict(load_bytes=load, store_bytes=store))
                self.assertEqual(asdict(config_from_dict(item['config'])), item['config'])
                validate_attention_v2(item, initial, output, final)

    def test_configuration_types_and_policy(self):
        config = fixture(1, 129)[0]['config']
        for name, value in (('output_fp8', 1), ('packed_decode', 1), ('query_rows', True),
                ('query_start', 128.0), ('head_dim', 32), ('p_exponent', -7), ('scale', True),
                ('o_exponent', 61), ('mask', 'none'), ('keys', 131)):
            with self.subTest(name=name, value=value):
                changed = dict(config, **{name: value})
                with self.assertRaises(ValueError):
                    config_from_dict(changed)

    def test_tail_guard_written_hole_and_input_corruptions(self):
        item, initial, output, final = fixture(1, 129)
        offset = item['output_offset']
        for at in (0, item['offsets']['descriptors'], item['offsets']['key_counts'],
                item['offsets']['masks'], offset-1, offset, offset+32, len(final)-1):
            with self.subTest(at=at):
                changed = bytearray(final)
                changed[at] ^= 1
                with self.assertRaises(ValueError):
                    validate_attention_v2(item, initial, output, bytes(changed))
        # Even a self-consistent nonzero output hole must be rejected.
        changed_output, changed_final = bytearray(output), bytearray(final)
        changed_output[32] = changed_final[offset+32] = 1
        with self.assertRaises(ValueError):
            validate_attention_v2(item, initial, bytes(changed_output), bytes(changed_final))

    def test_metadata_cannot_conflate_extent_with_dma(self):
        item, initial, output, final = fixture(1, 129)
        for key, value in (('written_spans', [dict(offset=0, bytes=65536)]),
                ('requested_dma', dict(load_bytes=184416, store_bytes=65536)),
                ('output_extent_bytes', 2048), ('output_scale', -12), ('packed_decode_effective', 1),
                ('input_layout', 'bf16_tiled'), ('output_dtype', 'bf16')):
            with self.subTest(key=key):
                changed = dict(item, **{key: value})
                with self.assertRaises(ValueError):
                    validate_attention_v2(changed, initial, output, final)

    def test_initial_tail_and_zero_output_are_independent(self):
        item, initial, output, final = fixture(1, 130)
        for at in (item['offsets']['descriptors'], item['offsets']['masks'], item['output_offset']+32):
            changed = bytearray(initial)
            changed[at] ^= 1
            with self.assertRaises(ValueError):
                validate_attention_v2(item, bytes(changed), output, final)

    def test_companion_uses_effective_packing_and_exact_spans(self):
        prefill, decode = fixture(128, 128)[0], fixture(1, 129)[0]
        source = '\n'.join(companion_c([prefill, decode]))
        self.assertIn('{2,1,2,524288,65536,655488,32,655520,2048,262144,-13,0,', source)
        self.assertIn('{2,2,2,16384,81920,180288,32,180320,4096,65536,-13,1,', source)
        self.assertIn('{1024,32}', source)
        bad = deepcopy(decode)
        bad['written_spans'][0]['bytes'] = 64
        with self.assertRaises(ValueError):
            companion_c([bad])


if __name__ == '__main__':
    unittest.main()
