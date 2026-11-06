"""V2 staging, exact BF16 codecs, sparse output, and worker admission boundaries."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import numpy as np
from tiny_full import attention_v2_worker as worker


class AttentionV2WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='attention-v2-test-')
        self.root = Path(self.temporary.name)
        self.config = worker.generator.AttentionV2Config(1, 129, 32, 4, mask='causal', query_start=128,
            q_exponent=-6, k_exponent=-5, v_exponent=-10, output_fp8=True, o_exponent=-13, packed_decode=True)
        self.request = dict(schema=worker.SCHEMA, command='attention_v2', root=str(self.root), out=str(self.root/'out'),
                            config=asdict(self.config), mode='functional')
        self.values = {}
        for name, shape in [('q', (32, 1, 64)), ('k', (4, 129, 64)), ('v', (4, 129, 64))]:
            value = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80], dtype='<u2'), shape)
            self.values[name] = value
            self.request[name+'_bf16'] = self.descriptor(name+'.bin', value.tobytes(), list(shape), 'uint16')
            self.request[name+'_tiled'] = self.descriptor(name+'-normal.bin', worker.generator.pack_tiles(value))

    def tearDown(self):
        self.temporary.cleanup()

    def descriptor(self, name, raw, shape=None, dtype='uint8'):
        (self.root/name).write_bytes(raw)
        return dict(file=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                    shape=shape if shape is not None else [len(raw)], dtype=dtype)

    def test_actual_tiny_decode_contract_and_sparse_extent(self):
        value = worker.prepare(self.request, self.root)
        self.assertEqual(len(value['initial']), 250464)
        self.assertEqual(value['offsets']['output'], 184672)
        self.assertEqual(value['offsets']['output_bytes'], 65536)
        self.assertEqual([len(value['staged'][key]) for key in ('q', 'k', 'v')], [16384, 81920, 81920])
        self.assertEqual(value['requested_dma'], dict(load_bytes=184416, store_bytes=2048))
        self.assertEqual(len(value['spans']), 64)
        self.assertEqual(value['spans'][0], dict(offset=0, bytes=32))
        self.assertEqual(value['spans'][1], dict(offset=1024, bytes=32))
        self.assertEqual(value['initial'][184672:184672+65536], bytes(65536))

    def test_opaque_normal_q_padding_ignored_and_real_bits_preserved(self):
        raw = worker.generator.pack_tiles(self.values['q'])
        padded = np.frombuffer(raw, dtype='<u2').copy().reshape(1, 32, 4, 32, 16)
        padded[:, :, :, 1:] = 0x7fc1
        self.request['q_tiled'] = self.descriptor('q-opaque.bin', padded.tobytes())
        value = worker.prepare(self.request, self.root)
        logical = worker.generator.unpack_tiles(value['staged']['q'], 4, 8).reshape(32, 1, 64)
        np.testing.assert_array_equal(logical, self.values['q'])
        all_rows = worker.generator.unpack_tiles(value['staged']['q'], 4, 32)
        self.assertFalse(all_rows[:, 8:].any())

    def test_future_kv_padding_must_be_zero(self):
        padded = np.frombuffer(worker.generator.pack_tiles(self.values['k']), dtype='<u2').copy().reshape(5, 4, 4, 32, 16)
        padded[-1, 0, 0, -1, 0] = 0x3f80
        self.request['k_tiled'] = self.descriptor('k-nonzero-future.bin', padded.tobytes())
        with self.assertRaisesRegex(ValueError, 'future KV padding'):
            worker.prepare(self.request, self.root)

    def test_full_finite_bf16_word_round_trip(self):
        words = np.arange(65536, dtype=np.uint16)
        finite = words[(words & 0x7f80) != 0x7f80]
        self.assertEqual(len(finite), 65280)
        value = np.concatenate((finite, np.zeros(256, dtype=np.uint16))).reshape(16, 64, 64)
        encoded = worker.generator.pack_tiles(value)
        decoded = worker.generator.unpack_tiles(encoded, 16, 64)
        np.testing.assert_array_equal(value, decoded)

    def test_complete_config_and_real_boolean_fields_required(self):
        for name, value in [('output_fp8', 1), ('packed_decode', 'yes'), ('scale', True), ('query_start', False)]:
            config = dict(self.request['config'], **{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError):
                worker.config_from(config)
        config = dict(self.request['config'])
        config.pop('p_exponent')
        with self.assertRaises(ValueError):
            worker.config_from(config)
        with self.assertRaises(ValueError):
            worker.config_from(dict(self.request['config'], decode_packed=True))

    def test_descriptor_tamper_and_source_alias_conflict_rejected(self):
        (self.root/'q.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'descriptor identity'):
            worker.prepare(self.request, self.root)
        self.request['q_bf16'] = self.descriptor('q.bin', self.values['q'].tobytes(), [32, 1, 64], 'uint16')
        self.request['q_source_tiled'] = dict(self.request['q_tiled'], sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'conflicting normal C'):
            worker.prepare(self.request, self.root)

    def test_sparse_holes_and_guards_preserved(self):
        value = worker.prepare(self.request, self.root)
        initial, offset, extent = value['initial'], value['offsets']['output'], value['offsets']['output_bytes']
        final = bytearray(initial)
        for span in value['spans']:
            final[offset+span['offset']:offset+span['offset']+span['bytes']] = b'\x38'*span['bytes']
        worker.validate_arena(initial, bytes(final), offset, extent, value['spans'])
        for at in (0, offset-1, offset+64, offset+extent):
            changed = bytearray(final)
            changed[at] ^= 1
            with self.subTest(at=at), self.assertRaises(ValueError):
                worker.validate_arena(initial, bytes(changed), offset, extent, value['spans'])

    def test_native_environment_refused_before_executor(self):
        for key in ('ATLAS_NPU_MODEL_NATIVE', 'NPU_MODEL_RTL_NATIVE_MXU0'):
            with self.subTest(key=key), patch.dict(os.environ, {key: '1'}), patch.object(worker, 'Executor') as executor:
                with self.assertRaisesRegex(ValueError, 'native execution disabled'):
                    worker.Worker(self.root/'not-a-model')
                executor.assert_not_called()

    def test_explicit_bf16_decode_also_distinguishes_extent_from_dma(self):
        self.request['config']['output_fp8'] = False
        value = worker.prepare(self.request, self.root)
        self.assertEqual(value['offsets']['output_bytes'], 131072)
        self.assertEqual(value['requested_dma']['store_bytes'], 4096)
        self.assertEqual(len(value['spans']), 128)


if __name__ == '__main__':
    unittest.main()
