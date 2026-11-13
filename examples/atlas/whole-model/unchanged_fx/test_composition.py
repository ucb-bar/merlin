"""Composition closes generated words, rather than merely stage golden claims."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from composition import close_seams, verify_stage_inputs, generated_stage_inputs


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.array([0x3f800000, 0xbf800000], np.uint32)
        self.digest = hashlib.sha256(self.raw.tobytes()).hexdigest()
        self.link = {"input_index": 0, "region": "layer0", "output_index": 0,
                     "boundary": "hidden", "source_node": "hidden"}

    def test_raw_float_carrier_reinterpretation_preserves_every_bit(self):
        source = {"source_boundaries": ["hidden"], "tensors": {}}
        metadata = {"input_0": {"sha256": self.digest, "shape": [2],
                    "origin": {"reference_stage": "layer0", "output_index": 0}}}
        overrides, rows = verify_stage_inputs(source, [self.link], {"input_0": self.raw}, metadata,
                                              {"input_0": np.zeros(2, np.float32)})
        self.assertEqual(overrides['input_0'].dtype, np.float32)
        self.assertEqual(overrides['input_0'].tobytes(), self.raw.tobytes())
        self.assertEqual(rows[0]['sha256'], self.digest)

    def test_changed_oracle_producer_refuses(self):
        source = {"source_boundaries": ["hidden"], "tensors": {}}
        metadata = {"input_0": {"sha256": self.digest, "shape": [2],
                    "origin": {"reference_stage": "layer7", "output_index": 0}}}
        with self.assertRaisesRegex(ValueError, 'producer differs'):
            verify_stage_inputs(source, [self.link], {"input_0": self.raw}, metadata,
                                {"input_0": np.zeros(2, np.float32)})

    def records(self):
        return {'layer0': {'mismatches': 0, 'inputs': [],
                'execution': {'outputs': [{'sha256': self.digest}]}},
                'layer1': {'mismatches': 0, 'inputs': [{'input_index': 0, 'sha256': self.digest}],
                'execution': {'outputs': [{'sha256': 'last'}]}}}

    def test_exact_generated_seam_closes(self):
        closure = close_seams(['layer0', 'layer1'], {'layer0': [], 'layer1': [self.link]}, self.records())
        self.assertEqual(closure[0]['sha256'], self.digest)

    def test_output_equality_alone_cannot_hide_changed_seam(self):
        records = self.records()
        records['layer1']['inputs'][0]['sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'seam does not close'):
            close_seams(['layer0', 'layer1'], {'layer0': [], 'layer1': [self.link]}, records)

    def test_wrong_generated_output_refuses_even_when_seam_matches(self):
        records = self.records()
        records['layer1']['mismatches'] = 1
        with self.assertRaisesRegex(ValueError, 'differs from independent oracle'):
            close_seams(['layer0', 'layer1'], {'layer0': [], 'layer1': [self.link]}, records)


class QualityInitialInputTests(unittest.TestCase):
    def fixtures(self):
        activation = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)
        mask = np.array([0, 0xff7f], dtype=np.uint16)
        digest = hashlib.sha256(activation.tobytes()).hexdigest()
        source = {'kind': 'layer0', 'source_boundaries': ['embedding', 'mask'],
                  'tensors': {'input_0': {'sha256': 'original fixture'},
                              'input_1': {'sha256': hashlib.sha256(mask.tobytes()).hexdigest()}}}
        metadata = {'input_0': {'sha256': digest, 'shape': list(activation.shape),
                    'origin': {'source_boundary': 'embedding', 'preparation_sha256': 'prepared', 'tokens_sha256': 'tokens'}},
                    'input_1': {'sha256': source['tensors']['input_1']['sha256'], 'shape': list(mask.shape),
                    'origin': {'source_boundary': 'mask'}}}
        receipt = {'status': 'PASS', 'independent_raw_gather_verified': True,
                   'other_preparation_boundaries_token_independent': True, 'weight_cast_or_requantization': False,
                   'input_0': {'sha256': digest}, 'preparation_sha256': 'prepared', 'tokens_sha256': 'tokens'}
        arrays = {'input_0': activation, 'input_1': mask}
        return source, metadata, receipt, arrays

    def test_quality_embedding_requires_explicit_authentication_and_binds_raw_words(self):
        source, metadata, receipt, arrays = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'immutable preparation fixture'):
            verify_stage_inputs(source, [], arrays, metadata, arrays)
        overrides, rows = verify_stage_inputs(source, [], arrays, metadata, arrays, arrays['input_0'], receipt)
        self.assertEqual(set(overrides), {'input_0'})
        self.assertEqual(overrides['input_0'].tobytes(), arrays['input_0'].tobytes())
        self.assertTrue(rows[0]['initial_source_embedding'])

    def test_sequential_embedding_publishes_token_and_preparation_binding(self):
        source, _, receipt, arrays = self.fixtures()
        overrides, rows = generated_stage_inputs(source, [], arrays, Path('.'), {}, arrays['input_0'], receipt)
        self.assertEqual(overrides['input_0'].tobytes(), arrays['input_0'].tobytes())
        self.assertEqual(rows[0]['tokens_sha256'], receipt['tokens_sha256'])
        self.assertEqual(rows[0]['preparation_sha256'], receipt['preparation_sha256'])

    def test_authenticated_embedding_never_authorizes_mask_changes(self):
        source, metadata, receipt, arrays = self.fixtures()
        arrays['input_1'] = arrays['input_1'] ^ np.uint16(1)
        metadata['input_1']['sha256'] = hashlib.sha256(arrays['input_1'].tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'immutable preparation fixture'):
            verify_stage_inputs(source, [], arrays, metadata, arrays, arrays['input_0'], receipt)

    def test_quality_preparation_and_tokens_must_match_oracle_origin(self):
        source, metadata, receipt, arrays = self.fixtures()
        metadata['input_0']['origin']['tokens_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'authenticated source gather'):
            verify_stage_inputs(source, [], arrays, metadata, arrays, arrays['input_0'], receipt)

    def test_quality_override_is_not_allowed_on_a_later_hidden_boundary(self):
        source, metadata, receipt, arrays = self.fixtures()
        source['kind'] = 'layer1'
        with self.assertRaisesRegex(ValueError, 'immutable preparation fixture'):
            verify_stage_inputs(source, [], arrays, metadata, arrays, arrays['input_0'], receipt)



class SequentialInputTests(unittest.TestCase):
    def test_actual_predecessor_words_replace_captured_fixture_without_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'layer0').mkdir()
            actual = np.array([0x3f80, 0xbf80], dtype=np.uint16)
            captured = {'input_0': np.zeros(2, dtype=np.uint16)}
            np.savez(root / 'layer0/actual.npz', output_0=actual)
            digest = hashlib.sha256(actual.tobytes()).hexdigest()
            records = {'layer0': {'mismatches': 0, 'completed_monotonic_ns': 1,
                       'execution': {'outputs': [{'sha256': digest, 'shape': [2], 'dtype': '<u2'}]}}}
            link = {'input_index': 0, 'region': 'layer0', 'output_index': 0}
            overrides, rows = generated_stage_inputs({'source_boundaries': ['hidden']},
                [link], captured, root, records)
            self.assertEqual(overrides['input_0'].tobytes(), actual.tobytes())
            self.assertNotEqual(overrides['input_0'].tobytes(), captured['input_0'].tobytes())
            self.assertEqual(rows[0]['input_origin'], 'generated_actual')
            # Replacing actual words after publication must fail before execution.
            np.savez(root / 'layer0/actual.npz', output_0=actual ^ np.uint16(1))
            with self.assertRaisesRegex(ValueError, 'publication changed'):
                generated_stage_inputs({'source_boundaries': ['hidden']}, [link], captured, root, records)

    def test_sequential_seam_requires_prior_completion(self):
        digest = 'same words'
        link = {'input_index': 0, 'region': 'layer0', 'output_index': 0}
        records = {'layer0': {'mismatches': 0, 'inputs': [], 'completed_monotonic_ns': 3,
                   'execution': {'outputs': [{'sha256': digest}]}},
                   'layer1': {'mismatches': 0, 'started_monotonic_ns': 2,
                   'inputs': [{'input_index': 0, 'sha256': digest, 'input_origin': 'generated_actual'}]}}
        with self.assertRaisesRegex(ValueError, 'before consumer started'):
            close_seams(['layer0', 'layer1'], {'layer0': [], 'layer1': [link]}, records)
        records['layer1']['started_monotonic_ns'] = 4
        self.assertEqual(len(close_seams(['layer0', 'layer1'], {'layer0': [], 'layer1': [link]}, records)), 1)

    def test_changed_immutable_fixture_refuses_without_oracle(self):
        with self.assertRaisesRegex(ValueError, 'immutable preparation fixture changed'):
            generated_stage_inputs({'source_boundaries': ['mask'], 'tensors': {'input_0': {'sha256': 'wrong'}}},
                [], {'input_0': np.zeros(2, dtype=np.uint16)}, Path('.'), {})


if __name__ == "__main__":
    unittest.main()
