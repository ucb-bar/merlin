"""Directed exact-arithmetic and original-source rotary interpreter checks."""
from fractions import Fraction
from pathlib import Path
import unittest
import numpy as np
import rotary_exact as exact


class ExactTests(unittest.TestCase):
    def test_rne_ties_even(self):
        self.assertEqual(exact.ratio_to_f32((1 << 24)+1, 1 << 24), 0x3f800000)
        self.assertEqual(exact.ratio_to_f32((1 << 24)+3, 1 << 24), 0x3f800002)
        self.assertEqual(exact.ratio_to_f32(-((1 << 24)+1), 1 << 24), 0xbf800000)
        self.assertEqual(exact.ratio_to_f32(1, 1 << 150), 0)
        self.assertEqual(exact.ratio_to_f32(3, 1 << 150), 2)

    def test_ambiguous_interval_refuses(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            exact.round_interval((1 << 24), (1 << 24)+2, 1 << 24)

    def test_pi_enclosure_and_roundtrip(self):
        lo, hi = exact._pi_over_two_interval()
        # Independent decimal prefix lower/upper encloses pi/2 as a coarse check.
        self.assertGreater(Fraction(lo, exact.SCALE), Fraction('1.57079632679489661923132169163975144209858469'))
        self.assertLess(Fraction(hi, exact.SCALE), Fraction('1.57079632679489661923132169163975144209858470'))
        for raw in (0, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x44ffe000, 0xbf800000):
            self.assertEqual(exact.ratio_to_f32(*exact.f32_ratio(raw)), raw)

    def test_known_exact_and_correctly_rounded_values(self):
        self.assertEqual(exact.sincos_f32(0), (0, 0x3f800000))
        self.assertEqual(exact.sincos_f32(0x3f800000), (0x3f576aa4, 0x3f0a5140))
        # The rejected newlib BF16 boundary case (position 1897, frequency 24).
        self.assertEqual(int(exact.angles(np.array([1897], np.int64))[0, 24]), 0x3ff2d0e6)
        self.assertEqual(exact.sincos_f32(0x3ff2d0e6)[0], 0x3f728001)

    def test_domain_refusal(self):
        for positions in (np.array([-1], np.int64), np.array([2048], np.int64), np.array([1], np.int32)):
            with self.assertRaises(ValueError):
                exact.angles(positions)
        for raw in (0x80000000, 0xbf800000, 0x45000000, 0x7fc00000):
            with self.assertRaises(ValueError):
                exact.sincos_f32(raw)

    def test_original_source_rotary_slice(self):
        import whole_reference as ref
        import rotary_source
        import os
        source_trace = os.environ.get('ROTARY_SOURCE_TRACE')
        if not source_trace:
            self.skipTest('ROTARY_SOURCE_TRACE is not selected')
        trace = Path(source_trace)
        if not trace.exists():
            self.skipTest('source capture unavailable')
        outputs, provenance, interpreter = rotary_source.replay(trace, ref.primitives.POLICIES)
        expected = exact.prepare(np.arange(64, dtype=np.int64)[None, :])
        for name in ('cos', 'sin'):
            np.testing.assert_array_equal(outputs[name].raw, expected[name+'_bf16'])
        self.assertEqual(provenance['source_node_count'], 22)
        self.assertEqual(interpreter.env, {})

    def test_stage_sequence_refuses_rotary_fixture(self):
        import json
        from hashlib import sha256
        import tempfile
        import torch
        from safetensors.torch import save_file
        import whole_reference as ref
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capsule = root/'capsules/layer0'
            capsule.mkdir(parents=True)
            (root/'capsules/manifest.json').write_text(json.dumps({
                'whole_model': {'regions': ['layer0'], 'links': [{'kind': 'layer0', 'inputs': []}]},
                'source_model_sha256': 'test'}))
            (capsule/'model.pt2').write_bytes(b'not loaded before rotary refusal')
            value = torch.zeros((1, 2, 64), dtype=torch.bfloat16)
            save_file({'input_0': value}, str(capsule/'tensors.safetensors'))
            (capsule/'source.json').write_text(json.dumps({
                'input_keys': ['input_0'], 'source_boundaries': ['to_7'],
                'model_sha256': sha256((capsule/'model.pt2').read_bytes()).hexdigest(),
                'tensors': {'input_0': {'shape': [1, 2, 64]}}}))
            with self.assertRaisesRegex(ValueError, 'v3 refuses rotary fixtures'):
                ref.run_sequence(root/'capsules', root/'output', ref.primitives.POLICIES)
            receipt = json.loads((root/'output/reference.json').read_text())
            self.assertEqual(receipt['status'], 'REFUSED')
            self.assertEqual(receipt['reference_version'], 3)


if __name__ == '__main__':
    unittest.main()
