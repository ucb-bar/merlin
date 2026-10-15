"""QKV substitution authenticates projections and admits the changed cache."""

import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from merlin.common.paths import repo_root


path = repo_root() / 'examples/atlas/comparisons/compare_tinyllama_qkv_attention.py'
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location('native_qkv_attention', path)
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)


class NativeQKVAttentionTest(unittest.TestCase):
    def test_original_projection_input_and_output_are_both_authenticated(self):
        source = {'A_bf16': np.full((4, 8), 0x3F80, dtype=np.uint16),
                  'native_C_bf16': np.full((4, 2), 0x3F00, dtype=np.uint16)}
        a, c = source['A_bf16'][None].copy(), source['native_C_bf16'][None].copy()
        torch = SimpleNamespace(bfloat16=np.dtype('uint16'))
        with patch.object(edge.attention, 'raw_array', side_effect=lambda value, _: value):
            edge.admit_projection(a, c, source, torch)
            for name in ('input', 'output', 'shape'):
                changed_a, changed_c = a.copy(), c.copy()
                if name == 'input':
                    changed_a[0, 0, 0] ^= np.uint16(1)
                elif name == 'output':
                    changed_c[0, 0, 0] ^= np.uint16(1)
                else:
                    changed_c = changed_c[:, :2]
                with self.subTest(name=name), self.assertRaises(ValueError):
                    edge.admit_projection(changed_a, changed_c, source, torch)

    def test_candidate_cache_must_match_actual_transform_not_old_native_cache(self):
        original = np.zeros((1, 2, 4, 2), dtype=np.uint16)
        expected_actual = np.full_like(original, 0x3F80)
        edge.require_raw_words(expected_actual.copy(), expected_actual, 'Candidate K cache')
        with self.assertRaises(ValueError):
            edge.require_raw_words(original, expected_actual, 'Candidate K cache')
        changed = expected_actual.copy()
        changed[0, 0, 0, 0] ^= np.uint16(1)
        with self.assertRaises(ValueError):
            edge.require_raw_words(changed, expected_actual, 'Candidate V cache')


if __name__ == '__main__':
    unittest.main()
