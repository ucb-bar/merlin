"""Native Q substitution refuses the wrong source invocation before injection."""

import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from merlin.common.paths import repo_root


path = repo_root() / "examples/atlas/comparisons/compare_tinyllama_attention.py"
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("native_attention_linkage", path)
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


class NativeAttentionLinkageTest(unittest.TestCase):
    def test_full_source_input_and_output_words_are_both_required(self):
        source = {name: np.full((128, 2048), 0x3F80, dtype=np.uint16)
                  for name in ("A_bf16", "native_C_bf16")}
        inputs = source["A_bf16"][None].copy()
        output = source["native_C_bf16"][None].copy()
        torch = SimpleNamespace(bfloat16=np.dtype("uint16"))
        with patch.object(attention, "raw_array", side_effect=lambda value, _: value):
            attention.admit_query(inputs, output, source, torch)
            for name in ("input", "output", "shape"):
                a, c = inputs.copy(), output.copy()
                if name == "input":
                    a[0, 0, 0] ^= np.uint16(1)
                elif name == "output":
                    c[0, 0, 0] ^= np.uint16(1)
                else:
                    c = c[:, :32]
                with self.subTest(name=name), self.assertRaises(ValueError):
                    attention.admit_query(a, c, source, torch)

    def test_native_local_gate_keeps_explicit_tolerances(self):
        native = np.ones((4, 4), dtype=np.float32)
        self.assertTrue(attention.error(native, native)["allclose_rtol1e4_atol1e4"])
        changed = native.copy()
        changed[0, 0] = 1.01
        diagnostic = attention.error(changed, native)
        self.assertFalse(diagnostic["allclose_rtol1e4_atol1e4"])
        self.assertEqual(diagnostic["violating_values"], 1)


if __name__ == "__main__":
    unittest.main()
