"""Full K/V capture derives dimensions and authenticates original raw mappings."""

import importlib.util
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from merlin.common.paths import repo_root


path = repo_root() / "examples/atlas/comparisons/capture_tinyllama_native_kv.py"
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("native_kv_capture", path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class NativeKVCaptureTest(unittest.TestCase):
    def test_dimensions_come_from_config_and_actual_linear(self):
        config = SimpleNamespace(hidden_size=96, num_attention_heads=6,
                                 num_key_value_heads=2)
        projection = SimpleNamespace(in_features=96, out_features=32,
                                     weight=SimpleNamespace(shape=(32, 96)))
        self.assertEqual(capture.projection_dimensions(config, projection), (96, 32, 16))
        config.head_dim = 24
        with self.assertRaises(ValueError):
            capture.projection_dimensions(config, projection)
        projection.out_features, projection.weight.shape = 48, (48, 96)
        self.assertEqual(capture.projection_dimensions(config, projection), (96, 48, 24))

    def test_coherent_mapping_mutations_are_refused_against_observed_source(self):
        shapes = {"A": (4, 8), "B": (8, 2), "native_C": (4, 2)}
        arrays = {}
        for name, shape in shapes.items():
            words = (np.arange(np.prod(shape)).reshape(shape) + 0x3F80).astype(np.uint16)
            arrays[name + "_bf16"] = words
            arrays[name] = (words.astype(np.uint32) << 16).view(np.float32)
        expected = {name: arrays[name + "_bf16"].copy() for name in shapes}
        capture.validate_projection_arrays(arrays, 4, 8, 2, expected)
        controls = capture.source_controls(arrays, 4, 8, 2, expected)
        self.assertEqual(len(controls), 6)
        self.assertTrue(all(controls.values()))
        wrong = dict(arrays, unknown=np.zeros(1))
        with self.assertRaises(ValueError):
            capture.validate_projection_arrays(wrong, 4, 8, 2, expected)


if __name__ == "__main__":
    unittest.main()
