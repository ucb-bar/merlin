"""Native projection capture admits only its explicitly selected source bounds."""

import importlib.util
import unittest

import numpy as np

from merlin.common.paths import repo_root


path = repo_root() / "examples/atlas/comparisons/capture_tinyllama_native.py"
spec = importlib.util.spec_from_file_location("native_query_bounds", path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def arrays_for(full_query):
    rows, columns = capture.projection_bounds(full_query)
    result = {}
    for name, shape in {"A": (rows, 2048), "B": (2048, columns),
                        "native_C": (rows, columns)}.items():
        result[name] = np.ones(shape, dtype=np.float32)
        result[name + "_bf16"] = np.full(shape, 0x3F80, dtype=np.uint16)
    return result


class NativeQueryBoundsTest(unittest.TestCase):
    def test_default_tile_bounds_remain_explicit(self):
        arrays = arrays_for(False)
        capture.validate_arrays(arrays)
        with self.assertRaises(ValueError):
            capture.validate_arrays(arrays, full_query=True)

    def test_full_query_requires_opt_in_and_authentic_raw_words(self):
        arrays = arrays_for(True)
        self.assertEqual(arrays["native_C"].shape, (128, 2048))
        capture.validate_arrays(arrays, full_query=True)
        with self.assertRaises(ValueError):
            capture.validate_arrays(arrays)
        arrays["A_bf16"][0, 0] ^= np.uint16(1)
        with self.assertRaises(ValueError):
            capture.validate_arrays(arrays, full_query=True)

    def test_full_query_shape_control_refuses_square_weight_truncation(self):
        arrays = arrays_for(True)
        self.assertEqual(arrays["B"].shape, (2048, 2048))
        arrays["B"] = arrays["B"][:, :-1].copy()
        with self.assertRaises(ValueError):
            capture.validate_arrays(arrays, full_query=True)


if __name__ == "__main__":
    unittest.main()
