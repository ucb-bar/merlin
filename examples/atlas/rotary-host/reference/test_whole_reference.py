"""Independent original-FX interpreter regressions, with real layer fixtures."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import whole_reference as reference


class ReplayTests(unittest.TestCase):
    def test_typed_inputs_and_repeated_runs(self):
        class Movement(torch.nn.Module):
            def forward(self, codes, exponents, bits):
                return codes.reshape(2, 2).transpose(0, 1), exponents.reshape(2, 1), bits.clone()
        program = torch.export.export(Movement(), (torch.arange(4, dtype=torch.uint8),
                                                  torch.tensor([3, -4], dtype=torch.int32),
                                                  torch.tensor([1, 2], dtype=torch.int64)))
        interpreter = reference.Interpreter(program, reference.primitives.POLICIES)
        inputs = [reference.raw_tensor(value) for value in (torch.arange(4, dtype=torch.uint8),
                                                          torch.tensor([3, -4], dtype=torch.int32),
                                                          torch.tensor([1, 2], dtype=torch.int64))]
        for _ in range(2):
            outputs = interpreter.run(inputs)
            np.testing.assert_array_equal(outputs[0].raw, np.array([[0, 2], [1, 3]], np.uint8))
            np.testing.assert_array_equal(outputs[1].raw, np.array([[3], [-4]], np.int32))
            self.assertEqual(outputs[2].dtype, "i64")
            self.assertEqual(interpreter.env, {})
            self.assertEqual(interpreter.saved, {})

    def test_raw_encoding_refusal(self):
        class Identity(torch.nn.Module):
            def forward(self, value):
                return value
        program = torch.export.export(Identity(), (torch.ones(2, dtype=torch.bfloat16),))
        interpreter = reference.Interpreter(program, reference.primitives.POLICIES)
        with self.assertRaisesRegex(ValueError, "raw encoding"):
            interpreter.run([np.ones(2, np.float32)])

    def test_explicit_all_boundary_retention(self):
        class Movement(torch.nn.Module):
            def forward(self, value):
                return value.reshape(2, 2)
        program = torch.export.export(Movement(), (torch.arange(4, dtype=torch.uint8),))
        interpreter = reference.Interpreter(program, reference.primitives.POLICIES, retain=("*",))
        interpreter.run([np.arange(4, dtype=np.uint8)])
        self.assertEqual(set(interpreter.saved), set(interpreter.boundaries))
        self.assertEqual(interpreter.env, {})

    def test_head_requires_explicit_policy(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.ones((2, 3), dtype=torch.bfloat16))
            def forward(self, value):
                return torch.nn.functional.linear(value, self.weight)
        program = torch.export.export(Head(), (torch.ones((1, 3), dtype=torch.bfloat16),))
        interpreter = reference.Interpreter(program, reference.primitives.POLICIES)
        with self.assertRaisesRegex(ValueError, "explicit selected policy"):
            interpreter.run([np.full((1, 3), 0x3f80, np.uint16)])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
