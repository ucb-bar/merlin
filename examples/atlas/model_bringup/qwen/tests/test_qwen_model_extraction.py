"""Validate model-derived cuts on live tensor boundaries and repeated calls."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "requires the full-model capture environment",
)
class ModelExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "full_model"))
        import torch
        from capture_boundaries import cut

        cls.torch, cls.cut = torch, staticmethod(cut)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / "full_model"))

    def exported(self):
        torch = self.torch

        class Session(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4)
                )

            def forward(self, ids):
                first = self.mlp(self.embedding(ids))
                next_ids = first.argmax(dim=-1)
                second = self.mlp(self.embedding(next_ids))
                return first + second, first.square(), second.square()

        torch.manual_seed(17)
        model = Session().eval()
        ids = torch.tensor([[1, 2, 3]])
        program = torch.export.export(model, (ids,), strict=False)
        values = [
            program.state_dict[s.target] if s.kind.name == "PARAMETER" else ids
            for s in program.graph_signature.input_specs
        ]
        interpreter = torch.fx.Interpreter(
            program.graph_module, garbage_collect_values=False
        )
        interpreter.run(*values)
        return program, interpreter.env

    def test_prefill_and_decode_cuts_match_actual_graph_boundaries(self):
        program, observed = self.exported()
        names = []
        for step in (0, 1):
            kernel, inputs, outputs, selected = self.cut(
                program.graph_module.graph, "mlp", step
            )
            actual = kernel(*(observed[node] for node in inputs))
            for value, node in zip(actual, outputs, strict=True):
                self.torch.testing.assert_close(value, observed[node], rtol=0, atol=0)
            self.assertTrue(any(node.op == "placeholder" for node in inputs))
            names.append({node.name for node in selected})
        self.assertTrue(names[0].isdisjoint(names[1]))

    def test_missing_region_is_refused(self):
        program, _ = self.exported()
        with self.assertRaisesRegex(RuntimeError, "No exported operations"):
            self.cut(program.graph_module.graph, "absent")

    def test_missing_decode_step_is_refused(self):
        program, _ = self.exported()
        with self.assertRaisesRegex(RuntimeError, "No exported operations"):
            self.cut(program.graph_module.graph, "mlp", 2)


if __name__ == "__main__":
    unittest.main()
