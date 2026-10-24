"""Byte Q/DQ admission retains centering and range instead of assuming zero."""

from types import SimpleNamespace

import pytest

from merlin.targetgen._recipe_integer import _per_tensor_scale, integer_recipes


class Node:
    def __init__(self, target, args, **meta):
        self.op, self.target, self.args, self.kwargs = "call_function", target, args, {}
        self.meta, self.users, self.name = meta, [], "test_node"


@pytest.mark.parametrize(
    "token,zero,low,high", [("int8", -11, -128, 127), ("uint8", 131, 0, 255), ("int8", -2, -8, 7), ("uint8", 7, 0, 15)]
)
def test_byte_recipe_records_the_complete_source_qparams(token, zero, low, high):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, token)
    source = Node("quantized_decomposed.dequantize_per_tensor.default", (None, 0.25, zero, low, high, dtype))
    output = Node("quantized_decomposed.quantize_per_tensor.default", (None, 0.5, zero, low, high, dtype))
    node = Node("aten.clone.default", (source,), merlin_recipe_family="movement")
    node.users = [output]
    rows = integer_recipes(SimpleNamespace(graph=SimpleNamespace(nodes=[node])))
    assert len(rows) == 1 and "refused" not in rows[0]
    assert rows[0]["input_qparams"][0]["zero_point"] == zero
    assert rows[0]["integer_recipe"]["centering"] == "retained_source_integer"
    assert rows[0]["readout"] == "full_i32"


def test_invalid_byte_parameters_still_refuse():
    torch = pytest.importorskip("torch")
    for zero, low, high, dtype in [(256, 0, 255, torch.uint8), (0, -128, 255, torch.uint8), (0, -129, 127, torch.int8)]:
        source = Node("quantized_decomposed.dequantize_per_tensor.default", (None, 0.25, zero, low, high, dtype))
        assert _per_tensor_scale(source, "quantized_decomposed.dequantize_per_tensor") is None


def test_recipe_canonicalizes_before_independent_reference_and_recipe_extraction(monkeypatch):
    torch = pytest.importorskip("torch")
    scales = pytest.importorskip("m2m.capture.quant_scales")
    from merlin.targetgen import _recipe_quantizer as quantizer

    original = scales.canonicalize_pt2e_scales
    seen = []

    def observe(model):
        for node in model.graph.nodes:
            if str(node.target).startswith("quantized_decomposed.dequantize_per_tensor."):
                node.args = (node.args[0], 0.1, *node.args[2:])
                break
        ledger = original(model)
        seen.extend(ledger)
        return ledger

    monkeypatch.setattr(scales, "canonicalize_pt2e_scales", observe)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    recipe = dict(
        schema="quant_recipe_v1",
        status="derived",
        families=["contraction"],
        weight=dict(dtype="int8", granularity="tensor", symmetric=True, quant_min=-127, quant_max=127),
        activation=dict(
            dtype="int8", granularity="tensor", symmetric=True, quant_min=-128, quant_max=127, mode="static"
        ),
    )
    model = quantizer.apply_recipe(
        Model().eval(), recipe, example_inputs=(torch.tensor([[0.5, -0.75]]),), calibration_samples=1
    )
    assert seen and seen[0]["source"] == 0.1
    assert seen[0]["canonical"] == torch.tensor(0.1, dtype=torch.float32).item()
    assert model._recipe_quantization_stats["scale_canonicalization"]["changes"] == seen
    assert original(model) == seen  # The owner is idempotent.
