"""Recipe-selected probability planes survive an actual exported inference identity."""
import importlib
from pathlib import Path

import pytest

from merlin.targetgen import _dynamic_contractions as DC

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(("probability", "training", "recognized"), [
    (0.0, False, True), (0.5, False, True), (0.0, True, True), (0.5, True, False),
])
def test_actual_exported_dropout_selects_only_legal_probability_planes(
    monkeypatch, probability, training, recognized,
):
    class Attention(torch.nn.Module):
        def forward(self, scores, values):
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            weights = torch.nn.functional.dropout(weights, p=probability, training=training)
            return weights @ values

    generator = torch.Generator().manual_seed(23)
    inputs = (torch.randn(2, 4, 3, 17, generator=generator),
              torch.randn(2, 4, 17, 9, generator=generator))
    module = Attention()
    exported = torch.export.export(module, inputs).graph_module
    matmul, = [node for node in exported.graph.nodes
               if node.op == "call_function" and node.target == torch.ops.aten.matmul.default]
    assert matmul.args[0].target == torch.ops.aten.dropout.default
    assert DC.softmax_produced(matmul.args[0]) is recognized

    # The actual static rewrite's planner must select the existing recipe's
    # probability AND partner planes; a matcher result alone is insufficient.
    monkeypatch.syspath_prepend(str(Path(DC.__file__).parent))
    planner = importlib.import_module("_static_attention")
    plan = planner.plan(matmul, {"planes": {"softmax_output": 2, "softmax_partner": 2}})
    assert plan["context"] is recognized
    assert (plan["planes"], plan["digits"]) == ((2, 2) if recognized else (1, 1))
    if recognized:
        # Prove the skipped node is an identity on the actual exported operands.
        expected = torch.softmax(inputs[0], dim=-1, dtype=torch.float32) @ inputs[1]
        assert torch.equal(module(*inputs), expected)


def test_symbolic_training_and_nonzero_probability_do_not_grant_identity():
    graph = torch.fx.Graph()
    scores = graph.placeholder("scores")
    training = graph.placeholder("training")
    weights = graph.call_function(torch.ops.aten.softmax.int, (scores, -1))
    dropout = graph.call_function(torch.ops.aten.dropout.default, (weights, 0.5, training))
    assert not DC.softmax_produced(dropout)


def test_selected_probability_planes_reach_the_executed_quantized_graph(monkeypatch):
    from merlin.targetgen import quant_recipe as QR

    monkeypatch.syspath_prepend(str(Path(DC.__file__).parent))
    quantizer_module = importlib.import_module("_recipe_quantizer")

    class Attention(torch.nn.Module):
        def forward(self, scores, values):
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            return torch.nn.functional.dropout(weights, p=0.0, training=False) @ values

    generator = torch.Generator().manual_seed(47)
    inputs = (torch.randn(1, 4, 3, 17, generator=generator),
              torch.randn(1, 4, 17, 9, generator=generator))
    module = Attention().eval()
    graph = torch.export.export(module, inputs).module()
    recipe = QR.QuantRecipe(
        target="test", unit="test",
        weight=QR.TensorSpec("int8", "tensor", True, -127, 127),
        activation=QR.TensorSpec("int8", "tensor", True, -128, 127),
    ).to_dict() | {"accumulator_dtype": "int32"}
    recipe["recipe_sha256"] = QR.digest(recipe)
    recipe = QR.with_dynamic_contractions(
        recipe, planes={QR.SOFTMAX_OUTPUT: 2, QR.SOFTMAX_PARTNER: 2},
        static_families=[], scales=QR.STATIC_HEADS,
    )
    quantizer = quantizer_module.build_quantizer(
        recipe, layer_plan=quantizer_module._plan_layers(recipe, module),
    )
    rows = quantizer.rewrite_dynamic(graph, [inputs], 1)
    assert len(rows) == 1
    assert rows[0]["planes"] == [2, 2]
    assert rows[0]["device_calls"] == 16  # four heads, four digit-pair contractions
    with torch.no_grad():
        actual, reference = graph(*inputs), module(*inputs)
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference) < 0.001
