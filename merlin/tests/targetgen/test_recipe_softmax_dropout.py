"""A literal evaluation dropout preserves the existing softmax plane selection."""

from __future__ import annotations

import copy

import pytest

from merlin.targetgen import _dynamic_contractions as DC
from merlin.targetgen import _static_attention as SA

torch = pytest.importorskip("torch")


def _graph(probability=0.0, training=False, *, target=None):
    graph = torch.fx.Graph()
    scores, values = graph.placeholder("scores"), graph.placeholder("values")
    probabilities = graph.call_function(torch.ops.aten.softmax.int, (scores, -1))
    cast = graph.call_function(torch.ops.aten.to.dtype, (probabilities, torch.float32))
    dropout = graph.call_function(target or torch.ops.aten.dropout.default, (cast, probability, training))
    result = graph.call_function(torch.ops.aten.matmul.default, (dropout, values))
    graph.output(result)
    return graph, cast, dropout, result


@pytest.mark.parametrize("probability", [0.0, 0])
def test_literal_evaluation_dropout_keeps_the_selected_softmax_planes(probability):
    graph, _, dropout, result = _graph(probability)
    before = str(graph)
    block = {"planes": {DC.SOFTMAX_OUTPUT: 2, SA.SOFTMAX_PARTNER: 2}}
    original = copy.deepcopy(block)
    assert DC.softmax_produced(dropout)
    plan = SA.plan(result, block)
    assert plan["context"] and plan["planes"] == plan["digits"] == 2
    assert plan["residual_digits"]
    assert str(graph) == before and block == original


@pytest.mark.parametrize(
    ("probability", "training"),
    [(0.1, False), (float("nan"), False), (0.0, True), (0.0, 0), (False, False), ("0", False)],
)
def test_other_dropout_selections_are_not_a_softmax_identity(probability, training):
    _, _, dropout, result = _graph(probability, training)
    assert not DC.softmax_produced(dropout)
    assert not SA.plan(result, {"planes": {DC.SOFTMAX_OUTPUT: 2}})["context"]


@pytest.mark.parametrize("field", ["probability", "training"])
def test_symbolic_dropout_selection_is_not_assumed_to_be_an_identity(field):
    graph, cast, dropout, _ = _graph()
    with graph.inserting_before(dropout):
        dynamic = graph.placeholder(field)
    dropout.args = (cast, dynamic, False) if field == "probability" else (cast, 0.0, dynamic)
    assert not DC.softmax_produced(dropout)


def test_mutating_dropout_and_changed_live_operands_remain_barriers():
    _, _, dropout, _ = _graph(target=torch.ops.aten.dropout_.default)
    assert not DC.softmax_produced(dropout)
    _, cast, dropout, _ = _graph()
    assert DC.softmax_produced(dropout)
    dropout.args = (cast, 0.1, False)
    assert not DC.softmax_produced(dropout)
    dropout.args = (cast, 0.0, True)
    assert not DC.softmax_produced(dropout)


def test_unknown_kwargs_and_missing_literal_fields_are_not_assumed_safe():
    _, cast, dropout, _ = _graph()
    dropout.kwargs = {"unknown": False}
    assert not DC.softmax_produced(dropout)
    dropout.kwargs = {}
    dropout.args = (cast, 0.0)
    assert not DC.softmax_produced(dropout)


@pytest.mark.parametrize("operation", ["rand_like", "neg", "detach"])
def test_random_or_unknown_wrappers_remain_barriers(operation):
    _, cast, dropout, _ = _graph()
    dropout.target = getattr(torch.ops.aten, operation).default
    dropout.args = (cast,)
    assert not DC.softmax_produced(dropout)


def test_exported_evaluation_dropout_reaches_the_existing_two_plane_rewrite():
    class Attention(torch.nn.Module):
        def forward(self, scores, values):
            probabilities = torch.softmax(scores, dim=-1).to(values.dtype)
            dropped = torch.nn.functional.dropout(probabilities, p=0.0, training=False)
            return probabilities, dropped, dropped @ values

    torch.manual_seed(7)
    scores = torch.randn(1, 2, 4, 16)
    values = torch.randn(1, 2, 16, 8)
    model = Attention().eval()
    exported = torch.export.export(model, (scores, values)).module()
    sites = [node for node in exported.graph.nodes if node.target == torch.ops.aten.matmul.default]
    assert len(sites) == 1
    dropout = sites[0].args[0]
    assert dropout.target == torch.ops.aten.dropout.default and dropout.args[1:] == (0.0, False)
    rows = SA.rewrite_sites(
        exported,
        sites,
        {"planes": {DC.SOFTMAX_OUTPUT: 2, SA.SOFTMAX_PARTNER: 2}},
        [(scores, values)],
        low=-128,
        high=127,
        eps=2.0**-12,
        owner=lambda node: node.name,
    )
    assert len(rows) == 1 and rows[0]["status"] == "rewritten" and rows[0]["planes"] == [2, 2]
    actual_calls = sum(node.target == torch.ops.aten._int_mm.default for node in exported.graph.nodes)
    assert rows[0]["device_calls"] == actual_calls == 8  # Four plane products for each of two heads.
    with torch.no_grad():
        actual, expected = exported(scores, values), model(scores, values)
        # The untouched observers retain the literal dropout's alias and values.
        assert actual[0].data_ptr() == actual[1].data_ptr()
        assert expected[0].data_ptr() == expected[1].data_ptr()
        assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], atol=1e-3, rtol=1e-3)
