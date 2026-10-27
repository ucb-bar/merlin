"""Small row profiles preserve frozen feature dimensions and reject attention."""

import importlib.util

import pytest

from merlin.common.paths import repo_root

torch = pytest.importorskip("torch")


@pytest.fixture
def shared(monkeypatch):
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("shared_chunks_test", folder / "prepare_shared_chunks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_row_profile_preserves_features_weights_and_exact_outputs(shared):
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.arange(12, dtype=torch.int8).reshape(3, 4))

        def forward(self, x):
            flat = torch.ops.aten.view.default(x, [-1, 3])
            product = torch.ops.aten._int_mm.default(flat, self.weight)
            return (torch.ops.aten.view.default(product, [1, 4, 4]),)

    inputs = (torch.arange(12, dtype=torch.int8).reshape(1, 4, 3),)
    parent = torch.export.export(Projection(), inputs).module()
    calls = [n for n in parent.graph.nodes if n.op == "call_function"]
    parent, _, _ = shared.slice_graph(parent, calls[0].name, calls[-1].name)
    original_weight = parent.weight.clone()
    derived, arguments = shared.row_profile(parent, inputs, 4, 1)
    torch.testing.assert_close(derived(*arguments)[0], parent(*inputs)[0][:, :1])
    torch.testing.assert_close(derived.weight, original_weight)
    assert parent(*inputs)[0].shape == (1, 4, 4)


def test_row_reduction_and_runtime_right_operand_are_refused(shared):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    reduced = graph.call_function(torch.ops.aten.mean.dim, (x, [1], True))
    graph.output((reduced,))
    parent = torch.fx.GraphModule(torch.nn.Module(), graph)
    with pytest.raises(ValueError, match="last-feature-axis"):
        shared.row_profile(parent, (torch.ones(1, 4, 3),), 4, 1)

    graph = torch.fx.Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    product = graph.call_function(torch.ops.aten._int_mm.default, (x, y))
    graph.output((product,))
    parent = torch.fx.GraphModule(torch.nn.Module(), graph)
    with pytest.raises(ValueError, match="frozen"):
        shared.row_profile(parent, (), 4, 1)


def test_attention_is_refused(shared):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    output = graph.call_function(torch.ops.aten.softmax.int, (x, -1))
    graph.output((output,))
    parent = torch.fx.GraphModule(torch.nn.Module(), graph)
    with pytest.raises(ValueError, match="Not a proved row-local"):
        shared.row_profile(parent, (torch.ones(1, 4, 3),), 4, 1)
