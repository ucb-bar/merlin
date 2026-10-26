"""Source-region preparation must retain shared edges and independent arithmetic."""

import importlib.util

import numpy as np
import pytest

from merlin.common.paths import repo_root

torch = pytest.importorskip("torch")


@pytest.fixture
def preparation(monkeypatch):
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("exported_chunks_test", folder / "prepare_exported_chunks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_keeps_shared_live_output_and_external_residual(preparation):
    class Program(torch.nn.Module):
        def forward(self, x):
            shared = torch.ops.aten.mul.Tensor(x, 2)
            hidden = torch.ops.aten.add.Tensor(shared, x)
            output = torch.ops.aten.mul.Tensor(hidden, 3)
            return output + shared

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    parent = torch.export.export(Program(), (x,)).module()
    calls = [n for n in parent.graph.nodes if n.op == "call_function"]
    chunk, inputs, outputs = preparation.slice_graph(parent, calls[0].name, calls[2].name)
    assert len(inputs) == 1
    assert [n.name for n in outputs] == [calls[0].name, calls[2].name]
    actual = chunk(x)
    torch.testing.assert_close(actual[0], x * 2)
    torch.testing.assert_close(actual[1], x * 9)


def test_capture_omits_unneeded_later_computation(preparation):
    class Program(torch.nn.Module):
        def forward(self, x):
            a = x + 1
            return a * 7

    x = torch.ones(2, 3)
    parent = torch.export.export(Program(), (x,)).module()
    first = next(n for n in parent.graph.nodes if n.op == "call_function")
    capture = preparation.capture_graph(parent, [first])
    assert len([n for n in capture.graph.nodes if n.op == "call_function"]) == 1
    torch.testing.assert_close(capture(x)[0], x + 1)


def test_mutable_source_call_is_refused(preparation):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    mutation = graph.call_function(torch.ops.aten.add_.Tensor, (x, 1))
    graph.output(mutation)
    parent = torch.fx.GraphModule(torch.nn.Module(), graph)
    with pytest.raises(ValueError, match="Mutable"):
        preparation.slice_graph(parent, mutation.name, mutation.name)


def test_integer_reference_uses_checked_wide_contraction(preparation):
    graph = torch.fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    product = graph.call_function(torch.ops.aten._int_mm.default, (a, b))
    graph.output((product,))
    parent = torch.fx.GraphModule(torch.nn.Module(), graph)
    a = torch.tensor([[127, -128, 31], [1, 0, -1]], dtype=torch.int8)
    b = torch.tensor([[127, 4], [-128, 2], [15, -1]], dtype=torch.int8)
    actual = preparation.independent_values(parent, (a, b))[0].numpy()
    np.testing.assert_array_equal(actual, a.numpy().astype(np.int64) @ b.numpy().astype(np.int64))
    assert actual.dtype == np.int32
