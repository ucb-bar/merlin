"""Integer-domain numerical preparation must retain explicit bounds and refusals."""

import importlib.util

import pytest

from merlin.common.paths import repo_root

torch = pytest.importorskip("torch")


@pytest.fixture
def prepare(monkeypatch):
    path = repo_root() / "examples/gemmini/comparisons/tvm"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location("accumulator_bias_test", path / "prepare_accumulator_bias.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def program(bias):
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.full((3, 4), 127, dtype=torch.int8))
            self.register_buffer("bias", torch.full((4,), bias, dtype=torch.float32))

        def forward(self, x):
            return torch.ops.aten._int_mm.default(x, self.weight).float() + self.bias

    x = torch.tensor([[1, -2, 3], [-4, 5, 6]], dtype=torch.int8)
    return torch.export.export(Module(), (x,)).module(), x


def test_bias_moves_into_int32_with_proved_bound(prepare):
    parent, x = program(17)
    expected = parent(x)
    module, records = prepare.normalize(parent)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    graph = torch.export.export(module, (x,)).graph
    additions = [n for n in graph.nodes if n.op == "call_function" and n.target == torch.ops.aten.add.Tensor]
    assert additions[0].meta["val"].dtype == torch.int32
    assert records[0]["original_fp32_order_proved_exact"]
    assert records[0]["biased_abs_bound"] < 2**31


def test_fractional_bias_is_not_silently_rounded(prepare):
    parent, _ = program(0.25)
    with pytest.raises(ValueError, match="integral"):
        prepare.normalize(parent)


def test_possible_accumulator_overflow_is_refused(prepare):
    parent, _ = program(2147483520)
    with pytest.raises(ValueError, match="overflow"):
        prepare.normalize(parent)
