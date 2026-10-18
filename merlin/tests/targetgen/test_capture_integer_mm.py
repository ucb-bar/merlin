"""The eager speedup preserves bounded integer values, native fallback and source graphs."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.targetgen import _capture_integer_mm as evaluation

GUARDS = r"""
import json, sys, os

sys.path.insert(0, sys.argv[1])
import torch
from _capture_integer_mm import exact_integer_mm
from torch._subclasses.fake_tensor import FakeTensorMode

torch.set_num_threads(1)
torch.manual_seed(0)
with exact_integer_mm(torch) as availability:
    pass
if availability["status"] != "enabled":
    print(json.dumps({"unavailable": availability}), flush=True)
    raise SystemExit(0)
rows = []


def run(a, b, want):
    before = [x.clone() for x in (a, b)]
    versions = [x._version for x in (a, b)]
    rng = torch.get_rng_state().clone()
    stock = torch._int_mm(a, b)
    with exact_integer_mm(torch) as stats:
        fast = torch._int_mm(a, b)
    assert (
        torch.equal(stock, fast)
        and fast.dtype == torch.int32
        and fast.shape == stock.shape
        and fast.stride() == stock.stride()
    )
    assert (
        all(torch.equal(x, y) for x, y in zip((a, b), before))
        and [x._version for x in (a, b)] == versions
        and torch.equal(torch.get_rng_state(), rng)
    )
    assert stats["accelerated_calls"] == want
    rows.append(
        {"shape": [list(a.shape), list(b.shape)], "strides": [list(a.stride()), list(b.stride())], "stats": stats}
    )


for m, k, n in [(1, 1, 1), (3, 17, 5), (17, 31, 33), (9, 129, 7)]:
    for value in [-128, 127]:
        run(torch.full((m, k), value, dtype=torch.int8), torch.full((k, n), -128, dtype=torch.int8), 1)
    a = torch.randint(-128, 128, (m, k + 7), dtype=torch.int8)[:, :k]
    b = torch.randint(-128, 128, (k, n + 5), dtype=torch.int8)[:, :n]
    run(a, b, 1)
    b = torch.randint(-128, 128, (n, k + 3), dtype=torch.int8)[:, :k].t()
    run(a, b, 1)
k = torch.iinfo(torch.int32).max // max(abs(torch.iinfo(torch.int8).min), abs(torch.iinfo(torch.int8).max)) ** 2
run(torch.full((1, k), -128, dtype=torch.int8), torch.full((k, 1), -128, dtype=torch.int8), 1)
run(torch.full((1, k + 1), -128, dtype=torch.int8), torch.full((k + 1, 1), -128, dtype=torch.int8), 0)
run(torch.randint(-128, 128, (3, 12), dtype=torch.int8)[:, ::2], torch.ones(6, 5, dtype=torch.int8), 0)
for a, b in [
    (torch.ones(3, 4), torch.ones(4, 5)),
    (torch.ones(3, 4, dtype=torch.int8), torch.ones(5, 6, dtype=torch.int8)),
]:
    errors = []
    for use in [False, True]:
        try:
            if use:
                with exact_integer_mm(torch):
                    torch._int_mm(a, b)
            else:
                torch._int_mm(a, b)
        except RuntimeError as e:
            errors.append(str(e))
    assert len(errors) == 2 and errors[0] == errors[1]
with FakeTensorMode():
    a = torch.empty(3, 4, dtype=torch.int8)
    b = torch.empty(4, 5, dtype=torch.int8)
    with exact_integer_mm(torch) as fake_stats:
        out = torch._int_mm(a, b)
    assert fake_stats["accelerated_calls"] == 0 and out.shape == (3, 5)
run(torch.ones(3, 4, dtype=torch.int8), torch.nn.Parameter(torch.ones(4, 5, dtype=torch.int8), requires_grad=False), 0)
a = torch.ones(3, 4, dtype=torch.int8)
b = torch.ones(4, 5, dtype=torch.int8)
out = torch.empty(3, 5, dtype=torch.int32)
with exact_integer_mm(torch) as out_stats:
    returned = torch._int_mm(a, b, out=out)
assert returned.data_ptr() == out.data_ptr() and out_stats["accelerated_calls"] == 0
assert torch.equal(out, torch._int_mm(a, b))
x = torch.randn(3, 4)
b = torch.randn(4, 5)
stock = x @ b
with exact_integer_mm(torch):
    fast = x @ b
assert torch.equal(stock, fast)
fp_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8), torch.nn.GELU(approximate="tanh")).eval()
fx = torch.randn(2, 3, 4)
fp_state = {k: (v.clone(), v._version) for k, v in fp_model.state_dict().items()}
grad = torch.is_grad_enabled()
fp_stock = fp_model(fx)
with exact_integer_mm(torch):
    fp_fast = fp_model(fx)
assert torch.equal(fp_stock, fp_fast) and torch.is_grad_enabled() == grad
assert all(
    torch.equal(v, fp_model.state_dict()[k]) and ver == fp_model.state_dict()[k]._version
    for k, (v, ver) in fp_state.items()
)
a = torch.ones(2, 3, 4, dtype=torch.int8)
b = torch.ones(2, 4, 5, dtype=torch.int8)
vmap_stats = []


def batched(a, b):
    with exact_integer_mm(torch) as stats:
        out = torch._int_mm(a, b)
    vmap_stats.append(stats)
    return out


vmap_stock = torch.vmap(torch._int_mm)(a, b)
vmap_fast = torch.vmap(batched)(a, b)
assert torch.equal(vmap_stock, vmap_fast) and all(s["accelerated_calls"] == 0 for s in vmap_stats)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.randint(-128, 128, (4, 5), dtype=torch.int8))

    def forward(self, x):
        return torch._int_mm(x, self.weight)


m = Model()
x = torch.randint(-128, 128, (3, 4), dtype=torch.int8)
before = torch.export.export(m, (x,))
state_before = {k: v.clone() for k, v in m.state_dict().items()}
with exact_integer_mm(torch):
    m(x)
after = torch.export.export(m, (x,))
assert before.graph_module.code == after.graph_module.code
assert all(torch.equal(v, m.state_dict()[k]) for k, v in state_before.items())
assert any(n.target == torch.ops.aten._int_mm.default for n in after.graph.nodes)
assert all(getattr(n.meta.get("val"), "dtype", None) != torch.float64 for n in after.graph.nodes)
print(
    json.dumps(
        {
            "status": "passed",
            "cases": rows,
            "fake": fake_stats,
            "floating_operators_unchanged": True,
            "source_export_and_state_unchanged": True,
            "float_module_and_grad_state_unchanged": True,
            "vmap_native_preserved": True,
        },
        indent=2,
    ),
    flush=True,
)
"""
TRACING = r"""
import json, os, sys

sys.path.insert(0, sys.argv[1])
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from _capture_integer_mm import exact_integer_mm

torch.set_num_threads(1)
torch.manual_seed(0)
with exact_integer_mm(torch) as availability:
    pass
if availability["status"] != "enabled":
    print(json.dumps({"unavailable": availability}), flush=True)
    raise SystemExit(0)
a = torch.randint(-128, 128, (3, 4), dtype=torch.int8)
b = torch.randint(-128, 128, (4, 5), dtype=torch.int8)
f = lambda a, b: torch._int_mm(a, b)
stock = make_fx(f, tracing_mode="real")(a, b)
with exact_integer_mm(torch) as outer:
    captured = make_fx(f, tracing_mode="real")(a, b)
assert outer["accelerated_calls"] == 0
assert stock.code == captured.code, (stock.code, captured.code)
inside = []


def g(a, b):
    with exact_integer_mm(torch) as stats:
        result = f(a, b)
    inside.append(stats)
    return result


captured = make_fx(g, tracing_mode="real")(a, b)
assert stock.code == captured.code, (stock.code, captured.code)
assert all(s["accelerated_calls"] == 0 for s in inside)
for gm in [stock, captured]:
    assert any(n.target == torch.ops.aten._int_mm.default for n in gm.graph.nodes)
    assert all(getattr(n.meta.get("val"), "dtype", None) != torch.float64 for n in gm.graph.nodes)
    assert torch.equal(gm(a, b), f(a, b))
print(
    json.dumps({"status": "passed", "outer": outer, "inner": inside, "real_make_fx_preserves_int8_i32_graph": True}),
    flush=True,
)
"""


@pytest.fixture(scope="module")
def observed():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the capture interpreter is not configured")
    values = []
    for script in (GUARDS, TRACING):
        run = subprocess.run(
            [python, "-c", script, str(Path(evaluation.__file__).parent)], capture_output=True, text=True, timeout=120
        )
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-5000:]
        value = json.loads(run.stdout)
        if "unavailable" in value:
            pytest.skip("this framework lacks the optional eager evaluation APIs")
        values.append(value)
    return values


def test_signed_extreme_padded_and_boundary_inputs_preserve_native_results(observed):
    cases = observed[0]["cases"]
    assert len(cases) == 20
    assert sum(c["stats"]["accelerated_calls"] for c in cases) == 17
    assert sum(c["stats"]["native_integer_calls"] for c in cases) == 3


def test_float_state_rng_and_source_export_stay_unchanged(observed):
    assert observed[0]["floating_operators_unchanged"]
    assert observed[0]["source_export_and_state_unchanged"]
    assert observed[0]["float_module_and_grad_state_unchanged"]
    assert observed[0]["vmap_native_preserved"]
    assert observed[0]["fake"]["accelerated_calls"] == 0


def test_real_proxy_tracing_in_both_nesting_orders_retains_integer_graph(observed):
    row = observed[1]
    assert row["real_make_fx_preserves_int8_i32_graph"]
    assert row["outer"]["status"] == "enabled" and row["outer"]["accelerated_calls"] == 0
    assert all(v["accelerated_calls"] == 0 for v in row["inner"])


def test_missing_framework_api_uses_native_context():
    from types import SimpleNamespace

    with evaluation.exact_integer_mm(SimpleNamespace()) as stats:
        assert stats["status"] == "native_api_unavailable"
        assert stats["accelerated_calls"] == 0
