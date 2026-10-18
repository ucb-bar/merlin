"""The generic PT2E quantizer under a methodology recipe: the quantized graph IS the output path's arithmetic.

Recipes are derived (:mod:`merlin.targetgen.quant_methodology`) over SYNTHETIC output paths -- the
reference path, one without a rectifier, one with a scale per column, one with another clamp, one
without a bias preload -- and each is applied by ``_recipe_quantizer.apply_recipe`` to the same small
network in the capture interpreter. The converted graphs must differ exactly as the paths do: which
stages a contraction's pattern absorbs, how many distinct device multipliers a layer needs, the
requant clamp, whether a bias rides in the accumulator as an integer code. Every graph must stay an
integer program the independent PT2E integer reference reproduces bit for bit.

Torch lives in the capture interpreter (``MERLIN_M2M_PYTHON``); the structural checks run without it.
"""

# ruff: noqa: E501 - the interpreter script keeps its graph-building calls on one line.

from __future__ import annotations

import json
import os
import subprocess

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import quant_methodology as QM
from merlin.targetgen import quant_recipe as QR

_DIR = merlin_dir() / "python" / "merlin" / "targetgen"

_METHODOLOGY = {
    "scale_granularity": "from_output_path",
    "bias": "accumulator_preload_int32",
    "epilogue": "integer_domain",
    "rounding": "from_output_path",
    "clamp": "from_output_path",
    "calibration": {"activation": "mse", "weight": "minmax"},
    "recovery": {"cross_layer_equalization": True, "bias_correction": True},
    "float_island": {"dtype": "fp32", "ops": ["softmax"]},
}


def _path(*, relu=True, scale=None, clamp=(-128, 127), preload=True) -> dict:
    readout = [{"kind": "activation", "function": "relu", "domain": "accumulator"}] if relu else []
    readout += [
        {"kind": "convert", "to": "f32", "rounding": "half_even"},
        {
            "kind": "scale",
            "dtype": "f32",
            "rounding": "half_even",
            **(scale or {"granularity": "per_command", "columns_per_scale": 16}),
        },
        {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": list(clamp)},
        {"kind": "pool", "reductions": ["max"], "padding": "unknown"},
    ]
    stages = (
        [{"kind": "preload", "dtype": "i32", "range": [-(2**31), 2**31 - 1], "rounding": "none"}] if preload else []
    )
    return {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": stages,
        "readout": readout,
        "exits": {"accumulator": True, "requant": True},
    }


PATHS = {
    "reference": _path(),
    "no_relu": _path(relu=False),
    "per_column": _path(scale={"granularity": "per_channel", "columns_per_scale": None}),
    "clamp_127": _path(clamp=(-127, 127)),
    "no_preload": _path(preload=False),
    "padded_pool": _path(),
}


def _recipes() -> dict[str, dict]:
    base = QR.QuantRecipe(
        target="t",
        unit="mesh",
        weight=QR.TensorSpec("int8", "tensor", True, -127, 127),
        activation=QR.TensorSpec("int8", "tensor", True, -128, 127),
        families=(QR.CONTRACTION,),
    ).to_dict() | {"accumulator_dtype": "int32"}
    recipes = {name: QM.derive(base, _METHODOLOGY, path) for name, path in PATHS.items()}
    recipes["partitioned_per_column"] = QM.derive(
        base, {**_METHODOLOGY, "scale_granularity": "per_channel"}, _path(),
    )
    return recipes


def test_the_quantizer_reads_the_epilogue_block_through_its_sibling() -> None:
    quantizer = (_DIR / "_recipe_quantizer.py").read_text(encoding="utf-8")
    epilogue = (_DIR / "_integer_epilogue.py").read_text(encoding="utf-8")
    assert 'recipe.get("epilogue")' in quantizer and "import _integer_epilogue as IE" in quantizer
    assert "DerivedQuantizationSpec" in epilogue and "def equalize" in epilogue and "def correct_biases" in epilogue
    assert all(r["status"] == QR.DERIVED for r in _recipes().values())


@pytest.mark.parametrize("change", [
    {"logical_granularity": "per_tensor"}, {"columns_per_scale": False},
    {"columns_per_scale": 0}, {"granularity": "per_tensor"},
])
def test_logical_channel_policy_cannot_replace_physical_readout_facts(change):
    from copy import deepcopy

    from merlin.targetgen import _integer_epilogue as IE

    block = deepcopy(_recipes()["partitioned_per_column"]["epilogue"])
    block["scale"].update(change)
    with pytest.raises(IE.EpilogueError):
        IE.check(block)


_SCRIPT = r"""
import json
import sys
sys.path.insert(0, sys.argv[1])
import torch
import torchao  # noqa: F401  (registers the quantized_decomposed ops)
import _recipe_quantizer as RQ

class Net(torch.nn.Module):
    def __init__(self, padded_pool=False):
        super().__init__()
        self.pool_padding = 1 if padded_pool else 0
        self.c1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = torch.nn.Conv2d(32, 16, 1)
        self.fc = torch.nn.Linear(16 * (5 if padded_pool else 4) ** 2, 10)
    def forward(self, x):
        y = torch.nn.functional.max_pool2d(torch.relu(self.c1(x)), 2, padding=self.pool_padding)
        y = torch.relu(self.c2(y))
        return self.fc(self.c3(y).flatten(1))

qd = torch.ops.quantized_decomposed
out = {}
for name, path in json.loads(sys.argv[2]).items():
    recipe = json.load(open(path))
    torch.manual_seed(0)
    model = Net(padded_pool=name == "padded_pool").eval()
    x = torch.randn(1, 3, 8, 8)
    calibration = [(torch.randn(1, 3, 8, 8),) for _ in range(6)]
    q = RQ.apply_recipe(model, recipe, example_inputs=(x,), calibration_inputs=calibration)
    census = q._recipe_quantization_stats["integer_epilogue"]
    sites = {s["node"]: s for s in census["kept"]}
    graph = q.graph
    nodes = {n.name: n for n in graph.nodes}
    convs = [n for n in graph.nodes if n.op == "call_function" and n.target in (torch.ops.aten.conv2d.default, torch.ops.aten.linear.default)]
    first = convs[0]
    users = [u.target for u in first.users]
    scales = [getattr(q, n.target) for n in graph.nodes if n.op == "get_attr" and str(n.target).endswith("_scale")]
    quantizes = [n for n in graph.nodes if n.op == "call_function" and n.target == qd.quantize_per_tensor.default and n.args[1] == 1.0]
    bias_codes = [n for n in graph.nodes if n.op == "get_attr" and str(n.target).endswith("_bias") and not str(n.target).endswith("_host_bias")]
    float_biases = [n for n in graph.nodes if n.op == "get_attr" and str(n.target).endswith("_host_bias")]
    unit_operands = all(c.args[0].target == qd.dequantize_per_tensor.default and c.args[0].args[1] == 1.0 for c in convs)
    pools = [n for n in graph.nodes if n.op == "call_function" and n.target == torch.ops.aten.max_pool2d.default]
    before_pool = pools[0].args[0] if pools else None
    if before_pool is not None and before_pool.target == torch.ops.aten.relu.default:
        before_pool = before_pool.args[0]
    result = {
        "host_pool_input_requantized": bool(before_pool is not None and before_pool.target == qd.dequantize_per_tensor.default),
        "by_exit": census["by_exit"], "absorbing": census["absorbing"], "kept": len(census["kept"]),
        "first_reads": [str(t) for t in users],
        "first_distinct_multipliers": int(torch.unique(scales[0]).numel()) if scales else 0,
        "unit_requant_clamps": sorted({(int(n.args[3]), int(n.args[4])) for n in quantizes}),
        "bias_codes": len(bias_codes), "float_biases": len(float_biases), "unit_operands": unit_operands,
        "equalized": len(census["equalized_pairs"]), "corrected": census["bias_corrections"]["corrected"],
    }
    from m2m.capture.pt2e_integerize import integerize_pt2e
    from m2m.capture import pt2e_integer_reference as IR
    independent = IR.run_pt2e_integer_reference(q, (x,))
    integer, receipt = integerize_pt2e(q, (x,))
    result["integerized"] = receipt["quantized_contractions_integerized"]
    result["bit_exact"] = bool(torch.equal(independent.output, integer(x)))
    out[name] = result
print(json.dumps(out))
"""


def test_each_output_path_yields_its_own_integer_epilogue(tmp_path) -> None:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not os.path.exists(python):
        pytest.skip("the model2MLIR torch interpreter is not configured")
    files = {}
    for name, recipe in _recipes().items():
        files[name] = str(tmp_path / f"{name}.json")
        (tmp_path / f"{name}.json").write_text(json.dumps(recipe), encoding="utf-8")
    completed = subprocess.run(
        [python, "-c", _SCRIPT, str(_DIR), json.dumps(files)], capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    got = json.loads(completed.stdout.strip().splitlines()[-1])
    ref = got["reference"]
    # The reference path: conv+relu+pool, conv+relu and the conv read only by the Linear leave requantized;
    # the Linear's logits leave through the full-width exit.
    assert ref["by_exit"] == {"requant": 3, "accumulator": 1} and ref["absorbing"] == {"relu": 2, "pool": 1, "bias": 4}
    assert ref["first_reads"] == ["aten.relu.default"]  # the rectifier reads the accumulator, before the scale
    assert ref["first_distinct_multipliers"] <= 2  # 32 columns, one multiplier per 16
    assert ref["unit_requant_clamps"] == [[-128, 127]] and ref["bias_codes"] == 4 and ref["float_biases"] == 0
    assert ref["unit_operands"] and ref["equalized"] >= 1 and ref["corrected"] == 4 and ref["kept"] == 0
    # No rectifier on the path: a contraction read by a host relu leaves full width; only the one read by
    # the Linear still requantizes.
    plain = got["no_relu"]
    assert (
        plain["absorbing"]["relu"] == 0
        and plain["absorbing"]["pool"] == 0
        and plain["by_exit"] == {"requant": 1, "accumulator": 3}
    )
    assert plain["first_reads"] == ["aten.mul.Tensor"]
    # A scale per column: one multiplier per channel.
    assert got["per_column"]["first_distinct_multipliers"] > 2
    assert got["partitioned_per_column"]["first_distinct_multipliers"] > 2
    assert got["partitioned_per_column"]["kept"] == 0
    assert got["partitioned_per_column"]["by_exit"] == got["per_column"]["by_exit"]
    # Another clamp: the requantizing exit clamps to it.
    assert got["clamp_127"]["unit_requant_clamps"] == [[-127, 127]]
    # No preload: the bias stays a float add on the host and no pattern requantizes across it.
    host = got["no_preload"]
    assert (
        host["bias_codes"] == 0 and host["float_biases"] == 4 and host["by_exit"]["requant"] == 0 and host["kept"] == 0
    )
    # Undeclared pool padding leaves through the available full-width exit. Quantizing before
    # that host operation would split an integer-to-float shell away from its device producer.
    padded = got["padded_pool"]
    assert padded["by_exit"] == {"requant": 2, "accumulator": 2}
    assert padded["absorbing"] == {"relu": 1, "pool": 0, "bias": 4}
    assert padded["first_reads"] == ["aten.mul.Tensor"] and not padded["host_pool_input_requantized"]
    for name, row in got.items():
        assert row["bit_exact"] and row["integerized"] == 4, name


_BOUND_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
import _recipe_quantizer as RQ
recipe = json.loads(sys.argv[2])
recipe['recovery'] = {'cross_layer_equalization': False, 'bias_correction': False}
torch.manual_seed(0)
out = {}
for width in (7, 1100):
    layer = torch.nn.Linear(width, 19).eval()
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.fill_(0.125)
    sample = (torch.ones(1, width),)
    quantized = RQ.apply_recipe(layer, recipe, example_inputs=sample, calibration_inputs=[sample])
    out[width] = quantized._recipe_quantization_stats['integer_epilogue']
print(json.dumps(out))
"""


def test_biased_accumulators_must_fit_the_exact_f32_integer_carrier(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not os.path.exists(python):
        pytest.skip("the model2MLIR torch interpreter is not configured")
    completed = subprocess.run(
        [python, "-c", _BOUND_SCRIPT, str(_DIR), json.dumps(_recipes()["reference"])],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    rows = json.loads(completed.stdout.strip().splitlines()[-1])
    assert rows["7"]["rewritten"] == 1 and not rows["7"]["kept"]
    assert rows["1100"]["rewritten"] == 0
    assert "exact integer range" in rows["1100"]["kept"][0]["why"]
