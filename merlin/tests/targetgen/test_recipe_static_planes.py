"""A recipe's ``static_planes`` block widens a Linear's static activation grid into int8 digit planes.

WHY THIS EXISTS. One static per-tensor int8 scale per activation costs SmolVLA's prefix too much range:
measured with GPTQ weights, a static int8 activation per tensor on every Linear misses the fp32-referenced
bar, while the same scale on a 15-bit grid (two int8 digits, the top one signed, the low one in [0, 127])
passes it (0.047 / 0.100 against 0.069 / 0.104). The unit multiplies int8 by int8, so the wide activation
is sent as two digit planes against the SAME int8 weight, and the host sums ``128 * acc_top + acc_low``
inside the per-column rescale it applies anyway: two device contractions, no extra host pass. The scale
stays ONE static number per tensor, calibrated by PT2E's own observer on the wide grid.

What this pins:

* ``quant_recipe.with_static_planes`` states the block (digits, observer, owner scope), refuses what it
  cannot state, and changes the content digest; ``with_activation_observer`` names a percentile observer;
* with the block absent, the converted graph has no digit planes;
* with it, every Linear in scope becomes one ``_int_mm`` per digit, a quantized activation read by
  several Linears is split once, a Linear out of scope keeps its int8 PT2E form, and each rewritten
  output equals an independent statement of the arithmetic (wide-grid quantize, per-channel int8
  weight, exact integer sum) within float rounding;
* the percentile observer's range is the percentile of ``|x|`` to within one histogram bin;
* under GPTQ weight rounding the planes multiply GPTQ's codes (the split happens after convert, so GPTQ
  saw the float activation), and the dynamic block, which rounds stored weights itself, refuses GPTQ.

The quantizer runs in the capture venv (``MERLIN_M2M_PYTHON``); that half is skipped without it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import quant_recipe as QR

_QUANTIZER = merlin_dir() / "python/merlin/targetgen/_recipe_quantizer.py"


def _recipe(weight_granularity: str = "channel") -> dict:
    body = {
        "schema": QR.SCHEMA,
        "target": "t",
        "unit": "u",
        "status": QR.DERIVED,
        "families": [QR.CONTRACTION],
        "weight": {
            "dtype": "int8",
            "granularity": weight_granularity,
            "symmetric": True,
            "quant_min": -127,
            "quant_max": 127,
        },
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
            "observer": "minmax",
        },
        "bias_domain": None,
        "unquantized": {},
        "why": {},
        "underivable": {},
    }
    body["recipe_sha256"] = QR.digest(body)
    return body


# --- the recipe block ------------------------------------------------------------------------------


def test_the_block_states_digits_observer_and_scope_and_changes_the_digest():
    base = _recipe()
    body = QR.with_static_planes(base, owners=["front", "front"])
    assert body[QR.STATIC_PLANES] == {
        "schema": QR.STATIC_PLANES_SCHEMA,
        "families": [QR.CONTRACTION],
        "planes": 2,
        "observer": "minmax",
        "owners": ["front"],
    }
    assert body["recipe_sha256"] == QR.digest(body) != base["recipe_sha256"]
    assert QR.with_static_planes(base)[QR.STATIC_PLANES]["owners"] is None
    observed = QR.with_activation_observer(base, "p99.999")
    assert observed["activation"]["observer"] == "p99.999" and observed["recipe_sha256"] == QR.digest(observed)


@pytest.mark.parametrize(
    ("change", "kwargs", "reason"),
    [
        ({"activation": {"mode": "dynamic"}}, {}, "static per-tensor"),
        ({"activation": {"granularity": "token"}}, {}, "static per-tensor"),
        ({"families": [QR.WINDOW_MEAN]}, {}, "quantizes contractions"),
        ({}, {"planes": 1}, "2 or 3 digits"),
        ({}, {"planes": 4}, "2 or 3 digits"),
        ({}, {"observer": "p0"}, "p<percent>"),
        ({}, {"observer": "mse"}, "p<percent>"),
        ({}, {"owners": []}, "module-path prefixes"),
    ],
)
def test_what_the_block_cannot_state_is_refused(change, kwargs, reason):
    recipe = _recipe()
    for key, value in change.items():
        recipe[key] = {**recipe[key], **value} if isinstance(value, dict) else value
    with pytest.raises(ValueError, match=reason):
        QR.with_static_planes(recipe, **kwargs)


def test_the_block_and_the_observer_have_a_command_line(tmp_path):
    source, wide, observed = tmp_path / "in.json", tmp_path / "wide.json", tmp_path / "observed.json"
    source.write_text(json.dumps(_recipe()))
    assert QR._main(["static-planes", str(source), str(wide), "--owners", "a.b,c"]) == 0
    assert json.loads(wide.read_text())[QR.STATIC_PLANES]["owners"] == ["a.b", "c"]
    assert QR._main(["activation-observer", str(wide), str(observed), "p99.9"]) == 0
    body = json.loads(observed.read_text())
    assert body["activation"]["observer"] == "p99.9" and QR.STATIC_PLANES in body


# --- the rewrite (capture venv) --------------------------------------------------------------------

_SCRIPT = r"""
import importlib.util
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source.parent))
sys.path.insert(0, str(source.parents[2]))
import torch
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

spec = importlib.util.spec_from_file_location("recipe_quantizer", source)
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)
recipe = json.loads(sys.argv[2])


class Front(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(48, 40)
        self.b = nn.Linear(48, 24, bias=False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = Front()
        self.back = nn.Linear(64, 16)

    def forward(self, x):
        a, b = self.front.a(x), self.front.b(x)  # two readers of one quantized activation
        return a, b, self.back(torch.relu(torch.cat([a, b], dim=-1)))


torch.manual_seed(0)
model = Block().eval()
samples = [(torch.randn(2, 9, 48) * (1 + i),) for i in range(4)]
graph = torch.export.export(model, samples[0]).module()
q = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model))
prepared = prepare_pt2e(graph, q)
with torch.no_grad():
    for sample in samples:
        prepared(*sample)
converted = convert_pt2e(prepared, fold_quantize=True)
rows = quant._rewrite_planes(converted, recipe, q.annotated_planes)
nodes = list(converted.graph.nodes)
count = lambda target: sum(n.target == target for n in nodes)
qd = torch.ops.quantized_decomposed
result = {
    "rows": rows,
    "annotated": q.annotated,
    "annotated_planes": q.annotated_planes,
    "int_mm": count(torch.ops.aten._int_mm.default),
    "floor": count(torch.ops.aten.floor.default),
    "quantize": count(qd.quantize_per_tensor.default),
}
if rows:
    planes = rows[0]["planes"]
    base = recipe["activation"]["quant_max"] + 1
    span = base ** (planes - 1)
    low, high = recipe["activation"]["quant_min"] * span, recipe["activation"]["quant_max"] * span + span - 1
    x = samples[1][0]
    with torch.no_grad():
        got = converted(x)
    errors = {}
    for row, linear, out in ((r, model.front.a if r["owner"] == "front.a" else model.front.b, None) for r in rows):
        # The independent statement: the wide-grid quantize of x and the per-channel int8 weight (both
        # rounded as PT2E rounds: times the f32 reciprocal), then an exact integer sum.
        s = row["scale"]
        xq = torch.clamp(torch.round(x * (1.0 / s)), low, high).double()
        w = linear.weight.detach()
        sw = w.abs().amax(dim=1, keepdim=True) / 127.0
        wq = torch.clamp(torch.round(w * (1.0 / sw)), -127, 127).double()
        want = (xq @ wq.T) * s * sw.double().T
        if linear.bias is not None:
            want = want + linear.bias.detach().double()
        index = 0 if linear is model.front.a else 1
        errors[row["owner"]] = float((got[index].double() - want).abs().max() / want.abs().max())
    result["errors"] = errors
    result["first_layer_scale"] = rows[0]["scale"]
    result["calibrated_absmax"] = max(float(s[0].abs().max()) for s in samples)
from _recipe_observers import percentile_observer, LOW, PER_OCTAVE

observer = percentile_observer().with_args(percentile=99.0)(quant_min=-128, quant_max=127)
data = torch.randn(200000)
observer(data)
result["percentile_clip"] = observer.clip()
result["percentile_true"] = float(data.abs().quantile(0.99))
result["percentile_scale"] = float(observer.calculate_qparams()[0])
print(json.dumps(result, default=str))
"""


def _run(recipe: dict) -> dict:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    completed = subprocess.run(
        [python, "-c", _SCRIPT, str(_QUANTIZER), json.dumps(recipe)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_without_the_block_no_linear_takes_planes():
    data = _run(_recipe())
    assert data["rows"] == [] and data["annotated_planes"] == 0 and data["annotated"] == 3
    assert data["int_mm"] == 0 and data["floor"] == 0


@pytest.mark.parametrize("planes", [2, 3])
def test_linears_in_scope_take_digit_planes_against_one_int8_weight(planes):
    data = _run(QR.with_static_planes(_recipe(), planes=planes, owners=["front"]))
    rows = data["rows"]
    assert sorted(r["owner"] for r in rows) == ["front.a", "front.b"] and data["annotated_planes"] == 2
    assert all(r["status"] == "rewritten" and r["planes"] == planes and r["device_calls"] == planes for r in rows)
    assert data["int_mm"] == 2 * planes, "one device contraction per digit plane per Linear"
    assert data["floor"] == planes - 1, "the shared activation is split into digits once"
    assert data["annotated"] == 1 and data["quantize"] == 1, "the Linear out of scope keeps its int8 PT2E form"
    assert all(r["host_elements"] > 0 and r["int8_macs"] == planes * 18 * 48 * r["shape"][2] for r in rows)
    assert all(error < 1e-6 for error in data["errors"].values()), data["errors"]
    span = 128 ** (planes - 1)
    grid = (128 * span + 127 * span + span - 1) / 2
    assert data["first_layer_scale"] == pytest.approx(data["calibrated_absmax"] / grid, rel=1e-6)


def test_the_percentile_observer_ranges_at_the_percentile_of_magnitudes():
    data = _run(QR.with_activation_observer(_recipe(), "p99"))
    assert data["percentile_true"] <= data["percentile_clip"] <= data["percentile_true"] * 2 ** (1 / 32) * 1.0001
    assert data["percentile_scale"] == pytest.approx(data["percentile_clip"] / 127.5, rel=1e-6)


# --- with GPTQ weight codes (capture venv, through apply_recipe) ----------------------------------

_APPLY = r"""
import json
import sys
from pathlib import Path

targetgen, m2m_dir = Path(sys.argv[1]), sys.argv[2]
for path in (str(targetgen), str(targetgen.parents[1]), m2m_dir):
    sys.path.insert(0, path)
import torch
from torch import nn

import _recipe_quantizer as RQ

planes, dynamic = json.loads(sys.argv[3]), json.loads(sys.argv[4])
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(48, 40), nn.ReLU(), nn.Linear(40, 16)).eval()
samples = [(torch.randn(16, 48),) for _ in range(8)]
quantized = RQ.apply_recipe(model, planes, example_inputs=samples[0], calibration_inputs=samples)
stats = quantized._recipe_quantization_stats
with torch.no_grad():
    got, want = quantized(*samples[1]), model(*samples[1])
result = {
    "rewritten": stats["static_planes"]["rewritten"],
    "annotated": stats["annotated_contractions"],
    "rounded": stats.get("weight_rounding") is not None,
    "error": float((got - want).abs().max() / want.abs().max()),
}
try:
    RQ.build_quantizer(dynamic, layer_plan=RQ._plan_layers(dynamic, model))
    result["refused"] = None
except RQ.RecipeError as error:
    result["refused"] = str(error)
print(json.dumps(result))
"""


def test_planes_take_the_gptq_codes_and_the_dynamic_block_refuses_them():
    python, m2m_dir = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m_dir:
        pytest.skip("no capture interpreter and model2MLIR checkout are configured")
    gptq = QR.with_gptq_weights(_recipe(), calibration_samples=8)
    planes = QR.with_static_planes(gptq)
    dynamic = QR.with_dynamic_contractions(gptq, families=[QR.CONTRACTION], static_families=[])
    completed = subprocess.run(
        [python, "-c", _APPLY, str(_QUANTIZER.parent), m2m_dir, json.dumps(planes), json.dumps(dynamic)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    data = json.loads(completed.stdout.strip().splitlines()[-1])
    assert data["rewritten"] == 2 and data["annotated"] == 0 and data["rounded"]
    assert data["error"] < 0.02, data["error"]
    assert data["refused"] and "GPTQ" in data["refused"]
