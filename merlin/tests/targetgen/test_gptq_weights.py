"""A recipe may choose its int8 weight codes by GPTQ instead of rounding them to nearest.

WHY THIS EXISTS. Round-to-nearest puts each weight on the recipe's grid independently, so errors of
weights that read correlated inputs add up in the layer's output. Measured on a vision-language-action
model with per-channel int8 weights and float activations, nearest rounding gave 3.4x the float noise
floor and GPTQ 1.4x. The option is ``weight.rounding = "gptq"`` with its parameters under
``weight.gptq`` (:func:`quant_recipe.with_gptq_weights`); the algorithm and the capture hook are in
``_gptq.py``. This file holds:

* the recipe switch: default recipes are byte-for-byte what they were (no key, same digest), the
  variant records every parameter in the digest, and unrealisable recipes are refused with the reason;
* the algorithm on small linears: GPTQ's layer-output error is no worse than nearest on correlated
  inputs and strictly lower on a constructed case, the codes stay in [-127, 127] at every granularity
  (per tensor, per output channel, per block of output channels or of reduced elements), and the
  result is deterministic across processes;
* the capture: a toy model through ``apply_recipe`` with the option carries in its converted program
  exactly the int8 codes the module's own GPTQ computes from the same calibration inputs and the
  observer's scales, and leaves the source model's weights alone.

The algorithm and the capture run in the capture venv (``MERLIN_M2M_PYTHON``); without it those
tests skip. The recipe half needs no torch.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import _gptq as G
from merlin.targetgen import quant_recipe as QR

_TARGETGEN = merlin_dir() / "python/merlin/targetgen"
_WEIGHT_KEYS = {"dtype", "granularity", "symmetric", "quant_min", "quant_max", "block", "mode"}


def _recipe(*, weight: dict | None = None, **activation) -> dict:
    body = QR.QuantRecipe(
        target="t",
        unit="u",
        weight=QR.TensorSpec(dtype="int8", granularity="channel", symmetric=True, quant_min=-127, quant_max=127),
        activation=QR.TensorSpec(dtype="int8", granularity="tensor", symmetric=True, quant_min=-128, quant_max=127),
    ).to_dict()
    body["weight"].update(weight or {})
    body["activation"].update(activation)
    body["recipe_sha256"] = QR.digest(body)
    return body


# --- the recipe switch -----------------------------------------------------------------------------


def test_a_default_recipe_carries_no_rounding_key_and_keeps_its_digest():
    base = _recipe()
    assert set(base["weight"]) == _WEIGHT_KEYS, "a default weight spec gains no key"
    assert G.rounding_of(base["weight"]) is None
    variant = QR.with_gptq_weights(base)
    assert variant["recipe_sha256"] == QR.digest(variant) != base["recipe_sha256"]
    stripped = json.loads(json.dumps(variant))
    del stripped["weight"]["rounding"], stripped["weight"]["gptq"]
    assert QR.digest(stripped) == base["recipe_sha256"], "only the option moves the digest"


def test_the_variant_records_every_parameter_and_only_touches_the_weight():
    base = _recipe()
    variant = QR.with_gptq_weights(base, calibration_samples=8, act_order=False)
    assert variant["weight"]["rounding"] == G.GPTQ
    assert variant["weight"]["gptq"] == {"calibration_samples": 8, "act_order": False, "damp": 0.01, "block": 128}
    assert list(variant["weight"]["gptq"]) == list(G.PARAMETERS)
    assert G.rounding_of(variant["weight"]) == variant["weight"]["gptq"]
    assert "weight.rounding" in variant["why"]
    for key in ("activation", "families", "unquantized", "underivable", "target", "bias_domain"):
        assert variant[key] == base[key], key
    assert {k: v for k, v in variant["weight"].items() if k not in ("rounding", "gptq")} == base["weight"]
    assert "rounding" not in base["weight"], "the input recipe is not mutated"
    assert QR.with_gptq_weights(variant, calibration_samples=8, act_order=False) == variant
    for name, value in (("calibration_samples", 32), ("act_order", True), ("damp", 0.1), ("block", 64)):
        other = QR.with_gptq_weights(base, **{"calibration_samples": 8, "act_order": False, name: value})
        assert other["recipe_sha256"] != variant["recipe_sha256"], f"{name} is part of the digest"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"activation": {"mode": "dynamic", "granularity": "token"}}, "static path"),
        ({"weight": {"dtype": "fp8_e4m3", "quant_min": None, "quant_max": None}}, "symmetric integer"),
        ({"weight": {"symmetric": False}}, "symmetric integer"),
        ({"weight": {"granularity": "block", "block": 32}}, "no static spec"),
    ],
)
def test_an_unrealisable_recipe_is_refused_with_the_reason(change, reason):
    recipe = _recipe(weight=change.get("weight"), **change.get("activation", {}))
    with pytest.raises(ValueError, match=reason):
        QR.with_gptq_weights(recipe)


def test_a_recipe_without_contractions_or_not_derived_is_refused():
    with pytest.raises(ValueError, match="no contraction"):
        QR.with_gptq_weights({**_recipe(), "families": [QR.WINDOW_MEAN]})
    with pytest.raises(ValueError, match="derived"):
        QR.with_gptq_weights({**_recipe(), "status": QR.UNDERIVABLE})


@pytest.mark.parametrize(
    "parameters",
    [{"calibration_samples": 0}, {"calibration_samples": 1.5}, {"act_order": 1}, {"damp": 0}, {"block": -1}],
)
def test_invalid_parameters_are_refused(parameters):
    with pytest.raises(G.GPTQError):
        QR.with_gptq_weights(_recipe(), **parameters)


def test_an_unknown_parameter_is_refused():
    with pytest.raises(G.GPTQError, match="unknown"):
        QR.with_gptq_weights(_recipe(), group_size=8)


def test_the_quantizer_refuses_a_rounding_it_does_not_know():
    assert G.rounding_of({"rounding": G.NEAREST}) is None
    with pytest.raises(G.GPTQError, match="not one this build realises"):
        G.rounding_of({"rounding": "stochastic"})
    with pytest.raises(G.GPTQError, match="weight.gptq"):
        G.rounding_of({"rounding": G.GPTQ})
    with pytest.raises(G.GPTQError, match="rounds to nearest"):
        G.rounding_of({"gptq": dict(G.DEFAULT_PARAMETERS)})


def test_the_switch_has_a_command_line(tmp_path):
    source = tmp_path / "recipe.json"
    source.write_text(json.dumps(_recipe()))
    out = tmp_path / "variant" / "recipe.json"
    argv = ["gptq-weights", str(source), str(out), "--calibration-samples", "8", "--no-act-order", "--block", "64"]
    assert QR._main(argv) == 0  # noqa: SLF001
    expected = QR.with_gptq_weights(_recipe(), calibration_samples=8, act_order=False, block=64)
    assert json.loads(out.read_text()) == expected
    assert QR._main(["gptq-weights", str(source), str(out)]) == 0  # noqa: SLF001
    assert json.loads(out.read_text()) == QR.with_gptq_weights(_recipe())


# --- the algorithm (capture venv) ------------------------------------------------------------------

_ALGORITHM = r"""
import hashlib
import json
import sys

sys.path.insert(0, sys.argv[1])
import torch

import _gptq as G

result = {"cases": [], "granularities": {}}
GRANULARITIES = {
    "tensor": {"granularity": "tensor"},
    "channel": {"granularity": "channel"},
    "output_block": {"granularity": "block", "block": 16, "block_axis": "output"},
    "reduction_block": {"granularity": "block", "block": 16, "block_axis": "reduction"},
}


def layer_error(x, w, codes, scales):
    return float(((x.double() @ (codes.double() * scales.double() - w.double()).T) ** 2).sum())


def run(w, x, scales, **options):
    h = G.HessianSum(w.shape[1])
    for rows in x.split(64):
        h.add(rows)
    codes, info = G.gptq_codes(w, h.value(), scales, quant_min=-127, quant_max=127, **options)
    return codes, info


digest = hashlib.sha256()
for seed in range(4):
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(256, 6, generator=generator)
    x = latent @ torch.randn(6, 48, generator=generator) + 0.1 * torch.randn(256, 48, generator=generator)
    w = torch.randn(40, 48, generator=generator)
    w[3, 7] = 9.0  # an outlier: its row's grid is coarse, and compensation must stay in range
    for name, spec in GRANULARITIES.items():
        scales = G.symmetric_scales(w, quant_min=-127, quant_max=127, **spec)
        for act_order in (True, False):
            codes, info = run(w, x, scales, damp=0.01, act_order=act_order, block=16)
            again, _ = run(w, x, scales, damp=0.01, act_order=act_order, block=16)
            nearest = G.nearest_codes(w, scales, quant_min=-127, quant_max=127)
            digest.update(codes.numpy().tobytes())
            result["cases"].append({
                "seed": seed, "granularity": name, "act_order": act_order,
                "nearest": layer_error(x, w, nearest, scales), "gptq": layer_error(x, w, codes, scales),
                "dtype": str(codes.dtype), "shape": list(codes.shape),
                "min": int(codes.min()), "max": int(codes.max()), "repeatable": bool(torch.equal(codes, again)),
            })
        if seed == 0:
            full = torch.broadcast_to(scales, w.shape)
            result["granularities"][name] = {
                "distinct_per_row": [len(set(row.tolist())) for row in full[:2]],
                "rows_equal_in_first_block": bool(torch.equal(full[0], full[15])),
                "rows_equal_across_blocks": bool(torch.equal(full[15], full[16])),
            }

# Constructed: x2 and x3 carry the same signal, and w2, w3 both sit 0.4 of a step above a code. Nearest
# rounding makes the same error twice; GPTQ pushes the first error into the second weight.
generator = torch.Generator().manual_seed(7)
signal = torch.randn(512, 1, generator=generator)
x = torch.cat([torch.randn(512, 1, generator=generator), signal, signal], dim=1)
x = x + 1e-3 * torch.randn(512, 3, generator=generator)
step = 1.0 / 127
w = torch.tensor([[1.0, 10.4 * step, 20.4 * step]])
scales = G.symmetric_scales(w, granularity="tensor", quant_min=-127, quant_max=127)
codes, _ = run(w, x, scales, damp=0.01, act_order=True, block=128)
nearest = G.nearest_codes(w, scales, quant_min=-127, quant_max=127)
result["constructed"] = {
    "nearest_codes": nearest.tolist(), "gptq_codes": codes.tolist(),
    "nearest": layer_error(x, w, nearest, scales), "gptq": layer_error(x, w, codes, scales),
}

# A column whose input never moved has no evidence: rounded to nearest, never zeroed.
x = torch.randn(128, 8, generator=generator)
x[:, 2] = 0
w = torch.randn(5, 8, generator=generator)
scales = G.symmetric_scales(w, granularity="channel", quant_min=-127, quant_max=127)
codes, info = run(w, x, scales, damp=0.01, act_order=True, block=4)
nearest = G.nearest_codes(w, scales, quant_min=-127, quant_max=127)
same = bool(torch.equal(codes[:, 2], nearest[:, 2]))
result["dead"] = {"dead_columns": info["dead_columns"], "column_is_nearest": same}
result["digest"] = digest.hexdigest()
print(json.dumps(result))
"""


def _capture_python() -> str:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    return python


def _run(script: str, *args: str) -> dict:
    completed = subprocess.run(
        [_capture_python(), "-I", "-B", "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def algorithm() -> dict:
    return _run(_ALGORITHM, str(_TARGETGEN))


def test_gptq_is_no_worse_than_nearest_on_correlated_inputs(algorithm):
    cases = algorithm["cases"]
    assert len(cases) == 4 * 4 * 2
    for case in cases:
        assert case["gptq"] <= case["nearest"], case
    # Correlated inputs are what GPTQ exploits: on these it is far better, not marginally.
    assert sum(case["gptq"] for case in cases) < 0.5 * sum(case["nearest"] for case in cases)


def test_gptq_is_strictly_lower_on_the_constructed_case(algorithm):
    constructed = algorithm["constructed"]
    assert constructed["nearest_codes"] == [[127, 10, 20]]
    assert constructed["gptq_codes"] in ([[127, 10, 21]], [[127, 11, 20]]), constructed
    assert constructed["gptq"] < 0.25 * constructed["nearest"], constructed


def test_codes_stay_on_the_int8_grid_at_every_granularity(algorithm):
    for case in algorithm["cases"]:
        assert case["dtype"] == "torch.int8" and case["shape"] == [40, 48], case
        assert -127 <= case["min"] and case["max"] <= 127, case
    shapes = algorithm["granularities"]
    assert shapes["tensor"]["distinct_per_row"] == [1, 1] and shapes["tensor"]["rows_equal_across_blocks"]
    assert shapes["channel"]["distinct_per_row"] == [1, 1] and not shapes["channel"]["rows_equal_in_first_block"]
    # One scale per 16 output channels: rows 0 and 15 share it, rows 15 and 16 do not.
    block = shapes["output_block"]
    assert block["rows_equal_in_first_block"] and not block["rows_equal_across_blocks"]
    # One scale per 16 reduced elements: a row of 48 has three.
    assert shapes["reduction_block"]["distinct_per_row"] == [3, 3]


def test_a_column_with_no_calibration_evidence_is_rounded_to_nearest(algorithm):
    assert algorithm["dead"] == {"dead_columns": 1, "column_is_nearest": True}


def test_gptq_is_deterministic_across_processes(algorithm):
    assert all(case["repeatable"] for case in algorithm["cases"])
    assert _run(_ALGORITHM, str(_TARGETGEN))["digest"] == algorithm["digest"]


# --- the capture (capture venv) --------------------------------------------------------------------

_CAPTURE = r"""
import importlib.util
import json
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import torch
from torch.nn import functional as F

import _gptq as G

spec = importlib.util.spec_from_file_location("recipe_quantizer", sys.argv[1] + "/_recipe_quantizer.py")
quantizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantizer)
recipe, plain = json.loads(sys.argv[3]), json.loads(sys.argv[4])
options = recipe["weight"]["gptq"]


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.first = torch.nn.Linear(8, 24)
        self.second = torch.nn.Linear(24, 6)
        self.table = torch.nn.Parameter(torch.randn(6, 5))

    def forward(self, x):
        h = torch.relu(self.first(self.conv(x).mean((2, 3))))
        return torch.matmul(self.second(h), self.table)


torch.manual_seed(3)
model = Toy().eval()
source = {name: value.clone() for name, value in model.state_dict().items()}
stream = [(torch.randn(4, 3, 6, 6),) for _ in range(6)]


def capture(body):
    program = quantizer.apply_recipe(model, body, example_inputs=stream[0], calibration_inputs=stream)
    codes = {}
    for node in program.graph.nodes:
        if node.op == "call_function" and node.target in (
            torch.ops.aten.conv2d.default, torch.ops.aten.linear.default, torch.ops.aten.matmul.default
        ):
            folded = node.args[1].args[0]
            codes[node.name] = getattr(program, folded.target)
    return program, codes


# The module's own GPTQ, from the same calibration inputs (the first N samples), computed on the
# eager model: im2col rows for the convolution, tokens for the linears, the matmul as W = table^T.
sums = {"conv2d": G.HessianSum(27), "linear": G.HessianSum(8), "linear_1": G.HessianSum(24),
        "matmul": G.HessianSum(6)}
with torch.no_grad():
    for (x,) in stream[: options["calibration_samples"]]:
        sums["conv2d"].add(G.contraction_rows(("conv2d", (3, 3), (1, 1), (1, 1), (1, 1)), x))
        pooled = model.conv(x).mean((2, 3))
        sums["linear"].add(pooled)
        hidden = torch.relu(model.first(pooled))
        sums["linear_1"].add(hidden)
        sums["matmul"].add(model.second(hidden))
matrices = {"conv2d": model.conv.weight.detach().reshape(8, -1), "linear": model.first.weight.detach(),
            "linear_1": model.second.weight.detach(), "matmul": model.table.detach().T}
direct = {}
for name, matrix in matrices.items():
    # The observer is per OUTPUT channel: a row of every W, including W = table^T of the matmul
    # (its stored [K, N] operand is observed along N).
    scales = G.symmetric_scales(matrix, granularity="channel", quant_min=-127, quant_max=127)
    codes, _ = G.gptq_codes(matrix, sums[name].value(), scales, quant_min=-127, quant_max=127,
                            damp=options["damp"], act_order=options["act_order"], block=options["block"])
    if name == "conv2d":
        codes = codes.reshape(model.conv.weight.shape)
    direct[name] = codes.T if name == "matmul" else codes

program, captured = capture(recipe)
_, again = capture(recipe)
plain_program, nearest = capture(plain)
stats = program._recipe_quantization_stats
with torch.no_grad():
    reference = model(*stream[0])
    cosine = float(F.cosine_similarity(program(*stream[0]).flatten(), reference.flatten(), dim=0))
print(json.dumps({
    "equal_to_direct": {name: bool(torch.equal(captured[name], direct[name])) for name in direct},
    "dtypes": sorted({str(codes.dtype) for codes in captured.values()}),
    "differs_from_nearest": {name: bool(not torch.equal(captured[name], nearest[name])) for name in direct},
    "deterministic": all(torch.equal(captured[name], again[name]) for name in captured),
    "source_unchanged": all(torch.equal(source[name], value) for name, value in model.state_dict().items()),
    "report": stats["weight_rounding"],
    "plain_has_report": "weight_rounding" in plain_program._recipe_quantization_stats,
    "cosine": cosine,
}))
"""


@pytest.fixture(scope="module")
def captured() -> dict:
    root = os.environ.get("MERLIN_M2M_DIR")
    if not root:
        pytest.skip("the selected model2MLIR source is not configured")
    plain = {**_recipe(), "families": [QR.CONTRACTION]}
    plain["recipe_sha256"] = QR.digest(plain)
    recipe = QR.with_gptq_weights(plain, calibration_samples=4, block=8)
    return _run(_CAPTURE, str(_TARGETGEN), root, json.dumps(recipe), json.dumps(plain))


def test_the_captured_program_carries_the_modules_own_gptq_codes(captured):
    assert captured["equal_to_direct"] == {"conv2d": True, "linear": True, "linear_1": True, "matmul": True}
    assert captured["dtypes"] == ["torch.int8"]
    assert any(captured["differs_from_nearest"].values()), "the option changed no code"
    assert captured["cosine"] > 0.99, captured["cosine"]


def test_the_capture_reports_every_weight_and_leaves_the_source_alone(captured):
    report = captured["report"]
    assert report["rounding"] == G.GPTQ and report["parameters"]["calibration_samples"] == 4
    assert report["calibration_samples_used"] == 4, "only the first N samples feed the Hessians"
    assert [row["contractions"] for row in report["weights"]] == [["conv2d"], ["linear"], ["linear_1"], ["matmul"]]
    assert {row["status"] for row in report["weights"]} == {G.GPTQ}
    for row in report["weights"]:
        assert row["output_error"][G.GPTQ] <= row["output_error"][G.NEAREST], row
    assert captured["source_unchanged"], "the source model's parameters were written"
    assert not captured["plain_has_report"], "a recipe without the option takes the unchanged path"


def test_a_recapture_reproduces_the_codes(captured):
    assert captured["deterministic"]
