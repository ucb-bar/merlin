"""A float dtype conversion and a non-float32 fill are int8 work in a quantized graph, read out at their width.

WHY THIS EXISTS. PT2E quantizes float32 edges only, so ``x.to(torch.float64)`` and
``torch.full_like(x, v, dtype=torch.float64)`` stayed plain host operations under the int8 numeric flow:
the quantizer skipped any non-float32 result. On the int8 grid a value is a code and a scale, and its
floating width is chosen only when it is dequantized: the conversion is a copy of codes and the fill
is a constant fill of codes. ``_recipe_quantizer.read_out_floating_widths`` spells each at float32 (a
``clone``, a ``dtype=float32`` fill) and restores the requested width on the final dequantization, so the work
is quantized like any movement and the width becomes a property of the readout. The prospective
core-aten screen reads such a site as the int8 work the recipe makes of it, so the lane recaptures it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen.core_aten_spec_int8 import non_contraction_sites
from merlin.targetgen.host_capabilities import validate_host_capabilities

_BASE = repo_root() / "examples/gemmini/target"

_PROGRAM = """
module {
  func.func @forward(%x: tensor<2x3xf32>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
    %v = arith.constant 1.25 : f64
    %e = tensor.empty() : tensor<2x3xf64>
    %wide = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                            iterator_types = ["parallel", "parallel"]}
            ins(%x : tensor<2x3xf32>) outs(%e : tensor<2x3xf64>) attrs = {prov.aten = "aten._to_copy.default"} {
    ^bb0(%a: f32, %o: f64):
      %c = arith.extf %a : f32 to f64
      linalg.yield %c : f64
    } -> tensor<2x3xf64>
    %fill = linalg.fill {prov.aten = "aten.full_like.default"} ins(%v : f64) outs(%e : tensor<2x3xf64>)
            -> tensor<2x3xf64>
    return %wide, %fill : tensor<2x3xf64>, tensor<2x3xf64>
  }
}
"""


@pytest.fixture(scope="module")
def selection():
    return {
        "spec": fixture_resolved(_BASE, "gemmini"),
        "host": validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
        "capability_contract": load_yaml(_BASE / "contracts/target_contract.yaml"),
    }


def _sites(selection, *, prospective):
    with IR_LOCK:
        module = mq.parse(_PROGRAM)
        return {
            site["source_operator"]: site
            for site in non_contraction_sites(
                module, selection, contract=selection["capability_contract"], prospective=prospective
            )
        }


@pytest.mark.target("gemmini")
def test_a_prospective_screen_reads_a_float64_result_as_int8_work(selection):
    sites = _sites(selection, prospective=True)
    for name in ("aten._to_copy.default", "aten.full_like.default"):
        site = sites[name]
        assert site["quantized_island"], name
        assert site["signature"]["ordered_result_dtypes"] == ["i8"], name
        assert "f64" not in json.dumps(site["signature"]), name
    assert sites["aten._to_copy.default"]["signature"]["ordered_operand_dtypes"] == ["i8", "i8"]


@pytest.mark.target("gemmini")
def test_a_captured_float64_site_is_still_screened_by_its_precision(selection):
    sites = _sites(selection, prospective=False)
    for name in ("aten._to_copy.default", "aten.full_like.default"):
        assert not sites[name]["quantized_island"], name
        assert sites[name]["status"] == "refused", (name, sites[name]["reasons"])


_SCRIPT = r"""
import json
import torch
from merlin.targetgen import _recipe_quantizer as quant
recipe = {
    "schema": "quant_recipe_v1", "status": "derived",
    "families": ["contraction", "affine_map", "movement"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True, "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}
class ToCopy(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.float64)
class CoreToCopy(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten._to_copy.default(x, dtype=torch.float64)
class FullLike(torch.nn.Module):
    def forward(self, x):
        return torch.full_like(x, 1.25, dtype=torch.float64)
class Narrow(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.float32)
torch.manual_seed(0)
x = torch.linspace(-0.75, 1.25, 6).reshape(2, 3)
report = {}
for name, model, sample in (("to_copy", ToCopy(), x), ("core_to_copy", CoreToCopy(), x),
                            ("full_like", FullLike(), x),
                            ("float64_input", Narrow(), x.to(torch.float64))):
    try:
        converted = quant.apply_recipe(model.eval(), recipe, example_inputs=(sample,),
                                       source_layer_inventory=quant.layer_inventory(model))
    except quant.RecipeError as exc:
        report[name] = {"refused": type(exc).__name__}
        continue
    stats = converted._recipe_quantization_stats
    out = converted(sample)
    row = {"widths": stats["floating_readout_widths"], "moves": stats["annotated_movements"],
           "dtype": str(out.dtype)}
    recipes = [r for r in stats["integer_recipes"] if "refused" not in r]
    row["recipes"] = sorted(r["operator"] for r in recipes)
    if name in ("to_copy", "core_to_copy"):
        scale = recipes[0]["output_scale"]
        codes = torch.clamp(torch.round(sample * (1.0 / scale)), -128, 127)
        expected = codes.to(torch.float64) * scale
        row["exact"] = bool(torch.equal(out, expected))
    if name == "full_like":
        scale = recipes[0]["output_scale"]
        code = recipes[0]["constant_code"]
        value = code["code"] if isinstance(code, dict) else code
        row["exact"] = bool(torch.equal(out, torch.full((2, 3), value * scale,
                                                       dtype=torch.float64)))
    report[name] = row
print(json.dumps(report))
"""


def test_conversions_and_fills_are_quantized_and_read_out_at_their_width():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    root = repo_root()
    paths = [root / "src", root / "src/merlin/targetgen", *sorted((root / "packages").glob("*/src"))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(path) for path in paths)}
    result = subprocess.run([python, "-c", _SCRIPT], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])

    for name, operator in (("to_copy", "aten.to.dtype"), ("core_to_copy", "aten._to_copy.default")):
        to_copy = report[name]
        (width,) = to_copy["widths"]
        assert (width["operator"], width["quantized_as"], width["readout_dtype"]) == (
            operator,
            "aten.clone.default",
            "float64",
        )
        assert to_copy["moves"] == 1 and to_copy["recipes"] == ["aten.clone.default"]
        assert to_copy["dtype"] == "torch.float64" and to_copy["exact"], name

    full_like = report["full_like"]
    assert [row["quantized_as"] for row in full_like["widths"]] == ["aten.full_like.default"]
    assert full_like["moves"] == 1 and full_like["recipes"] == ["aten.full_like.default"]
    assert full_like["dtype"] == "torch.float64" and full_like["exact"]

    # A float64 INPUT cannot be quantized: nothing moves, and the recipe has no work to do.
    assert report["float64_input"].get("widths", []) == []
