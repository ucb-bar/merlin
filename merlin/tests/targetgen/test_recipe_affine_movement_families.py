"""The recipe quantizes affine maps and movements where the unit declares them, in their exact forms.

WHY THIS EXISTS. A derived recipe quantized contractions, operand sums and window means only, so every
other operation of the int8 support set stayed in floating point on the host: scalar add/sub/mul/div,
neg, standalone relu, sub.Tensor, and every movement (clone, permute, cat, pads, scatters, fills).
``affine_map`` and ``movement`` are now recipe families, licensed by the unit's declared standalone
elementwise + contraction and movement capabilities. The quantizer annotates them (an affine map gets
its own output scale; a movement shares its input's, or keeps independent ones), and after
conversion states each in its exact integer form (``recipe_numerics``).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import compute_units as cu
from merlin.targetgen import quant_recipe as QR
from merlin.targetgen import readout_facet as RF

_SOURCE = merlin_dir() / "python" / "merlin" / "targetgen" / "_recipe_quantizer.py"


def _facet() -> RF.ReadoutFacet:
    store = {"width": 64, "fields": {"out_scale": {"offset": 32, "width": None, "slot_width": 32}}}
    layouts = {"name": "register_bundle_layouts", "unresolved": {}, "bundles": {"StoreConfig": store}}
    datapaths = [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}]
    memories = [{"name": "accumulator", "bytes": 4096}]
    facts = {"facts": {"datapaths": datapaths, "memories": memories, "interfaces": [layouts]}}
    abi = dict(schema=RF.SCALAR_ABI_SCHEMA, accumulator_dtype="i32", output_dtype="i8", scale_dtype="f32")
    abi.update(clamp_min=-128, clamp_max=127, provenance={"scope": "test"})
    readouts = [{"selector": "i8", "applies": ["acc_scale"]}]
    return RF.derive("t", facts=facts, unit={"name": "u", "dtypes": ["int8"]}, scalar_abi=abi, readouts=readouts)


def _contract(*rows) -> dict:
    unit = {"name": "u", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"]}
    return {"compute_units": [{**unit, "semantic_capabilities": list(rows)}]}


_CONTRACTION = {"family": "contraction", "dtypes": ["int8"]}
_ELEMENTWISE = {
    "family": "elementwise_map",
    "dtypes": ["int8"],
    "standalone_evidence": [{"fact": "accumulate_on_load", "roles": ["operand_load", "config", "readout", "commit"]}],
}
_MOVEMENT = {"family": "movement", "dtypes": ["int8"], "forms": ["copy"]}


def _recipe(contract: dict):
    (candidate,) = QR.derive_candidates(contract, [_facet()])
    return candidate.recipe


def test_declared_capabilities_license_the_families():
    recipe = _recipe(_contract(_CONTRACTION, _ELEMENTWISE, _MOVEMENT))
    assert QR.AFFINE_MAP in recipe.families and QR.MOVEMENT in recipe.families
    assert recipe.why[QR.AFFINE_MAP] and recipe.why[QR.MOVEMENT]


def test_an_undeclared_capability_licenses_nothing_and_says_so():
    recipe = _recipe(_contract(_CONTRACTION))
    assert QR.AFFINE_MAP not in recipe.families and QR.MOVEMENT not in recipe.families
    assert recipe.unquantized[QR.AFFINE_MAP] and recipe.unquantized[QR.MOVEMENT]
    fused_only = {**_ELEMENTWISE, "composed_with": ["contraction"]}
    fused_only.pop("standalone_evidence")
    assert QR.AFFINE_MAP not in _recipe(_contract(_CONTRACTION, fused_only)).families


def test_a_caller_without_capabilities_keeps_the_old_families():
    assert set(QR.derive(_facet()).families) <= {QR.CONTRACTION, QR.OPERAND_SUM, QR.WINDOW_MEAN}


def _operator_table() -> dict[str, list[str]]:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_recipe_operators")
    table = next(n.value for n in ast.walk(function) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict))
    return {
        key.value: [ast.unparse(element) for element in value.elts]
        for key, value in zip(table.keys, table.values, strict=True)
    }


def test_the_quantizer_names_the_support_set_operators():
    table = _operator_table()
    affine = " ".join(table["affine_map"])
    movement = " ".join(table["movement"])
    for op in ("add.Scalar", "sub.Scalar", "mul.Scalar", "div.Scalar", "neg.default", "sub.Tensor", "relu.default"):
        assert op in affine, op
    for op in ("clone", "permute", "cat", "constant_pad_nd", "slice_scatter", "select_scatter", "repeat"):
        assert op in movement, op
    for op in ("avg_pool1d", "avg_pool3d", "adaptive_avg_pool1d", "adaptive_avg_pool3d"):
        assert op in " ".join(table["window_mean"]), op


_RECIPE = r"""
import json
import torch
from merlin.targetgen import _recipe_quantizer as quant
recipe = {
    "schema": "quant_recipe_v1", "status": "derived",
    "families": ["contraction", "window_mean", "affine_map", "movement"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True, "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}
"""
_SCRIPT = _RECIPE + r"""
import numpy as np
from merlin.targetgen import recipe_numerics as RN
class Model(torch.nn.Module):
    def forward(self, x, y):
        d = torch.sub(x, y)
        s = torch.add(d, 0.75)
        n = torch.neg(s)
        c = torch.cat([n, x], dim=1)
        return torch.nn.functional.pad(c, (1, 1), value=0.0)
torch.manual_seed(0)
inputs = (torch.randn(64, 32), torch.randn(64, 32) * 0.6)
model = Model().eval()
converted = quant.apply_recipe(
    model, recipe, example_inputs=inputs, source_layer_inventory=quant.layer_inventory(model)
)
stats = converted._recipe_quantization_stats
rows = {row["operator"]: row for row in stats["integer_recipes"]}
sub = rows["aten.sub.Tensor"]
(s_a, s_b), s_out = sub["input_scales"], sub["output_scale"]
verified = []
for row in rows.values():
 if "integer_recipe" not in row:
  assert "refused" in row or row["family"] == "movement"
  continue
 form = row["integer_recipe"]
 from fractions import Fraction
 if form["form"] == "retained_affine":
  assert form["readout"] == "full_i32"
  assert form["requantization"] == "retained_source_exact_integer"
  if row["operator"] == "aten.sub.Tensor":
   factors=[Fraction(int(n),int(d)) for n,d in zip(form["coefficients_numerator"],form["coefficients_denominator"])]
   assert factors == [Fraction(s_a)/Fraction(s_out),-Fraction(s_b)/Fraction(s_out)]
   assert int(form["offset_numerator"]) == 0
  verified.append(row["operator"])
  continue
 candidate = RN.IntegerRecipe(weights=tuple(form["weights"]), constant=form["constant"],
                               readout_scale=Fraction(int(form["scale_numerator"]),int(form["scale_denominator"])),
                               activation=form["activation"])
 if row["operator"] == "aten.sub.Tensor":
  from merlin.targetgen.exact_quantized_reference import quantize, rational
  grid=np.arange(-128,128)
  a,b=np.meshgrid(grid,grid)
  expected=quantize((a*rational(s_a)-b*rational(s_b)),s_out)
  assert np.array_equal(candidate.evaluate(a,b),expected)
 verified.append(row["operator"])
print(json.dumps({
    "maps": stats["annotated_affine_maps"], "moves": stats["annotated_movements"],
    "operators": sorted(rows), "verified": verified,
    "refused": {row["operator"]:row["refused"] for row in rows.values() if "refused" in row},
}))
"""


def _capture_report(script):
    """Run ``script`` under the TorchAO capture interpreter and return its last JSON line."""
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("TorchAO capture interpreter is not configured")
    root = repo_root()
    paths = [root / "src", root / "src/merlin/targetgen", *sorted((root / "packages").glob("*/src"))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(path) for path in paths)}
    result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_capture_quantizer_states_each_map_in_its_integer_form():
    report = _capture_report(_SCRIPT)
    assert report["maps"] == 3 and report["moves"] == 2
    operators = ["aten.add.Tensor", "aten.cat.default", "aten.neg.default", "aten.pad.default", "aten.sub.Tensor"]
    assert report["operators"] == operators
    for operator in ("aten.sub.Tensor", "aten.add.Tensor", "aten.neg.default"):
        assert operator in report["verified"] or operator in report["refused"]
    for reason in report["refused"].values():
        assert "exact" in reason or "quantization scales" in reason


def test_capability_rows_reach_the_recipe_from_the_contract():
    """The working-tree contract's capabilities, parsed, are what derive_candidates hands derive."""
    from merlin.common.yaml import load_yaml

    contract = load_yaml(repo_root() / "examples/gemmini/target/contracts/target_contract.yaml")
    families = {cap.family for unit in cu.compute_units(contract) for cap in unit.semantic_capabilities}
    assert {"contraction", "movement"} <= families


def test_backward_and_constant_ops_are_named():
    table = _operator_table()
    assert "convolution_backward" in " ".join(table["contraction"])
    for op in ("avg_pool2d_backward", "_adaptive_avg_pool2d_backward", "col2im"):
        assert op in " ".join(table["window_mean"]), op
    assert "aten.full.default" in " ".join(table["movement"])
    assert "add.Tensor" in " ".join(table["affine_map"])


_STANDALONE = _RECIPE + r"""
class Sum(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)
class Mean(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=-1, keepdim=True)
class Full(torch.nn.Module):
    def forward(self):
        return torch.full((4, 8), 1.25)
torch.manual_seed(0)
report = {}
for name, model, inputs in (("sum", Sum(), (torch.randn(8, 16), torch.randn(8, 16))),
                            ("mean", Mean(), (torch.randn(8, 16),)),
                            ("full", Full(), ())):
    converted = quant.apply_recipe(model.eval(), recipe, example_inputs=inputs,
                                   source_layer_inventory=quant.layer_inventory(model))
    stats = converted._recipe_quantization_stats
    report[name] = {key: stats[key] for key in ("annotated_sums", "annotated_means", "annotated_affine_maps",
                                                "annotated_movements")}
    report[name]["recipes"] = sorted(row["operator"] for row in stats["integer_recipes"] if "refused" not in row)
    report[name]["refused"] = {row["operator"]:row["refused"] for row in stats["integer_recipes"] if "refused" in row}
print(json.dumps(report))
"""


def test_a_sum_a_mean_and_a_constant_read_themselves_out():
    """No quantized reader closes them, so where a standalone readout is licensed their output is quantized."""
    report = _capture_report(_STANDALONE)
    assert report["sum"]["annotated_affine_maps"] == 1
    assert report["mean"]["annotated_means"] == 1
    for name, operator in (("sum", "aten.add.Tensor"), ("mean", "aten.mean.dim")):
        assert operator in report[name]["recipes"] or operator in report[name]["refused"]
        if operator in report[name]["refused"]:
            assert "exact" in report[name]["refused"][operator]
    assert report["full"]["annotated_movements"] == 1 and report["full"]["recipes"] == ["aten.full.default"]


def test_shape_only_inputs_and_exact_retained_affine_recipe():
    from test_exact_quantized_reference import run_torch

    run_torch(r"""
import torch
from fractions import Fraction as F
from merlin.targetgen._recipe_integer import integer_recipes
import torch.ao.quantization.fx._decomposed
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
for target,args in [(torch.ops.aten.fill.Scalar,(1.25,)),(torch.ops.aten.full_like.default,(1.25,)),
                    (torch.ops.aten.zeros_like.default,()),(torch.ops.aten.ones_like.default,())]:
 graph=torch.fx.Graph()
 x=graph.placeholder('shape')
 y=graph.call_function(target,(x,*args)); y.meta['merlin_recipe_family']='movement'
 q=graph.call_function(Q,(y,.125,0,-128,127,torch.int8));graph.output(q)
 row,=integer_recipes(torch.fx.GraphModule({},graph))
 assert row['input_scales']==[] and 'refused' not in row
 assert row['constant_code']==(0 if 'zeros' in str(target) else 8 if 'ones' in str(target) else 10)
# An adjoint's second tensor is geometry, not a quantized data operand.
for target,args in [(torch.ops.aten.avg_pool2d_backward.default,([2,2],[1,1],[0,0],False,True,3)),
                    (torch.ops.aten._adaptive_avg_pool2d_backward.default,())]:
 graph=torch.fx.Graph(); grad=graph.placeholder('grad'); shape=graph.placeholder('shape')
 d=graph.call_function(DQ,(grad,.125,0,-128,127,torch.int8))
 y=graph.call_function(target,(d,shape,*args)); y.meta['merlin_recipe_family']='window_mean'
 q=graph.call_function(Q,(y,.25,0,-128,127,torch.int8));graph.output(q)
 row,=integer_recipes(torch.fx.GraphModule({},graph))
 assert row['input_scales']==[.125] and 'refused' not in row
 assert row['divisor']=='retained_source_geometry'
 assert row['requantization']=='retained_source_exact_integer'
# A nonintegral seed must remain rational on host, without rounding into i32.
for target in [torch.ops.aten.add.Scalar,torch.ops.aten.sub.Scalar]:
 for alpha in [0,2,-3,.5]:
  graph=torch.fx.Graph(); x=graph.placeholder('x')
  scale=float(torch.tensor(.03))
  d=graph.call_function(DQ,(x,scale,0,-128,127,torch.int8))
  y=graph.call_function(target,(d,.5,alpha)); y.meta['merlin_recipe_family']='affine_map'
  q=graph.call_function(Q,(y,.125,0,-128,127,torch.int8));graph.output(q)
  row,=integer_recipes(torch.fx.GraphModule({},graph))
  assert 'refused' not in row
  if alpha:
   form=row['integer_recipe']; sign=-1 if 'sub' in str(target) else 1
   assert form['form']=='retained_affine'
   assert F(int(form['coefficients_numerator'][0]),int(form['coefficients_denominator'][0]))==F(scale)/F(.125)
   assert F(int(form['offset_numerator']),int(form['offset_denominator']))==sign*F(alpha)*F(.5)/F(.125)
""")


def test_closed_additive_reductions_retain_exact_divisors_and_readout():
    from test_exact_quantized_reference import run_torch

    run_torch(r"""
import torch
import torch.ao.quantization.fx._decomposed
from fractions import Fraction
from merlin.targetgen._recipe_integer import integer_recipes
from merlin.targetgen.core_aten_w8a8_reference import selected_pt2e_reference
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
x=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.int8)
for target,args,divisor,want in [
 (torch.ops.aten.mean.default,(),6,[2]),
 (torch.ops.aten.mean.dim,([-1],True),3,[[1],[2]]),
 (torch.ops.aten.sum.default,(),1,[10]),
 (torch.ops.aten.sum.dim_IntList,([-1],True),1,[[3],[8]])]:
 graph=torch.fx.Graph(); raw=graph.placeholder('x');raw.meta['val']=x
 d=graph.call_function(DQ,(raw,.125,0,-128,127,torch.int8))
 y=graph.call_function(target,(d,*args));y.meta.update(val=target(x.float(),*args),merlin_recipe_family='window_mean')
 q=graph.call_function(Q,(y,.25,0,-128,127,torch.int8));graph.output(q)
 model=torch.fx.GraphModule({},graph)
 row,=integer_recipes(model)
 assert row['divisor']==divisor and row['requantization']=='exact_rational_host',row
 assert Fraction(int(row['scale_numerator']),int(row['scale_denominator']))==Fraction(1,2*divisor)
 result,_=selected_pt2e_reference(model,(x,),numerics='integer_readout')
 actual=torch.utils._pytree.tree_leaves(result.output)[0]
 assert actual.reshape(-1).tolist()==torch.tensor(want).reshape(-1).tolist(),(target,actual)
# Explicit scalar axes must preserve the rank-zero one-element divisor.
from merlin.targetgen._recipe_integer import _window_divisor
from types import SimpleNamespace as N
raw=N(meta={'val':torch.tensor(3.)})
for dims in ([0],[-1],[]):
 assert _window_divisor(N(args=(raw,dims),target=torch.ops.aten.mean.dim,meta={'val':torch.tensor(3.)}))==1
""")
