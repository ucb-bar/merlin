"""Explicit normalized-probability INT8 context output, independent of model names."""

import json
import os
import subprocess

import pytest
from test_recipe_static_attention import _QUANTIZER, _recipe

from merlin.targetgen import quant_recipe as Q


def recipe():
    return Q.with_dynamic_contractions(
        _recipe(), scales=Q.STATIC_HEADS, static_families=[], context_output="int8_value_grid"
    )


def test_context_policy_is_explicit_and_single_plane():
    selected = recipe()
    assert selected[Q.DYNAMIC]["context_output"] == "int8_value_grid"
    assert selected["recipe_sha256"] == Q.digest(selected)
    with pytest.raises(ValueError, match="single-plane"):
        Q.with_dynamic_contractions(
            _recipe(), scales=Q.STATIC_HEADS, context_output="int8_value_grid", planes={Q.SOFTMAX_OUTPUT: 2}
        )


_SCRIPT = r"""
import importlib.util, json, sys
from pathlib import Path
source=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(source.parent))
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('quant',source)
quant=importlib.util.module_from_spec(spec);spec.loader.exec_module(quant)
class Attention(torch.nn.Module):
 def forward(self,q,k,v):
  return torch.softmax(q@k.transpose(-2,-1),dim=-1)@v
torch.manual_seed(42)
sample=tuple(torch.randn(1,2,int(sys.argv[3]),32) for _ in range(3))
model=Attention().eval()
graph=torch.export.export(model,sample).module()
q=quant.build_quantizer(json.loads(sys.argv[2]),layer_plan=quant._plan_layers(json.loads(sys.argv[2]),model))
rows=q.rewrite_dynamic(graph,[sample],100)
graph.graph.eliminate_dead_code();graph.recompile()
def scale(x):return (x.double().abs().amax((0,2,3))/127.5).clamp(min=2**-12).view(1,2,1,1)
def grid(x,s):return torch.round(x/s.float()).clamp(-127,127).to(torch.int32)
a,b,v=sample
sa,sb,sv=scale(a),scale(b),scale(v)
scores=(grid(a,sa).float()@grid(b,sb).float().transpose(-2,-1))*(sa*sb).float()
p=torch.softmax(scores,dim=-1)
pi=torch.round(p*127).clamp(0,127)
acc=pi@grid(v,sv).float()
want=torch.round(acc*(1/127)).clamp(-128,127)*sv.float()
got=graph(*sample)
torch.testing.assert_close(got,want,rtol=0,atol=0)
nodes=list(graph.graph.nodes)
mm=[n for n in nodes if n.target==torch.ops.aten._int_mm.default]
qt=[n for n in nodes if n.target==torch.ops.quantized_decomposed.quantize_per_tensor.default]
assert len(mm)==4 and len(qt)==2
assert all(n.meta['val'].dtype==torch.int8 and n.meta['val'].dim()==2 for n in qt)
assert len([r for r in rows if r.get('output_dtype')=='int8'])==1
print(json.dumps({'int_mm':len(mm),'native_context_readouts':len(qt),'exact':True}))
"""


@pytest.mark.parametrize("sequence", [16, 32])
def test_per_head_readout_before_join_matches_selected_integer_recipe(sequence):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("capture interpreter not configured")
    result = subprocess.run(
        [python, "-B", "-c", _SCRIPT, str(_QUANTIZER), json.dumps(recipe()), str(sequence)],
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.strip().splitlines()[-1])["exact"]
