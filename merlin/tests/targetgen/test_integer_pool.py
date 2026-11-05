"""Integer spatial means select one bounded sum/readout, never hidden f32 pools."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import _integer_pool as P


def test_integer_pool_work_limits_are_closed():
    good = dict(schema=P.SCHEMA, max_window_elements=32, max_weight_elements=32, max_macs=1000)
    assert P.check(good) == good
    for field in good:
        with pytest.raises(ValueError):
            P.check(good | {field: False})
    with pytest.raises(ValueError):
        P.check(good | {"padding": 0})


SCRIPT = r"""
import json,sys
sys.path.insert(0,sys.argv[1])
import torch,torchao
import _integer_pool as P
torch.set_num_threads(1);torch.set_num_interop_threads(1)
A,Q=torch.ops.aten,torch.ops.quantized_decomposed
out=[]
for mode in ('plain','keep','negative_axes','axes','bad_axis','duplicate','zero','budget','fanout','dtype'):
    g=torch.fx.Graph();root=torch.nn.Module()
    x=g.placeholder('x');x.meta['val']=torch.empty((2,3,2,5),dtype=torch.int8)
    d=g.call_function(Q.dequantize_per_tensor.default,(x,.25,0,-128,127,torch.int8))
    axes=[-2,-1] if mode=='negative_axes' else [2,3]
    if mode=='axes':axes=[1,2,3]
    if mode=='bad_axis':axes=[2,3,9]
    if mode=='duplicate':axes=[2,3,3]
    keep=mode=='keep'
    m=g.call_function(A.mean.dim,(d,axes,keep));m.meta['val']=torch.empty((2,3,1,1) if keep else (2,3))
    q=g.call_function(Q.quantize_per_tensor.default,(m,.5,0,-128,127,torch.int8))
    y=g.call_function(Q.dequantize_per_tensor.default,(q,.5,0,-128,127,torch.int8));g.output(y)
    gm=torch.fx.GraphModule(root,g)
    if mode=='zero':d.args=(x,.25,1,-128,127,torch.int8)
    if mode=='fanout':
        with g.inserting_before(q):g.call_function(A.neg.default,(m,))
    if mode=='dtype':m.kwargs={'dtype':torch.float64}
    limits=dict(schema=P.SCHEMA,max_window_elements=9 if mode=='budget' else 10,max_weight_elements=10,max_macs=60)
    try: report=P.rewrite(gm,limits,epilogue={'requant':{'clamp':[-128,127]}})
    except ValueError:
        assert mode not in ('plain','keep','negative_axes');out.append(mode);continue
    assert mode in ('plain','keep','negative_axes')
    raw=torch.arange(-30,30,dtype=torch.int8).reshape(2,3,2,5)
    expected=torch.round(raw.long().sum((2,3),keepdim=keep).float()*.05).clamp(-128,127).to(torch.int8).float()*.5
    assert torch.equal(gm(raw),expected)
    assert report['sites'][0]['macs']==60 and report['sites'][0]['weight_bytes']==10
    assert not any(n.target==A.mean.dim for n in g.nodes)
    assert sum(n.target==A.linear.default for n in g.nodes)==1
    out.append(mode)
print(json.dumps(out))
"""


def test_actual_integer_mean_graph_and_closed_negative_cases():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("selected capture interpreter required")
    result = subprocess.run(
        [python, "-B", "-c", SCRIPT, str(repo_root() / "src/merlin/targetgen")],
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(json.loads(result.stdout.strip().splitlines()[-1])) == 10
