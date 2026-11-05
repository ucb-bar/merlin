"""New output grids are explicit; residual and returned values retain wide exits."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import _integer_epilogue as E
from merlin.targetgen import quant_recipe as Q


def _recipe():
    return json.loads((repo_root() / "examples/gemmini/protocol/recipes/resnet50.json").read_text())


def test_explicit_output_policy_is_immutable_and_content_bound():
    original = _recipe()
    snapshot = copy.deepcopy(original)
    selected = Q.with_native_output_quantization(original)
    assert original == snapshot
    assert selected["recipe_sha256"] == Q.digest(selected) != Q.digest(original)
    assert Q.with_native_output_quantization(selected) == selected


@pytest.mark.parametrize("change", [
    {"schema": "foreign"}, {"reader_policy": "anything"}, {"operators": []},
    {"operators": ["linear", "linear"]}, {"operators": ["foreign"]},
    {"operators": [True]}, {"operators": [{}]}, {"allow_unproved": True},
])
def test_output_policy_rejects_unknown_or_ambiguous_selection(change):
    selected = Q.with_native_output_quantization(_recipe())
    selected["epilogue"]["output_quantization"].update(change)
    with pytest.raises(E.EpilogueError):
        E.check(selected["epilogue"])


_SCRIPT = r"""
import copy,json,sys
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import torch,torchao
import _recipe_quantizer as RQ
import _integer_epilogue as E
from merlin.targetgen.quant_recipe import with_native_output_quantization
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture import pt2e_integer_reference as IR
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(71)
recipe=json.loads(sys.argv[3])
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('a','b','c','out'):
            setattr(self,name,torch.nn.Linear(8,8,bias=False))
        self.left=torch.nn.Linear(8,12,bias=False)
        self.right=torch.nn.Linear(8,12,bias=False)
        self.down=torch.nn.Linear(12,8,bias=False)
        self.head=torch.nn.Linear(8,4,bias=False)
    def forward(self,x):
        a=self.a(x);b=self.b(x);c=self.c(x)
        y=x+self.out(a.sin()*b.cos()+c.sigmoid())
        z=y+self.down(torch.nn.functional.silu(self.left(y))*self.right(y))
        return self.head(z)
x=torch.randn(2,8)
model=Network().eval()
rows={}
for mode in ('legacy','native'):
    policy=recipe if mode=='legacy' else with_native_output_quantization(recipe)
    q=RQ.apply_recipe(copy.deepcopy(model),policy,example_inputs=(x,),calibration_inputs=[(x,),(x*.5,)])
    stats=q._recipe_quantization_stats
    reference=IR.run_pt2e_integer_reference(q,(x,))
    integer,receipt=integerize_pt2e(copy.deepcopy(q),(x,))
    rows[mode]={'exits':stats['integer_epilogue']['by_exit'],
                'integer_exact':bool(torch.equal(reference.output,integer(x))),
                'output_shape':list(integer(x).shape),'annotated':stats['annotated_contractions']}
# Exercise graph-output/view/fanout/effect refusals directly with real typed ATen schemas.
block=with_native_output_quantization(recipe)['epilogue']
e=E.Epilogue(block,activation_spec=object())
for kind in ('compute','add','output','output_view','mutable','unknown','fanout'):
    g=torch.fx.Graph();a=g.placeholder('a');w=g.placeholder('w')
    n=g.call_function(torch.ops.aten.linear.default,(a,w))
    if kind=='output':g.output(n)
    elif kind=='output_view':
        v=g.call_function(torch.ops.aten.transpose.int,(n,0,1));g.output(v)
    elif kind=='add':g.output(g.call_function(torch.ops.aten.add.Tensor,(n,a)))
    elif kind=='mutable':g.output(g.call_function(torch.ops.aten.copy_.default,(n,a)))
    elif kind=='unknown':g.output(g.call_function(torch.sin,(n,)))
    else:
        g.output(g.call_function(torch.ops.aten.sin.default,(n,)))
        if kind=='fanout':g.output(n)
    rows[kind]=e.exit_of(n,n,False,lambda *_:False)
print(json.dumps(rows))
"""


def test_real_pt2e_native_exits_and_independent_integer_reference():
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m:
        pytest.skip("selected TorchAO and model2MLIR interpreters are required")
    root = repo_root()
    result = subprocess.run(
        [python, "-B", "-c", _SCRIPT, str(root / "src/merlin/targetgen"), str(Path(m2m) / "src"),
         json.dumps(_recipe())],
        env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr[-7000:]
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    assert rows["legacy"]["exits"] == {"requant": 0, "accumulator": 8}
    assert rows["native"]["exits"] == {"requant": 5, "accumulator": 3}
    for mode in ("legacy", "native"):
        assert rows[mode]["integer_exact"] and rows[mode]["annotated"] == 8
        assert rows[mode]["output_shape"] == [2, 4]
    assert rows["compute"] == E.REQUANT
    assert all(rows[kind] == E.ACCUMULATOR for kind in ("add", "output", "output_view", "mutable", "unknown", "fanout"))
