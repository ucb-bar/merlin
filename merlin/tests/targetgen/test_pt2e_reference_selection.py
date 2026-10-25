"""Per-graph golden-reference selection and the exact oracle's casts and in-place writes."""

import os
import subprocess

import pytest

from merlin.common.paths import repo_root


def run_torch(script):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required")
    env = {**os.environ, "PYTHONPATH": str(repo_root() / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=180, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


def test_structure_selects_the_exact_oracle_only_for_closed_rational_graphs():
    run_torch(r"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.pt2e_reference_selection import classify
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
def dq(x): return DQ(x,.03,0,-128,127,torch.int8)
def q(x): return Q(x,.01,0,-128,127,torch.int8)
cases = {
 'closed': (lambda a,b: q(torch.relu(dq(a)@dq(b).t())), True),
 'in_place_residual': (lambda a,b: q(dq(a).add_(dq(b))), True),
 'widening_cast': (lambda a,b: (dq(a)@dq(b)).double(), True),
 'framework_operand': (lambda a,b: q(dq(a)+torch.nn.functional.layer_norm(b.float(),(4,))), True),
 'chained_readouts': (lambda a,b: dq(a)@dq(b)+dq(b)@dq(a), False),
 'transcendental': (lambda a,b: (dq(a)@dq(b)).sin(), False),
 'float_operand': (lambda a,b: dq(a)@b.float(), False),
 'bf16_island': (lambda a,b: (dq(a).bfloat16()@dq(b).bfloat16()).float(), False),
 'narrowing_after_readout': (lambda a,b: (dq(a)@dq(b)).half(), False),
}
a=torch.randint(-128,127,(4,4),dtype=torch.int8)
for name,(body,closed) in cases.items():
 module=type(name,(torch.nn.Module,),{'forward':lambda self,a,b,body=body: body(a,b)})()
 result=classify(torch.export.export(module,(a,a)).module())
 assert result['closed_rational'] is closed, (name,result)
 assert (result['reason'] is None) is closed, (name,result)
""")


def test_exact_casts_round_once_onto_the_target_grid_and_in_place_writes_reach_views():
    run_torch(r"""
import numpy as np, torch
from fractions import Fraction as F
from merlin.targetgen.exact_arithmetic import ExactTensor, materialize
from merlin.targetgen.exact_quantized_reference import evaluate_operation, evaluate_in_place
g=torch.Generator().manual_seed(1)
x=torch.cat([torch.randn(2000,generator=g)*s for s in (1e-42,1e-39,1e-6,1.0,1e4,1e30)]).float()
bits=torch.randint(0,2**31-1,(2000,),generator=g,dtype=torch.int64)
x=torch.cat([x,((bits & ~0xFFFF) | 0x8000).to(torch.int32).view(torch.float32)])
for target in (torch.bfloat16,torch.float16):
 y=x[torch.isfinite(x.to(target).float())&torch.isfinite(x)]
 e=ExactTensor(np.array([F.from_float(v) for v in y.tolist()],dtype=object),torch.float32)
 narrow=evaluate_operation('aten.to.dtype',(e,target),{})
 assert torch.equal(materialize(evaluate_operation('aten._to_copy.default',(narrow,),{'dtype':torch.float32})),
                    y.to(target).float())
try: evaluate_operation('aten.to.dtype',(ExactTensor(np.array([F(2)**200]),torch.float32),torch.bfloat16),{})
except ValueError: pass
else: raise AssertionError('finite reference produced an overflow infinity')
try: evaluate_operation('aten.to.dtype',(e,torch.int32),{})
except ValueError: pass
else: raise AssertionError('integer cast accepted')
third=ExactTensor(np.array([F(1,3)],dtype=object),torch.float32)
assert evaluate_operation('aten.to.dtype',(third,torch.float64),{}).values[0]==F(1,3)
a=ExactTensor(np.array([F(1,3),F(2)],dtype=object),torch.float32,codes=np.array([1,2]),scale=.5)
view=evaluate_operation('aten.view.default',(a,[2,1]),{})
same=evaluate_operation('aten.to.dtype',(a,torch.float32),{})
assert evaluate_in_place('aten.add_.Tensor',(a,ExactTensor(np.array([F(1,6),F(-1)],dtype=object),torch.float32)),{}) is a
assert a.values.tolist()==[F(1,2),F(1)] and a.codes is None
assert view.values.reshape(-1).tolist()==same.values.tolist()==[F(1,2),F(1)]
for name in ('aten.t_.default','aten.squeeze_.default'):
 try: evaluate_in_place(name,(a,),{})
 except ValueError: pass
 else: raise AssertionError(name+' accepted')
""")
