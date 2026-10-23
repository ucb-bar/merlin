"""Exact code and tie checks over full int8/uint8 domains and real FX graphs."""

import os
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.exact_quantized_reference import dequantize, quantize, round_binary


@pytest.mark.parametrize(
    "low,high,zp", [(-128, 127, 0), (-127, 127, 0), (-128, 127, -13), (0, 255, 123), (1, 254, 129)]
)
@pytest.mark.parametrize("scales", [(0.03, 0.01), (0.5, 0.25), (0.0137, 0.0193)])
def test_every_code_and_pair_against_independent_fraction_arithmetic(low, high, zp, scales):
    codes = np.arange(low, high + 1)
    exact = dequantize(codes, scales[0], zp, low, high)
    a, b = np.meshgrid(exact.values, exact.values)
    actual = quantize(a + b, scales[1], zp, low, high)
    ratio = Fraction.from_float(scales[0]) / Fraction.from_float(scales[1])
    expected = np.array(
        [[min(high, max(low, round((int(x) + int(y) - 2 * zp) * ratio) + zp)) for x in codes] for y in codes]
    )
    assert np.array_equal(actual, expected)


def test_channel_parameters_and_half_even():
    codes = np.array([[1, 3, 5], [-1, -3, -5]])
    values = dequantize(codes, [0.5, 0.25, 1.0], [0, 1, -1], axis=1).values
    actual = quantize(values, [1.0, 0.5, 2.0], [0, 2, 3], axis=1)
    expected = [
        [round(Fraction(int(codes[i, j]) - [0, 1, -1][j], 2)) + [0, 2, 3][j] for j in range(3)] for i in range(2)
    ]
    assert actual.tolist() == expected
    # A binary64 intermediate would erase this side of a binary32 rounding tie.
    tie = Fraction(1) + Fraction(1, 2**24)
    assert round_binary(tie + Fraction(1, 2**100)) == 1 + 2**-23
    assert round_binary(tie) == 1.0


def run_torch(script):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required")
    env = {**os.environ, "PYTHONPATH": str(repo_root() / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


def test_deterministic_masks_singleton_products_and_zero_exponential_branch_are_rational():
    run_torch(r'''
import numpy as np, torch
from fractions import Fraction as F
from merlin.targetgen.exact_arithmetic import ExactTensor
from merlin.targetgen.exact_quantized_reference import evaluate_operation
x=ExactTensor(np.array([F(-3,4),F(1,3)],dtype=object),torch.float32)
for p in (0,1):
 y,mask=evaluate_operation(torch.ops.aten.native_dropout.default,(x,p,True),{})
 assert y.values.tolist()==(x.values.tolist() if p==0 else [F(0),F(0)])
 assert mask.values.tolist()==[p==0,p==0]
try:
 evaluate_operation(torch.ops.aten.native_dropout.default,(x,.5,True),{})
except ValueError: pass
else: raise AssertionError('stochastic reference accepted')
singleton=ExactTensor(x.values.reshape(2,1),torch.float32)
out=evaluate_operation(torch.ops.aten.prod.dim_int,(singleton,1,False),{})
assert out.values.tolist()==x.values.tolist()
out=evaluate_operation(torch.ops.aten.elu.default,(x,0,2,1),{})
assert out.values.tolist()==[F(0),F(2,3)]
''')


def test_real_fx_graphs_use_exact_oracle_for_both_readout_selections():
    run_torch(r"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.core_aten_w8a8_reference import selected_pt2e_reference
from merlin.targetgen.exact_quantized_reference import quantize, rational
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def forward(self,x):
  y=DQ(x,0.03,0,-128,127,torch.int8)
  return Q(torch.mean(y,dim=1),0.01,0,-128,127,torch.int8)
x=torch.tensor([[-62,87],[37,50]],dtype=torch.int8)
model=torch.export.export(Model(),(x,)).module()
for numerics in ['float_readout','integer_readout']:
 result, identity=selected_pt2e_reference(model,(x,),numerics=numerics)
 output=torch.utils._pytree.tree_leaves(result.output)[0]
 expected=quantize([rational(.03)*25/2,rational(.03)*87/2],.01)
 assert output.tolist()==expected.tolist()
 assert identity['arithmetic']=='exact_rational'
 assert result.eager_float_divergence['informational'] is True
""")


@pytest.mark.parametrize("dtype,low,high,zp", [("int8", -128, 127, -7), ("int8", -127, 127, 0), ("uint8", 0, 255, 127)])
def test_fx_matmul_scales_zero_points_and_storage(dtype, low, high, zp):
    run_torch(f"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.core_aten_w8a8_reference import run_pt2e_reference
from merlin.targetgen.exact_quantized_reference import dequantize,quantize
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
dtype=torch.{dtype}
class Model(torch.nn.Module):
 def forward(self,a,b):
  return Q(DQ(a,.03,{zp},{low},{high},dtype) @ DQ(b,.02,{zp},{low},{high},dtype),.01,{zp},{low},{high},dtype)
a=torch.arange({low},{high},dtype=torch.int64)
a=a[:a.numel()//2*2].to(dtype).reshape(-1,2)
b=torch.tensor([[{low},{high}],[{high},{low}]],dtype=dtype)
model=torch.export.export(Model(),(a,b)).module()
result=run_pt2e_reference(model,(a,b))
output=torch.utils._pytree.tree_leaves(result.output)[0]
left=dequantize(a,.03,{zp},{low},{high}).values
right=dequantize(b,.02,{zp},{low},{high}).values
expected=quantize(left @ right,.01,{zp},{low},{high})
assert output.tolist()==expected.tolist()
assert result.contraction_count==1
""")


def test_recipe_parameters_cannot_erase_an_asymmetric_or_channel_scheme():
    run_torch(r"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen._recipe_quantizer import _per_tensor_scale
from types import SimpleNamespace
for dtype,zp in [(torch.int8,3),(torch.uint8,128)]:
 node=SimpleNamespace(op='call_function',target=torch.ops.quantized_decomposed.quantize_per_tensor.default,
                      args=(None,.03,zp,-128 if dtype==torch.int8 else 0,127 if dtype==torch.int8 else 255,dtype))
 assert _per_tensor_scale(node,'quantized_decomposed.quantize_per_tensor') is None
""")


def test_float_boundary_preserves_negative_underflow_zero():
    assert np.signbit(round_binary(-Fraction(1, 2**200)))
    assert round_binary(Fraction(1, 2**149)) == 2**-149


def test_fx_channel_quantization_preserves_axis_and_zero_points():
    run_torch(r"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.core_aten_w8a8_reference import run_pt2e_reference
from merlin.targetgen.exact_arithmetic import dequantize,quantize
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
DQC=torch.ops.quantized_decomposed.dequantize_per_channel.default
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('scales',torch.tensor([.25,.5]))
  self.register_buffer('zeros',torch.tensor([-3,7]))
 def forward(self,a,b):
  return Q(DQ(a,.125,5,-128,127,torch.int8) @ DQC(b,self.scales,self.zeros,1,-128,127,torch.int8),
           .25,-13,-128,127,torch.int8)
a=torch.arange(-24,24,dtype=torch.int8).reshape(16,3)
b=torch.tensor([[4,-2],[-9,10],[12,-7]],dtype=torch.int8)
model=torch.export.export(Model(),(a,b)).module()
result=run_pt2e_reference(model,(a,b))
left=dequantize(a,.125,5).values
right=dequantize(b,[.25,.5],[-3,7],axis=1).values
expected=quantize(left @ right,.25,-13)
assert torch.utils._pytree.tree_leaves(result.output)[0].tolist()==expected.tolist()
""")


def test_exact_capture_gate_preserves_int64_and_zero_bits():
    run_torch(r"""
import torch
from merlin.targetgen._capture_integer_agreement import integerized_agreement
from merlin.targetgen._m2m_capture_worker import _output_abi
for left,right in [(torch.tensor([2**53],dtype=torch.int64),torch.tensor([2**53+1],dtype=torch.int64)),
                   (torch.tensor([0.]),torch.tensor([-0.]))]:
 result=integerized_agreement(left,right,atol=0,rtol=0,output_abi=_output_abi)
 assert result['status']=='failed'
""")


def test_reference_serialization_keeps_every_dtype_as_exact_bytes():
    run_torch(r"""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from merlin.targetgen.core_aten_w8a8_reference import write_pt2e_reference
leaves=[torch.tensor([200],dtype=torch.uint8),torch.tensor([-0.],dtype=torch.float16),
        torch.tensor([2**53+1],dtype=torch.int64),torch.tensor([1+2**-52],dtype=torch.float64)]
reference=SimpleNamespace(sites=[],post_inputs=[leaves[2]],eager_float_divergence={'informational':True})
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'reference.json'
 write_pt2e_reference(path,reference,leaves,{'arithmetic':'exact_rational'})
 document=json.loads(path.read_text())
 assert document['output_bytes']==[x.numpy().tobytes().hex() for x in leaves]
 assert document['post_input_bytes']==[leaves[2].numpy().tobytes().hex()]
""")


def test_rational_layout_defaults_and_reduction_precision():
    run_torch(r"""
import torch
from merlin.targetgen.exact_arithmetic import dequantize,materialize
from merlin.targetgen.exact_quantized_reference import evaluate_operation
x=torch.arange(-12,12,dtype=torch.int8).reshape(2,3,4)
value=dequantize(x,.25,dtype=torch.float32)
flat=evaluate_operation(torch.ops.aten.flatten.using_ints,(value,1,2),{})
assert materialize(flat).shape==(2,12)
clamped=evaluate_operation(torch.ops.aten.hardtanh.default,(value,),{})
assert torch.equal(materialize(clamped),torch.clamp(x.float()*.25,-1,1))
reduced=evaluate_operation(torch.ops.aten.mean.dim,(value,[2],False),{'dtype':torch.float64})
assert materialize(reduced).dtype==torch.float64
""")


def test_required_ieee_output_overflow_has_exact_thresholds():
    run_torch(r"""
import math
from fractions import Fraction as F
from merlin.targetgen.exact_arithmetic import round_binary
for bits,p,emax in [(16,11,15),(32,24,127)]:
 threshold=F(2)**(emax+1)-F(2)**(emax-p)
 maximum=F(2)**(emax+1)-F(2)**(emax-p+1)
 assert round_binary(threshold-F(1,2**100),bits,overflow="infinity")==float(maximum)
 assert round_binary(threshold,bits,overflow="infinity")==math.inf
 assert round_binary(-threshold,bits,overflow="infinity")==-math.inf
""")
