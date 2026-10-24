"""Rational affine, movement and adjoint checks; every comparison is exact."""

import pytest
from test_exact_quantized_reference import run_torch


@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize("overload", ["Scalar", "Tensor"])
def test_alpha_across_quantization_schemes(op, overload):
    run_torch(f"""
import torch
import numpy as np
from fractions import Fraction
from merlin.targetgen.exact_quantized_reference import evaluate_operation
from merlin.targetgen.exact_arithmetic import dequantize,quantize
for low,high,zp in [(-128,127,0),(-127,127,-3),(0,255,131)]:
 x=dequantize(np.arange(low,high+1),float(torch.tensor(.03)),zp,low,high,dtype=torch.float32)
 for alpha in [0,1,2,-3,.5,-.125]:
  scalar=Fraction(1,2)
  y=scalar if '{overload}'=='Scalar' else dequantize(np.arange(high,low-1,-1),.125,zp,low,high,dtype=torch.float32)
  args=(x,float(scalar),alpha) if '{overload}'=='Scalar' else (x,y)
  kwargs={{}} if '{overload}'=='Scalar' else {{'alpha':alpha}}
  actual=evaluate_operation(getattr(torch.ops.aten.{op},'{overload}'),args,kwargs)
  if '{overload}'=='Scalar':
   keyword=evaluate_operation(getattr(torch.ops.aten.{op},'{overload}'),args[:2],{{'alpha':alpha}})
   assert np.array_equal(actual.values,keyword.values)
  right=scalar if '{overload}'=='Scalar' else y.values
  expected=x.values + {1 if op == "add" else -1} * Fraction(alpha)*right
  assert np.array_equal(actual.values,expected)
  assert np.array_equal(quantize(actual,.0625,zp,low,high),quantize(expected,.0625,zp,low,high))
""")


def test_movement_preserves_codes_and_rationals():
    run_torch(r"""
import torch
import numpy as np
from merlin.targetgen.exact_quantized_reference import evaluate_operation
from merlin.targetgen.exact_arithmetic import dequantize
for dims in [1,2,3]:
 codes=torch.arange(2*3**dims,dtype=torch.int64).reshape(2,*([3]*dims))
 for prefix in ['reflection','replication']:
  target=getattr(torch.ops.aten,f'{prefix}_pad{dims}d').default
  padding=[1,2]*dims
  expected=target(codes,padding).numpy()
  x=dequantize(codes,.03125,7,dtype=torch.float32)
  actual=evaluate_operation(target,(x,padding),{})
  assert np.array_equal(actual.values,(expected-7)*x.scale)
 x=dequantize(codes,.03125,7,dtype=torch.float32)
 repeats=[2]+[3]*dims
 actual=evaluate_operation(torch.ops.aten.repeat.default,(x,repeats),{})
 expected=codes.repeat(repeats).numpy()
 assert np.array_equal(actual.values,(expected-7)*x.scale)
""")


def test_pool_adjoints_with_nonbinary_divisors_and_overlap():
    run_torch(r"""
import torch
import numpy as np
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import evaluate_operation
from merlin.targetgen.exact_arithmetic import ExactTensor
x=ExactTensor(np.full((1,1,3,3),F(0),object),torch.float32)
g=ExactTensor(np.full((1,1,2,2),F(1,7),object),torch.float32)
for op,args,expected in [
 (torch.ops.aten.avg_pool2d_backward.default,(g,x,[2,2],[1,1],[0,0],False,True,3),
  [[1,2,1],[2,4,2],[1,2,1]]),
 (torch.ops.aten._adaptive_avg_pool2d_backward.default,(g,x),
  [[1,2,1],[2,4,2],[1,2,1]])]:
 result=evaluate_operation(op,args,{})
 divisor=21 if 'adaptive' not in str(op) else 28
 assert result.values.tolist()==[[[[F(v,divisor) for v in row] for row in expected]]]
# Partial ceil windows and excluded padding have different exact denominators.
g=ExactTensor(np.full((1,1,2,2),F(1),object),torch.float32)
r=evaluate_operation(torch.ops.aten.avg_pool2d_backward.default,(g,x,[2,2],[2,2],[0,0],True,False,None),{})
assert r.values.tolist()==[[[[F(1,4),F(1,4),F(1,2)],[F(1,4),F(1,4),F(1,2)],[F(1,2),F(1,2),F(1)]]]]
""")


def test_fold_overlap_dilation_and_unbatched():
    run_torch(r"""
import torch
import numpy as np
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import evaluate_operation
from merlin.targetgen.exact_arithmetic import ExactTensor
for batched in [False,True]:
 for dilation in [[1,1],[2,2]]:
  size=[3,3] if dilation==[1,1] else [4,4]
  codes=torch.arange(1,17).reshape(4,4)
  if batched: codes=codes[None]
  x=ExactTensor(np.vectorize(lambda v:F(int(v),7),otypes=[object])(codes.numpy()),torch.float32)
  actual=evaluate_operation(torch.ops.aten.col2im.default,(x,size,[2,2],dilation,[0,0],[1,1]),{})
  # Small integer sums are exactly representable; torch here supplies only integer geometry.
  expected=torch.ops.aten.col2im.default(codes.double(),size,[2,2],dilation,[0,0],[1,1]).numpy()
  assert np.array_equal(actual.values,np.vectorize(lambda v:F(int(v),7),otypes=[object])(expected))
""")


def test_convolution_backward_and_traced_tuple_selection():
    run_torch(r"""
import operator
import torch
import numpy as np
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import run
from merlin.targetgen.exact_arithmetic import dequantize
import torch.ao.quantization.fx._decomposed
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def forward(self,g,x,w):
  return torch.ops.aten.convolution_backward.default(
   DQ(g,.125,0,-128,127,torch.int8),DQ(x,.25,0,-128,127,torch.int8),DQ(w,.5,0,-128,127,torch.int8),
   [1],[1,1],[0,0],[1,1],False,[0,0],1,[True,True,True])
g=torch.tensor([[[[1,2],[3,4]]]],dtype=torch.int8)
x=torch.arange(1,10,dtype=torch.int8).reshape(1,1,3,3)
w=torch.tensor([[[[1,2],[3,4]]]],dtype=torch.int8)
model=torch.export.export(Model(),(g,x,w)).module()
def select(value,index):return operator.getitem(value,index)
select._m2m_tuple_selection=True
for node in model.graph.nodes:
 if node.target is operator.getitem:node.target=select
model.recompile()
r=run(model,(g,x,w))
expected=torch.ops.aten.convolution_backward.default(g.double()*.125,x.double()*.25,w.double()*.5,[1],[1,1],[0,0],[1,1],False,[0,0],1,[True,True,True])
assert all(torch.equal(a,b.float()) for a,b in zip(r.output,expected))
assert r.contraction_count==1 and len(r.sites)==2
""")


def test_constant_creation_does_not_round_before_quantization():
    run_torch(r"""
import torch
import numpy as np
from fractions import Fraction as F
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.exact_quantized_reference import run
from merlin.targetgen.exact_arithmetic import quantize
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
class Model(torch.nn.Module):
 def forward(self,x):
  return (Q(torch.full_like(x,.1),.2,0,-128,127,torch.int8),
          Q(torch.ops.aten.fill.Scalar(x,.1),.2,0,-128,127,torch.int8),
          Q(torch.full([2,3],.1),.2,0,-128,127,torch.int8))
x=torch.zeros(2,3)
model=torch.export.export(Model(),(x,)).module()
r=run(model,(x,))
expected=quantize(np.full((2,3),F(.1),object),.2)
assert not expected.any() # exact half-even tie, not a rounded float32 constant
assert all(np.array_equal(v.numpy(),expected) for v in r.output)
""")


def test_nonuniform_adaptive_adjoint_divisors():
    run_torch(r"""
import torch
import numpy as np
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import evaluate_operation
from merlin.targetgen.exact_arithmetic import ExactTensor
x=ExactTensor(np.full((1,1,5,1),F(0),object),torch.float32)
g=ExactTensor(np.full((1,1,3,1),F(1,7),object),torch.float32)
r=evaluate_operation(torch.ops.aten._adaptive_avg_pool2d_backward.default,(g,x),{})
expected=[F(1,14),F(1,14)+F(1,21),F(1,21),F(1,21)+F(1,14),F(1,14)]
assert r.values.reshape(-1).tolist()==expected
""")


def test_grouped_convolution_adjoints_strides_dilation_and_masks():
    run_torch(r"""
import torch
import numpy as np
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import convolution_backward
from merlin.targetgen.exact_arithmetic import dequantize
# Integer products and partial sums here are bounded by 20000, hence torch's
# binary64 computation on integer codes below is provably exact.
x=(torch.arange(100).reshape(1,4,5,5)%7-3).to(torch.int8)
w=(torch.arange(48).reshape(6,2,2,2)%5-2).to(torch.int8)
g=(torch.arange(54).reshape(1,6,3,3)%5-2).to(torch.int8)
scales=[float(torch.tensor(v)) for v in [.03,.02,.01]]
operands=[dequantize(v,s,-3,dtype=torch.float32) for v,s in zip([g,x,w],scales)]
centered=[v.double()+3 for v in [g,x,w]]
for mask in [(True,True,True),(True,False,False),(False,True,False),(False,False,True)]:
 args=(*operands,[6],[2,2],[1,1],[2,2],False,[0,0],2,mask)
 result=convolution_backward(operands,args,'test',[])
 expected=torch.ops.aten.convolution_backward.default(*centered,*args[3:])
 factors=[F(scales[0])*F(scales[2]),F(scales[0])*F(scales[1]),F(scales[0])]
 for value,integer,factor,wanted in zip(result,expected,factors,mask):
  if not wanted: assert value is None
  else:
   exact=np.vectorize(lambda v:F(int(v))*factor,otypes=[object])(integer.numpy())
   assert np.array_equal(value.values,exact)
""")


def test_padding_crops_keep_original_reflection_endpoints():
    run_torch(r"""
import torch
import numpy as np
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import evaluate_operation
from merlin.targetgen.exact_arithmetic import dequantize
for dims in [1,2,3]:
 codes=torch.arange(5**dims,dtype=torch.int64).reshape(1,*([5]*dims))
 x=dequantize(codes,F(1,32),-3,-128,127,dtype=torch.float32)
 for mode in ['reflection','replication']:
  target=getattr(torch.ops.aten,f'{mode}_pad{dims}d').default
  for pair in [[-3,3],[3,-3],[-1,2],[2,-1]]:
   padding=pair*dims
   expected=target(codes,padding).numpy()
   actual=evaluate_operation(target,(x,padding),{})
   assert np.array_equal(actual.values,(expected+3)/32)
""")
