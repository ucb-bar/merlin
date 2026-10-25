"""Declared schemes keep their own rounding boundaries and fail closed on unknowns."""

import pytest
from test_exact_quantized_reference import run_torch

from merlin.runtime.tensor import Tensor
from merlin.targetgen.capsule_golden import _apply_epilogue, _residual_sum
from merlin.targetgen.exact_quantized_reference import _reference_scheme
from merlin.targetgen.quant_methodology import scheme_geometry


def scheme(granularity="tensor", rounding="single_rounding"):
    path = {"readout": [{"kind": "scale", "granularity": "per_command", "columns_per_scale": 3}]}
    return {
        "scale_value": "arbitrary_f32",
        "weight_scale_granularity": granularity,
        "readout_strategy": "f32_product_rne",
        "operand_sum_rounding": rounding,
        **scheme_geometry(path, granularity),
    }


@pytest.mark.parametrize("granularity", ["tensor", "column_block", "column"])
def test_f32_contraction_bias_relu_and_channel_axis(granularity):
    selected = scheme(granularity)
    run_torch(f"""
import numpy as np, torch
import torch.ao.quantization.fx._decomposed
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
DQC=torch.ops.quantized_decomposed.dequantize_per_channel.default
selected={selected!r}
scales=([.3]*7 if {granularity!r}=='tensor' else
        [.3]*3+[.7]*3+[.9] if {granularity!r}=='column_block' else [.1,.2,.3,.4,.5,.6,.7])
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('scales',torch.tensor(scales))
  self.register_buffer('zeros',torch.zeros(7,dtype=torch.int64))
  # Bias lies on each output's accumulator grid.
  self.register_buffer('bias',self.scales.clone())
 def forward(self,x,w):
  a=DQ(x,1.,0,-128,127,torch.int8)
  b=DQC(w,self.scales,self.zeros,1,-128,127,torch.int8)
  return Q(torch.relu(a@b+self.bias),.1,0,-128,127,torch.int8)
x=torch.tensor([[1,2],[-3,1]],dtype=torch.int8)
w=torch.tensor([[1,2,3,4,5,6,7],[2,1,2,1,2,1,2]],dtype=torch.int8)
model=torch.export.export(Model(),(x,w)).module()
result=run(model,(x,w),scheme=selected)
acc=x.numpy().astype('int32')@w.numpy().astype('int32')+1
multipliers=np.array([float(F(float(s)) / F(.1)) for s in Model().scales],dtype=np.float32)
expected=np.clip(np.rint(np.maximum(acc,0).astype('float32')*multipliers),-128,127).astype('int8')
assert np.array_equal(result.output[0].numpy() if isinstance(result.output,tuple) else result.output.numpy(),expected)
assert result.arithmetic=='f32_product_rne'
assert result.sites[0]['readout_order']=='f32 accumulator; f32 product; boundary RNE'
""")


def test_power_of_two_large_accumulator_preserves_v1_and_selects_f32_explicitly():
    run_torch("""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def forward(self,x,w):
  return Q(DQ(x,1.,0,-128,127,torch.int8)@DQ(w,1.,0,-128,127,torch.int8),float(2**26),0,-128,127,torch.int8)
x=torch.full((1,4097),-128,dtype=torch.int8); x[0,-1]=1
w=torch.full((4097,1),-64,dtype=torch.int8); w[-1,0]=1
model=torch.export.export(Model(),(x,w)).module()
selected=dict(scale_value='power_of_two',weight_scale_granularity='tensor',operand_sum_rounding='single_rounding')
def leaf(result):return torch.utils._pytree.tree_leaves(result.output)[0]
legacy=leaf(run(model,(x,w)))
exact=leaf(run(model,(x,w),scheme={**selected,'readout_strategy':'exact_rational_host'}))
f32=leaf(run(model,(x,w),scheme={**selected,'readout_strategy':'f32_product_rne'}))
assert legacy.numpy().tobytes()==exact.numpy().tobytes()==bytes([1])
assert f32.item()==0
""")


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_max_pool_values_negative_ties_dilation_and_ceil(dims):
    run_torch(f"""
import numpy as np, torch
from merlin.targetgen.exact_arithmetic import dequantize,materialize
from merlin.targetgen.exact_quantized_reference import evaluate_operation
x=-torch.arange(1,1+5**{dims},dtype=torch.float32).reshape(1,1,*([5]*{dims}))
x.flatten()[1]=x.flatten()[0]  # Ties remain values, with no invented indices.
value=dequantize(x.to(torch.int16),.25,dtype=torch.float32,quant_min=-1024,quant_max=1024)
for dilation in [1,2]:
 for ceil in [False,True]:
  op=getattr(torch.ops.aten,'max_pool{dims}d').default
  args=([2]*{dims},[2]*{dims},[0]*{dims},[dilation]*{dims},ceil)
  actual=evaluate_operation(op,(value,*args),{{}})
  assert torch.equal(materialize(actual),op(x*.25,*args))
""")


@pytest.mark.parametrize("readout_first", [False, True])
def test_max_pool_composes_with_convolution_relu_and_readout(readout_first):
    run_torch(f"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def forward(self,x,w):
  y=torch.conv2d(DQ(x,.3,0,-128,127,torch.int8),DQ(w,1.,0,-128,127,torch.int8))
  y=torch.relu(y)
  if {readout_first!r}: y=DQ(Q(y,1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
  y=torch.ops.aten.max_pool2d_with_indices.default(y,[2,2])[0]
  return Q(y,1.,0,-128,127,torch.int8)
x=torch.tensor([[[[-9,-5,5,5],[-7,-6,2,1],[1,3,7,9],[4,2,9,8]]]],dtype=torch.int8)
w=torch.ones((1,1,1,1),dtype=torch.int8)
model=torch.export.export(Model(),(x,w)).module()
result=run(model,(x,w),scheme={scheme()!r})
expected=torch.tensor([[[[0,2],[1,3]]]],dtype=torch.int8)
assert torch.equal(torch.utils._pytree.tree_leaves(result.output)[0],expected)
""")


def test_max_pool_unknown_padding_and_live_indices_refuse():
    run_torch(f"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.exact_quantized_reference import run
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def __init__(self,padding,indices): super().__init__(); self.padding=padding; self.indices=indices
 def forward(self,x):
  values,indices=torch.ops.aten.max_pool2d_with_indices.default(torch.relu(DQ(x,.25,0,-128,127,torch.int8)),[2,2],[1,1],[self.padding]*2)
  return indices if self.indices else values
x=torch.ones((1,1,3,3),dtype=torch.int8)
for padding,indices,message in [(1,False,'padding semantics are UNKNOWN'),(0,True,'live indices refuse')]:
 model=torch.export.export(Model(padding,indices),(x,)).module()
 try: run(model,(x,),scheme={scheme()!r})
 except ValueError as exc: assert message in str(exc),str(exc)
 else: raise AssertionError('unqualified max pool accepted')
""")


@pytest.mark.parametrize("kind", ["mean", "avg_pool", "adaptive_avg_pool"])
def test_average_folds_reciprocal_into_one_f32_multiplier(kind):
    run_torch(f"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def forward(self,x):
  y=DQ(x,.3,0,-128,127,torch.int8)
  if {kind!r}=='mean':y=y.mean((2,3),keepdim=True)
  elif {kind!r}=='avg_pool':y=torch.nn.functional.avg_pool2d(y,(2,5))
  else:y=torch.nn.functional.adaptive_avg_pool2d(y,(1,1))
  return Q(y,.1,0,-128,127,torch.int8)
x=torch.tensor([1]*5+[0]*5,dtype=torch.int8).reshape(1,1,2,5)
model=torch.export.export(Model(),(x,)).module()
def leaf(result):return torch.utils._pytree.tree_leaves(result.output)[0].item()
assert leaf(run(model,(x,)))==1
assert leaf(run(model,(x,),scheme={scheme()!r}))==2
""")


def test_explicit_per_operand_sum_differs_from_single_rounding():
    selected = scheme("column_block", "per_operand")
    run_torch(f"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def __init__(self,explicit): super().__init__(); self.explicit=explicit
 def forward(self,a,b):
  a=DQ(a,.5,0,-128,127,torch.int8); b=DQ(b,.5,0,-128,127,torch.int8)
  if self.explicit:
   a=DQ(Q(a,1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
   b=DQ(Q(b,1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
   a=a.clone().view(a.shape)
   b=torch.relu(b)
  return Q(torch.relu(a+b),1.,0,-128,127,torch.int8)
x=torch.tensor([[1,3,127,-3]],dtype=torch.int8)
plain=torch.export.export(Model(False),(x,x)).module()
explicit=torch.export.export(Model(True),(x,x)).module()
def leaf(result):return torch.utils._pytree.tree_leaves(result.output)[0].tolist()
assert leaf(run(plain,(x,x)))==[[1,3,127,0]]
assert leaf(run(explicit,(x,x),scheme={selected!r}))==[[0,4,127,0]]
try:run(plain,(x,x),scheme={selected!r})
except ValueError as exc:assert 'explicit requantization nodes' in str(exc)
else:raise AssertionError('undeclared double rounding accepted')
""")


@pytest.mark.parametrize("granularity", ["tensor", "column_block", "column"])
def test_generated_predecessor_power_vectors_through_reference(granularity):
    run_torch(f"""
import numpy as np,torch
import torch.ao.quantization.fx._decomposed
from merlin.runtime.exact_readout import pred_pow2_vectors
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
multiplier=np.nextafter(np.float32(1/3),np.float32(0))
acc=pred_pow2_vectors(multiplier,-127,127,range(7))
assert len(acc)>10
products=acc.astype('float32')*multiplier
assert any(products==np.nextafter(np.float32(1),np.float32(0)))
class Model(torch.nn.Module):
 def forward(self,x,w):
  return Q(DQ(x,float(multiplier),0,-128,127,torch.int8)@DQ(w,1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
x=torch.tensor(acc[:,None],dtype=torch.int8); w=torch.ones((1,7),dtype=torch.int8)
model=torch.export.export(Model(),(x,w)).module()
result=run(model,(x,w),scheme={scheme(granularity)!r})
actual=torch.utils._pytree.tree_leaves(result.output)[0].numpy()
expected=np.repeat(np.clip(np.rint(products),-128,127).astype('int8')[:,None],7,axis=1)
assert np.array_equal(actual,expected)
""")


def test_capsule_scheme_dispatch_column_block_and_per_operand():
    lhs, rhs = Tensor((1, 4), [1, 3, 127, -3], "i8"), Tensor((1, 4), [1, 3, 127, -3], "i8")
    attrs = {"lhs_scale": 0.5, "rhs_scale": 0.5, "bound_lsb": 0}
    assert _residual_sum(lhs, rhs, attrs, op="test").data == [1, 3, 127, -3]
    assert _residual_sum(lhs, rhs, {**attrs, "operand_sum_rounding": "per_operand"}, op="test").data == [0, 4, 128, -4]
    column_attrs = {
        "epilogue": ["acc_scale"],
        "acc_scale_by_column": [0.3] * 3 + [0.7] * 3 + [0.9],
        "quantization_scheme": scheme("column_block"),
    }
    assert _apply_epilogue(Tensor((1, 7), [5] * 7, "i32"), column_attrs, {}).data == [2, 2, 2, 4, 4, 4, 4]
    with pytest.raises(ValueError, match="within a fact-derived column block"):
        _apply_epilogue(
            Tensor((1, 7), [5] * 7, "i32"),
            {**column_attrs, "acc_scale_by_column": [0.1, 0.2, 0.3, 0.7, 0.7, 0.7, 0.9]},
            {},
        )


def test_unresolved_scheme_refuses_and_capsule_exact_readout_stays_exact():
    with pytest.raises(ValueError, match="resolved from output-path facts"):
        _reference_scheme(None, {**scheme("column_block"), "columns_per_scale": "UNKNOWN"})
    value = Tensor((1, 1), [2**25 + 1], "i32")
    attrs = {
        "epilogue": ["acc_scale"],
        "scale_numerator": 1,
        "scale_denominator": 2**26,
        "quantization_scheme": {"readout_strategy": "exact_rational_host", "operand_sum_rounding": "single_rounding"},
    }
    assert _apply_epilogue(value, attrs, {}).data == [1]


def test_generated_predecessor_vectors_at_explicit_operand_loads():
    run_torch(f"""
import numpy as np,torch
import torch.ao.quantization.fx._decomposed
from merlin.runtime.exact_readout import pred_pow2_vectors
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
multiplier=np.nextafter(np.float32(1/3),np.float32(0))
acc=pred_pow2_vectors(multiplier,-127,127,range(7))
class Model(torch.nn.Module):
 def forward(self,x,y):
  x=DQ(Q(DQ(x,float(multiplier),0,-128,127,torch.int8),1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
  y=DQ(Q(DQ(y,.5,0,-128,127,torch.int8),1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
  return Q(x+y,1.,0,-128,127,torch.int8)
x=torch.tensor(acc,dtype=torch.int8); y=torch.ones_like(x)
model=torch.export.export(Model(),(x,y)).module()
result=run(model,(x,y),scheme={scheme("column_block", "per_operand")!r})
actual=torch.utils._pytree.tree_leaves(result.output)[0].numpy()
expected=np.clip(np.rint(acc.astype('float32')*multiplier),-128,127).astype('int8')
assert np.array_equal(actual,expected)
""")


def test_single_rounding_residual_retains_its_one_boundary_for_distinct_grids():
    run_torch(f"""
import numpy as np,torch
import torch.ao.quantization.fx._decomposed
from fractions import Fraction as F
from merlin.targetgen.exact_quantized_reference import run
from merlin.targetgen.core_aten_w8a8_reference import selected_pt2e_reference
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
class Model(torch.nn.Module):
 def forward(self,x,y):
  a=DQ(x,.3,0,-128,127,torch.int8).view(-1)
  b=DQ(y,.7,0,-128,127,torch.int8).view(-1)
  return Q(torch.relu(a+b),1.,0,-128,127,torch.int8)
x=torch.arange(-127,128,dtype=torch.int8); y=torch.flip(x,[0])
model=torch.export.export(Model(),(x,y)).module()
model._recipe_quantization_stats={{'quantization_scheme':{scheme()!r}}}
result,identity=selected_pt2e_reference(model,(x,y))
expected=[min(127,max(0,round(F(.3)*int(a)+F(.7)*int(b)))) for a,b in zip(x,y)]
assert torch.utils._pytree.tree_leaves(result.output)[0].tolist()==expected
assert identity['arithmetic']=='f32_product_rne'
""")


def test_f32_source_refuses_invalid_block_scales_and_oversized_operand_multipliers():
    run_torch(f"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen.exact_quantized_reference import run
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
DQC=torch.ops.quantized_decomposed.dequantize_per_channel.default
class BadBlock(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.register_buffer('scales',torch.tensor([.3,.3,.7]))
  self.register_buffer('zeros',torch.zeros(3,dtype=torch.int64))
 def forward(self,x,w):
  return Q(DQ(x,1.,0,-128,127,torch.int8)@DQC(w,self.scales,self.zeros,1,-128,127,torch.int8),1.,0,-128,127,torch.int8)
x=torch.ones((1,1),dtype=torch.int8); w=torch.ones((1,3),dtype=torch.int8)
model=torch.export.export(BadBlock(),(x,w)).module()
try:run(model,(x,w),scheme={scheme("column_block")!r})
except ValueError as exc:assert 'within a fact-derived column block' in str(exc)
else:raise AssertionError('invalid block accepted')
class BadLoad(torch.nn.Module):
 def forward(self,x,y):
  x=DQ(Q(DQ(x,2.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
  y=DQ(Q(DQ(y,.5,0,-128,127,torch.int8),1.,0,-128,127,torch.int8),1.,0,-128,127,torch.int8)
  return Q(x+y,1.,0,-128,127,torch.int8)
model=torch.export.export(BadLoad(),(x,x)).module()
try:run(model,(x,x),scheme={scheme("column_block", "per_operand")!r})
except ValueError as exc:assert 'multipliers in (0, 1]' in str(exc)
else:raise AssertionError('oversized operand multiplier accepted')
""")


def test_capsule_residual_declares_exact_double_rounding_through_golden_dispatch():
    import numpy as np

    from merlin.targetgen.capsule_golden import golden, materialize_capsule_leaves

    capsule = {
        "inputs": [{"name": name, "role": "input", "shape": [2, 7], "dtype": "i8"} for name in ("lhs", "rhs")],
        "operation": {
            "op": "residual_add",
            "attributes": {
                "lhs": "lhs",
                "rhs": "rhs",
                "out": "out",
                "lhs_scale": 0.5,
                "rhs_scale": 0.5,
                "bound_lsb": 0,
                "output_dtype": "i8",
                "epilogue": ["relu"],
                "quantization_scheme": scheme("column_block", "per_operand"),
            },
        },
    }
    inputs = materialize_capsule_leaves(capsule)
    operands = [np.rint(np.asarray(inputs[name].data, dtype=np.float32) * 0.5) for name in ("lhs", "rhs")]
    expected = np.clip(operands[0] + operands[1], 0, 127).reshape(2, 7).astype(int).tolist()
    assert golden(capsule) == {"out": expected}
    capsule["operation"]["attributes"]["bound_lsb"] = 1
    with pytest.raises(ValueError, match="bound_lsb=0"):
        golden(capsule)
