"""A declared observer domain grants no access to graded input values."""
import pytest

NONFINITE='byte_storage_zero_nan_saturating_infinity'
STEP=.015625
RECIPE=dict(schema='quant_recipe_v1',status='derived',families=['contraction'],
    weight=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-127,quant_max=127),
    activation=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-128,quant_max=127,mode='static'),
    calibration_domain=dict(schema='merlin.declared_observer_domain.v1',minimum=-1.,maximum=1.,evaluation_inputs='never_read',scale_policy='enclosing_power_of_two'))


def qdq(graph,node,scale=STEP,quantize=True):
    """``node`` through an int8 quantize/dequantize pair at ``scale`` (only the dequantize if not ``quantize``)."""
    import torch
    import torch.ao.quantization.fx._decomposed
    ops=torch.ops.quantized_decomposed;args=(scale,0,-128,127,torch.int8)
    if quantize:node=graph.call_function(ops.quantize_per_tensor.default,(node,*args))
    return graph.call_function(ops.dequantize_per_tensor.default,(node,*args))


def module(graph,result):
    import torch
    graph.output(result);return torch.fx.GraphModule(torch.nn.Module(),graph)


def declare(model,**policy):
    """Declare the byte-storage nonfinite contract (plus ``policy``) as ``model``'s capture policy."""
    model._recipe_quantization_stats={'framework_capture_policy':{'quantization_nonfinite':NONFINITE,**policy}}
    return model


def readout(*args):
    """A model of one ``m2m_q.exact_readout(acc, STEP, [STEP], 1., bias, *args)``."""
    import torch
    graph=torch.fx.Graph();acc=graph.placeholder('acc');bias=graph.placeholder('bias')
    return module(graph,graph.call_function(torch.ops.m2m_q.exact_readout.default,(acc,STEP,[STEP],1.,bias,*args)))


def test_program_qparams_and_integer_kernel_are_payload_independent():
    torch=pytest.importorskip('torch')
    from merlin.targetgen._recipe_quantizer import apply_recipe
    from merlin.capture.integerization import integerize_static_pt2e

    class Model(torch.nn.Module):
        def forward(self,x,y):
            return x @ y

    recipe=dict(RECIPE,quantization_nonfinite=NONFINITE)
    graphs=[];qparams=[]
    class ForbiddenStream:
        def __iter__(self):raise AssertionError('evaluation calibration stream was inspected')
    for number in (0.0,1.e30,float("inf"),float("nan")):
        inputs=(torch.full((2,2),number),torch.full((2,2),number))
        model=apply_recipe(Model(),recipe,example_inputs=inputs,calibration_inputs=ForbiddenStream())
        stats=model._recipe_quantization_stats
        assert stats['framework_capture_policy']['calibration_source']=='declared_observer_domain'
        qparams.append([(str(n.target),n.args[1:]) for n in model.graph.nodes if 'quantized_decomposed' in str(n.target)])
        from merlin.targetgen.exact_quantized_reference import run
        independent=run(model,inputs).output
        model,_=integerize_static_pt2e(model,inputs)
        actual=model(*inputs)
        torch.utils._pytree.tree_map(lambda a,b: torch.testing.assert_close(a,b,rtol=0,atol=0), actual, independent)
        graphs.append(model.code.encode())
    assert all(params==qparams[0] for params in qparams)
    assert all(graph==graphs[0] for graph in graphs)  # Complete executable integer graph, not a near-value comparison.


@pytest.mark.parametrize('code', [1, 15])
def test_nonexact_affine_intermediate_refuses_before_payload_agreement(code):
    torch = pytest.importorskip('torch')
    from merlin.targetgen._recipe_quantizer import apply_recipe, NoEligibleRecipeOperations

    class Model(torch.nn.Module):
        def forward(self, x):return x * 1.1

    recipe = dict(RECIPE, families=['contraction', 'affine_map'])
    # Code 1 happened to agree, whereas code 15 rounds the exact 16.5+epsilon
    # to an intermediate f32 tie. Both programs must receive the same refusal.
    with pytest.raises(NoEligibleRecipeOperations) as caught:
        apply_recipe(Model(), recipe, example_inputs=(torch.tensor([code / 64.]),))
    assert caught.value.code == 'declared_affine_domain_unproved'
    assert caught.value.details['integer_recipes'][0]['declared_affine_domain']['status'] == 'refused'


def test_affine_lattice_proof_covers_all_byte_codes_and_partial_sums():
    torch = pytest.importorskip('torch')
    from fractions import Fraction
    from merlin.targetgen.affine_float_domain import prove_affine_domain

    coefficients = (Fraction(3, 128), Fraction(-5, 64))
    offset = Fraction(7, 256)
    prove_affine_domain(coefficients, offset)
    left = torch.arange(-128, 128, dtype=torch.float32)[:, None]
    right = torch.arange(-128, 128, dtype=torch.float32)[None, :]
    got = left * float(coefficients[0]) + right * float(coefficients[1]) + float(offset)
    # Integer-grid expected words cover every pair, independent of input data.
    expected = (left.to(torch.int64) * 6 - right.to(torch.int64) * 20 + 7).double() / 256
    assert torch.equal(got.double(), expected)
    with pytest.raises(ValueError, match='intermediate floating rounding'):
        prove_affine_domain((Fraction(float(1.1)) / 64,), Fraction(0))
    with pytest.raises(ValueError, match='non-dyadic'):
        prove_affine_domain((Fraction(1, 3),), Fraction(0))


@pytest.mark.parametrize('token,low,high',[('int8',-128,127),('uint8',0,255),('int8',-8,7)])
def test_declared_nonfinite_byte_boundary_matches_every_half_nan_word(token,low,high):
    torch=pytest.importorskip('torch')
    import numpy as np
    from merlin.targetgen.exact_arithmetic import quantize
    words=np.arange(65536,dtype=np.uint16)
    values=torch.from_numpy(words.view(np.float16).copy()).float()
    special=values[~torch.isfinite(values)]
    import torch.ao.quantization.fx._decomposed
    dtype=getattr(torch,token)
    expected=torch.ops.quantized_decomposed.quantize_per_tensor.default(special,0.015625,0,low,high,dtype)
    got=quantize(special,0.015625,0,low,high,nonfinite='byte_storage_zero_nan_saturating_infinity')
    assert got.tolist()==expected.tolist()
    with pytest.raises(ValueError,match='finite rational'):
        quantize(special,0.015625,0,low,high)


def test_rational_aggregate_can_feed_a_declared_nan_selector():
    torch=pytest.importorskip('torch')
    from merlin.targetgen.exact_quantized_reference import run
    graph=torch.fx.Graph()
    pooled=graph.call_function(torch.ops.aten.avg_pool2d.default,(qdq(graph,graph.placeholder('x')),[2,2]))
    mask=graph.call_function(torch.ops.aten.isnan.default,(pooled,))
    fill=graph.call_function(torch.ops.aten.full_like.default,(pooled,0.))
    selected=graph.call_function(torch.ops.aten.where.self,(mask,fill,pooled))
    model=declare(module(graph,qdq(graph,selected)))
    for value in (0.,1.e30,float('inf'),float('nan')):
        inputs=(torch.full((1,1,2,2),value),)
        expected=run(model,inputs).output
        assert torch.equal(expected.view(torch.uint8),model(*inputs).view(torch.uint8))


@pytest.mark.parametrize('value',[0.,float('nan'),float('inf'),-float('inf')])
def test_pool_adjoint_reads_only_primal_shape(value):
    torch=pytest.importorskip('torch')
    import numpy as np
    from fractions import Fraction
    from merlin.targetgen.exact_arithmetic import ExactTensor
    from merlin.targetgen.exact_quantized_reference import evaluate_operation
    gradient=ExactTensor(np.full((1,1,2,2),Fraction(1,7),dtype=object),torch.float32)
    primal=torch.full((1,1,3,3),value)
    for target,args in [
        (torch.ops.aten._adaptive_avg_pool2d_backward.default,(gradient,primal)),
        (torch.ops.aten.avg_pool2d_backward.default,(gradient,primal,[2,2],[1,1],[0,0],False,True,None)),
    ]:
        actual=evaluate_operation(target,args,{})
        expected=np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=object)*Fraction(1,28)
        assert np.array_equal(actual.values[0,0],expected)


@pytest.mark.parametrize('mask',[(True,False,False),(False,True,False),(False,False,True)])
def test_convolution_adjoint_uses_only_selected_value_dependencies(mask):
    torch=pytest.importorskip('torch')
    import numpy as np
    from merlin.targetgen.exact_arithmetic import exact_array,ExactTensor
    from merlin.targetgen.exact_quantized_reference import convolution_backward
    gradient=ExactTensor(exact_array(torch.ones(1,1,2,2)),torch.float32)
    source=torch.full((1,1,3,3),1. if mask[1] else float('nan'))
    weight=torch.full((1,1,2,2),1. if mask[0] else float('inf'))
    args=(gradient,source,weight,[1],[1,1],[0,0],[1,1],False,[0,0],1,list(mask))
    actual=convolution_backward(args[:3],args,'adjoint',[])
    baseline=(gradient,torch.ones_like(source),torch.ones_like(weight),*args[3:])
    expected=convolution_backward(baseline[:3],baseline,'adjoint',[])
    for wanted,left,right in zip(mask,actual,expected,strict=True):
        if wanted:assert np.array_equal(left.values,right.values)
        else:assert left is None and right is None


@pytest.mark.parametrize('beta',[-2.,0.,2.])
@pytest.mark.parametrize('relu',[False,True])
def test_readout_special_overlay_is_only_the_broadcast_bias_dependence(beta,relu):
    torch=pytest.importorskip('torch')
    pytest.importorskip('m2m.capture.integer_readout')
    from fractions import Fraction
    from merlin.targetgen.declared_readout_specials import retain_special_readouts
    model=readout(beta,STEP,3,-128,127,relu,False)
    assert retain_special_readouts(model,NONFINITE)==1
    values=torch.tensor([[64,-64,0,128]],dtype=torch.int32)
    residual=torch.tensor([float('nan'),float('inf'),-float('inf'),1.25])
    got=model(values,residual)
    expected=[]
    for index,(integer,b) in enumerate(zip(values[0].tolist(),residual.tolist(),strict=True)):
        if beta and index<3:code=0 if index==0 else 127 if (index==1)==(beta>0) else max(-128,3) if relu else -128
        else:
            value=Fraction(integer,4096)+(Fraction(5,4)*Fraction(beta) if index==3 else 0)
            code=min(127,max(3 if relu else -128,round(value/Fraction(1,64))+3))
        expected.append(code)
    assert got.tolist()==[expected]
    original=values.clone();model(values,torch.zeros_like(residual));assert torch.equal(values,original)


def test_declared_special_bias_is_discarded_only_at_the_final_quantizer():
    torch=pytest.importorskip('torch')
    from merlin.targetgen.exact_quantized_reference import run
    graph=torch.fx.Graph();x=graph.placeholder('x');bias=graph.placeholder('bias')
    result=graph.call_function(torch.ops.aten.add.Tensor,(qdq(graph,x),bias))
    relu=graph.call_function(torch.ops.aten.relu.default,(result,))
    model=declare(module(graph,qdq(graph,relu)))
    inputs=(torch.ones(4),torch.tensor([float('nan'),float('inf'),-float('inf'),.25]))
    actual=run(model,inputs).output;expected=model(*inputs)
    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
    # Removing the Q boundary must not manufacture an IEEE NaN payload claim.
    output=next(n for n in graph.nodes if n.op=='output');output.args=(relu,)
    graph.eliminate_dead_code();model.recompile()
    with pytest.raises(ValueError,match='final byte quantizer'):run(model,inputs)



@pytest.mark.parametrize('beta',[0.,1.,1.5])
@pytest.mark.parametrize('relu',[False,True])
def test_floating_residual_overlay_preserves_single_nan_words(beta,relu):
    torch=pytest.importorskip('torch')
    pytest.importorskip('m2m.capture.integer_readout')
    from merlin.targetgen.declared_readout_specials import retain_special_readouts
    model=readout(beta,None,0,-128,127,relu)
    assert retain_special_readouts(model,NONFINITE,floating_policy='single_residual_quiet_word_f32')==1
    original=torch.tensor([0x7f800001,0xffc00023,0x7f800000,0xff800000,0x3fa00000],dtype=torch.uint32)
    residual=original.view(torch.float32);accumulator=torch.full((2,5),64,dtype=torch.int32)
    actual=model(accumulator,residual)
    if beta:
        expected=original.clone();expected[0]=0x7fc00001
        if relu:expected[3]=0
        expected[4]=torch.tensor([.015625+beta*1.25]).view(torch.uint32)[0]
    else:expected=torch.full((5,),0x3c800000,dtype=torch.uint32)
    assert torch.equal(actual.view(torch.uint32),expected.expand(2,5))
    assert torch.equal(residual.view(torch.uint32),original)


def test_floating_reference_needs_a_declared_single_nan_word_contract():
    torch=pytest.importorskip('torch')
    from merlin.targetgen.exact_quantized_reference import run
    graph=torch.fx.Graph();x=graph.placeholder('x');bias=graph.placeholder('bias')
    model=declare(module(graph,graph.call_function(torch.ops.aten.add.Tensor,(qdq(graph,x),bias))),readout_specials='single_residual_quiet_word_f32')
    bias=torch.tensor([0x7f800001,0xffc00023,0x7f800000,0xff800000],dtype=torch.uint32).view(torch.float32)
    inputs=(torch.ones(4),bias)
    assert torch.equal(run(model,inputs).output.view(torch.uint32),model(*inputs).view(torch.uint32))


def test_ambiguous_floating_residual_coefficient_refuses_statically():
    pytest.importorskip('torch')
    pytest.importorskip('m2m.capture.integer_readout')
    from merlin.targetgen.declared_readout_specials import retain_special_readouts
    from merlin.targetgen.capture_refusals import NumericCaptureRefused
    with pytest.raises(NumericCaptureRefused,match='single-NaN word proof'):
        retain_special_readouts(readout(-1.),NONFINITE,floating_policy='single_residual_quiet_word_f32')


def test_exact_capture_agreement_compares_nonfinite_storage_words():
    torch=pytest.importorskip('torch')
    from merlin.targetgen._capture_integer_agreement import integerized_agreement
    words=torch.tensor([0x7fc00001,0xffc00023,0x7f800000,0xff800000],dtype=torch.uint32)
    def abi(value):return [value],[{'dtype':str(value.dtype),'shape':list(value.shape)}]
    left=words.view(torch.float32)
    same=integerized_agreement(left,left.clone(),atol=0.,rtol=0.,output_abi=abi)
    assert same['status']=='passed' and not same['finite']
    different=words.clone();different[0]=0x7fc00002
    assert integerized_agreement(left,different.view(torch.float32),atol=0.,rtol=0.,output_abi=abi)['status']=='failed'


def test_static_readout_refusal_has_a_machine_readable_capture_receipt(tmp_path,monkeypatch):
    from merlin.capture import integerization
    from merlin.targetgen.capture_refusals import NumericCaptureRefused,read_refusal
    def refuse(*args,**kwargs):raise NumericCaptureRefused('unproved coefficient',code='unquantized_special_readout_unproved')
    monkeypatch.setattr(integerization,'integerize_static_pt2e',refuse)
    with pytest.raises(SystemExit) as caught:integerization.integerize_capture_model(None,(),numerics='integer_readout',directory=tmp_path)
    assert caught.value.code==4
    assert read_refusal(tmp_path/'refusal.json').code=='unquantized_special_readout_unproved'


def test_selected_f32_overflow_is_declared_without_changing_legacy_default():
    torch=pytest.importorskip('torch')
    from merlin.targetgen.exact_quantized_reference import run
    graph=torch.fx.Graph();scale=float(torch.tensor(1.e38,dtype=torch.float32))
    a,b=(qdq(graph,graph.placeholder(name),scale,quantize=False) for name in 'xy')
    model=module(graph,graph.call_function(torch.ops.aten.mm.default,(a,b)))
    inputs=(torch.tensor([[3]],dtype=torch.int8),torch.tensor([[5]],dtype=torch.int8))
    with pytest.raises(ValueError,match='boundary overflows'):run(model,inputs)
    declare(model,f32_boundary_overflow='ieee_infinity')
    assert run(model,inputs).output.view(torch.uint32).tolist()==[[0x7f800000]]
