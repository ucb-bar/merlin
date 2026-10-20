"""Native sibling projections preserve grids and real integer-reference lowering."""

# ruff: noqa: E501 -- embedded interpreter fixtures retain their own source formatting.

from __future__ import annotations

import json
import os
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import _projection_input_reuse as reuse


@pytest.mark.parametrize("policy", [None, {}, {"schema": "other"}, {"schema": reuse.SCHEMA, "raw": True}])
def test_policy_is_explicit_and_closed(policy):
    with pytest.raises(reuse.ProjectionReuseError):
        reuse.validate(policy, 4)


@pytest.mark.parametrize("width", [None, False, 0, -1, 1.5])
def test_physical_scale_width_must_be_selected(width):
    with pytest.raises(reuse.ProjectionReuseError):
        reuse.validate({"schema": reuse.SCHEMA}, width)


@pytest.mark.parametrize("budget", [False, 0, -1, 1.5, "64"])
def test_optional_nonuniform_readout_budget_is_a_positive_integer(budget):
    with pytest.raises(reuse.ProjectionReuseError):
        reuse.validate({"schema": reuse.SCHEMA, "max_nonuniform_readout_blocks": budget}, 16)


_SCRIPT = r"""
import copy, json, operator, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
import torch
import torchao
from merlin.targetgen import _projection_input_reuse as P

torch.set_num_threads(1)
torch.manual_seed(17)
DQ=torch.ops.quantized_decomposed.dequantize_per_tensor.default
Q=torch.ops.quantized_decomposed.quantize_per_tensor.default
LINEAR=torch.ops.aten.linear.default
POLICY={'schema':P.SCHEMA}

def fixture(widths=(2,4), *, duplicate_q=False, two_inputs=False, return_codes=False):
    root=torch.nn.Module();g=torch.fx.Graph();x=g.placeholder('x');x.meta['val']=torch.empty((3,5),dtype=torch.float32)
    other=g.placeholder('other') if two_inputs else x;other.meta['val']=torch.empty((3,5),dtype=torch.float32)
    q=g.call_function(Q,(x,.125,0,-128,127,torch.int8));xd=g.call_function(DQ,(q,1.,0,-128,127,torch.int8))
    outputs=[];sites=[]
    for i,n in enumerate(widths):
        if i and (duplicate_q or two_inputs):
            qi=g.call_function(Q,(other,.125,0,-128,127,torch.int8));lhs=g.call_function(DQ,(qi,1.,0,-128,127,torch.int8))
        else:lhs=xd
        root.register_buffer('w'+str(i),torch.randint(-127,128,(n,5),dtype=torch.int8))
        root.register_buffer('b'+str(i),torch.arange(n,dtype=torch.float32)-1)
        root.register_buffer('s'+str(i),torch.full((n,),.5 if i==0 else .25))
        w=g.get_attr('w'+str(i));wd=g.call_function(DQ,(w,1.,0,-127,127,torch.int8));b=g.get_attr('b'+str(i))
        linear=g.call_function(LINEAR,(lhs,wd,b));linear.meta['val']=torch.empty((3,n),dtype=torch.float32)
        s=g.get_attr('s'+str(i));mul=g.call_function(torch.ops.aten.mul.Tensor,(linear,s))
        quant=g.call_function(Q,(mul,1.,0,-128,127,torch.int8))
        out=quant if return_codes else g.call_function(DQ,(quant,.125 if i==0 else .25,0,-128,127,torch.int8))
        sites.append((linear,mul,quant));outputs.append(out)
    g.output(tuple(outputs));model=torch.fx.GraphModule(root,g)
    return model,sites,torch.randint(-8,9,(3,5),dtype=torch.float32)*.125

def run(model,x,other=False):
    with torch.no_grad():return model(x,x+.125) if other else model(x)

def count(model):return sum(n.op=='call_function' and n.target==LINEAR for n in model.graph.nodes)
def equal(a,b):assert len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b))

def exact_grids():
    m,sites,x=fixture();before=run(m,x);r=P.rewrite(m,POLICY,columns_per_scale=2)
    assert r['contractions_removed']==1 and count(m)==1
    equal(before,run(m,x))
    dqs=[n for n in m.graph.nodes if n.op=='call_function' and n.target==DQ and n.args[1] in (.125,.25)]
    assert sorted(n.args[1] for n in dqs)==[.125,.25]
    return r

def all_i8_and_ties():
    m,sites,x=fixture();old=copy.deepcopy(m);P.rewrite(m,POLICY,columns_per_scale=2)
    for start in range(-128,128,3):
        v=torch.arange(start,start+15,dtype=torch.float32).clamp(-128,127).reshape(3,5)*.125
        equal(run(old,v),run(m,v))

def output_allocations():
    m,_,x=fixture(return_codes=True);old=run(m,x);P.rewrite(m,POLICY,columns_per_scale=2);out=run(m,x);equal(old,out)
    assert all(a.is_contiguous() for a in out)
    assert out[0].untyped_storage().data_ptr()!=out[1].untyped_storage().data_ptr()
    keep=out[1].clone();out[0].zero_();assert torch.equal(keep,out[1])

def duplicate_inputs():
    m,_,x=fixture(duplicate_q=True);old=run(m,x);r=P.rewrite(m,POLICY,columns_per_scale=2);assert count(m)==1;equal(old,run(m,x))

def distinct_inputs():
    m,_,x=fixture(two_inputs=True);r=P.rewrite(m,POLICY,columns_per_scale=2);assert not r['groups'] and count(m)==2

def odd_widths():
    m,_,x=fixture((3,5));old=run(m,x);r=P.rewrite(m,POLICY,columns_per_scale=1);assert count(m)==1;equal(old,run(m,x))
    assert r['groups'][0]['output_columns']==[3,5]

def incompatible_physical_tail():
    m,_,x=fixture((3,5));old=run(m,x);r=P.rewrite(m,POLICY,columns_per_scale=2)
    assert not r['groups'] and any('physical scale group' in row['reason'] for row in r['refused']);equal(old,run(m,x))

def compatible_physical_tail():
    m,_,x=fixture((3,5));m.s1.fill_(.5);old=run(m,x);r=P.rewrite(m,POLICY,columns_per_scale=2);assert count(m)==1;equal(old,run(m,x))

def existing_attribute():
    m,_,x=fixture();m.register_buffer('_merlin_shared_projection_0_bias',torch.tensor([73.]))
    P.rewrite(m,POLICY,columns_per_scale=2)
    assert m._merlin_shared_projection_0_bias.tolist()==[73.]
    assert hasattr(m,'_merlin_shared_projection_1_weight')

def readout_cost_guard(kind):
    m,_,x=fixture()
    if kind=='uniform':m.s1.fill_(.5)
    before=run(m,x)
    limit=3 if kind=='fits' else 2
    r=P.rewrite(m,{**POLICY,'max_nonuniform_readout_blocks':limit},columns_per_scale=2)
    if kind=='over':
        assert not r['groups'] and count(m)==2
        row=next(row for row in r['refused'] if 'physical block budget' in row['reason'])
        assert row['physical_readout_blocks']==3 and row['max_nonuniform_readout_blocks']==2
    else:assert count(m)==1
    equal(before,run(m,x))

def narrow_dq_does_not_clamp_raw_codes():
    m,sites,_=fixture()
    m.register_buffer('raw_codes',torch.full((3,1033),-128,dtype=torch.int8))
    lhs=sites[0][0].args[0]
    with m.graph.inserting_before(lhs):raw=m.graph.get_attr('raw_codes')
    lhs.args=(raw,1.,0,-127,127,torch.int8)
    for i,n in enumerate((2,4)):
        setattr(m,'w'+str(i),torch.full((n,1033),127,dtype=torch.int8))
    assert 127*(1033*127)<2**24<128*(1033*127)
    m.graph.lint();m.recompile()
    r=P.rewrite(m,POLICY,columns_per_scale=2)
    assert not r['groups'] and count(m)==2
    assert all('exact f32 integer carrier' in row['reason'] for row in r['refused'])

def closed_grad_context(kind):
    from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled as wrap
    m,sites,x=fixture();bg=torch.fx.Graph();arg=bg.placeholder('arg')
    target=torch.ops.aten.zero_.default if kind=='mutable' else operator.neg if kind=='unknown' else torch.ops.aten.sin.default
    v=bg.call_function(target,(arg,));bg.output((v,))
    body=torch.fx.GraphModule(torch.nn.Module(),bg)
    if kind=='captured':body.register_buffer('hidden',torch.ones(1))
    elif kind=='forward':
        import types
        body.forward=types.MethodType(lambda self,arg:(arg+1,),body)
    elif kind=='global':
        import types
        old=body.forward.__func__;globals_=dict(old.__globals__);globals_['torch']=object()
        body.forward=types.MethodType(types.FunctionType(old.__code__,globals_),body)
    elif kind=='call':body._call_impl=lambda *a,**k:None
    elif kind=='class_call':
        code={};exec('def call_wrapped(self, *args, **kwargs):\n    return self._other(self, *args, **kwargs)',code)
        type(body).__call__=code['call_wrapped']
    elif kind=='budget':
        with body.graph.inserting_before(list(body.graph.nodes)[-1]):
            for _ in range(256):body.graph.call_function(torch.ops.aten.sin.default,(arg,))
        body.recompile()
    m.add_module('body',body)
    anchor=sites[0][0].args[0].args[0]
    if kind=='inside':anchor=sites[1][0]
    with m.graph.inserting_before(anchor):
        ref=m.graph.get_attr('body')
        value=xnode(m)
        if kind in ('constant','alias'):
            value=sites[0][0].args[1].args[0]
            # Move only the original pure get_attr before this new use.
            ref.prepend(value)
            if kind=='alias':value=m.graph.call_function(torch.ops.aten.view.default,(value,[-1]))
        call=m.graph.call_function(wrap,(False,ref,value))
        item=m.graph.call_function(operator.getitem,(call,True if kind=='index' else 0))
    output=list(m.graph.nodes)[-1];output.args=(tuple(output.args[0])+(item,),)
    m.graph.lint();m.recompile()
    if kind=='class_call':type(body).__call__=code['call_wrapped']
    old=copy.deepcopy(m) if kind=='readonly' else None
    r=P.rewrite(m,POLICY,columns_per_scale=2)
    if kind=='readonly':
        assert count(m)==1 and call in set(m.graph.nodes)
        equal(run(old,x),run(m,x))
    else:assert not r['groups'] and count(m)==2,kind

def causal_section_context():
    from transformers import LlamaConfig,LlamaForCausalLM
    from m2m.capture.causal_session import FixedCachePrefillStage
    import _recipe_quantizer as RQ
    recipe=json.loads((Path(sys.argv[1])/'examples/gemmini/protocol/recipes/tiny_llama-native-speed.json').read_bytes())
    c=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,
                  num_key_value_heads=2,max_position_embeddings=16,tie_word_embeddings=False)
    model=FixedCachePrefillStage(LlamaForCausalLM(c).eval(),8).eval()
    x=torch.tensor([[1,2,3,4]],dtype=torch.int64)
    q=RQ.apply_recipe(model,recipe,example_inputs=(x,),calibration_inputs=[(x,),(x+1,)])
    r=q._recipe_quantization_stats['projection_input_reuse']
    assert len(r['groups'])==2 and r['contractions_removed']==3,r
    assert q._recipe_quantization_stats['annotated_contractions']==5
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    from merlin.targetgen._capture_integer_mm import exact_integer_mm
    reference=run_pt2e_integer_reference(q,(x,))
    with exact_integer_mm(torch):actual=q(x)
    equal(reference.output,actual)
    return r

def framework_guard(kind):
    from torch.export._unlift import _convert_guards_code_to_fn
    from torch.utils._pytree import MappingKey
    m,sites,x=fixture()
    guard=_convert_guards_code_to_fn(["L['x'].size()[0] == 3"],[(MappingKey('x'),)])
    called=[]
    if kind=='mutator':
        def mutator(*args):
            called.append(True)
            args[0].zero_()
        guard.forward=torch._dynamo.dont_skip_tracing(mutator)
    elif kind=='hook':guard.register_forward_pre_hook(lambda *args:called.append(True))
    elif kind=='foreign_context':
        guard.forward=torch._dynamo.patch_dynamo_config(verbose=True)(lambda *args:None)
    elif kind=='invalid_context':
        class Trap:
            @property
            def changes(self):raise AssertionError('unsupported context must not be called')
        cells=dict(zip(guard.forward.__code__.co_freevars,guard.forward.__closure__))
        cells['self'].cell_contents.config_patch=Trap()
    elif kind=='invalid_method':
        cells=dict(zip(guard.forward.__code__.co_freevars,guard.forward.__closure__))
        type(cells['self'].cell_contents.config_patch).__enter__=None
    elif kind=='fake_carrier':
        class Fake(torch.nn.Module):
            def forward(self,*args):pass
        guard=Fake()
    m.add_module('_guards_fn',guard)
    with m.graph.inserting_before(sites[0][0]):n=m.graph.call_module('_guards_fn',(xnode(m),))
    m.recompile();r=P.rewrite(m,POLICY,columns_per_scale=2)
    assert not called
    if kind=='ordinary':assert r['contractions_removed']==1 and n in set(m.graph.nodes)
    else:assert not r['groups']

def mutation_case(kind):
    m,sites,x=fixture();first,mul,q=sites[0]
    if kind=='negative_gain':m.s0[0]=-1
    elif kind=='nonfinite_gain':m.s0[0]=float('inf')
    elif kind=='fractional_bias':m.b0[0]=.5
    elif kind=='bias_overflow':m.b0[0]=2**25
    elif kind=='bias_i64_overflow':m.b0[0]=1e38
    elif kind=='bias_nonfinite':m.b0[0]=float('inf')
    elif kind=='wrong_dtype':m.w0=m.w0.to(torch.int16)
    elif kind=='unobserved_geometry':first.meta.clear()
    elif kind=='unobserved_input_geometry':xnode(m).meta.clear()
    elif kind=='wrong_input_geometry':xnode(m).meta['val']=torch.empty(3,4)
    elif kind=='raw_exit':q.replace_all_uses_with(mul);m.graph.erase_node(q)
    elif kind=='other_liveout':
        output=list(m.graph.nodes)[-1];output.args=(tuple(output.args[0])+(first,),)
    elif kind=='different_input_grid':
        m,sites,x=fixture(duplicate_q=True);left=sites[1][0].args[0];quant=left.args[0];quant.args=(quant.args[0],.25,*quant.args[2:])
    elif kind=='different_output_clamp':q.args=(*q.args[:3],-127,127,torch.int8)
    elif kind=='relu':
        with m.graph.inserting_before(mul):r=m.graph.call_function(torch.ops.aten.relu.default,(first,))
        mul.replace_input_with(first,r)
    elif kind=='mutable_span':
        with m.graph.inserting_before(sites[1][0]):m.graph.call_function(torch.ops.aten.add_.Tensor,(xnode(m),0))
    elif kind=='random_span':
        with m.graph.inserting_before(sites[1][0]):m.graph.call_function(torch.ops.aten.rand.default,([1],))
    elif kind=='unknown_span':
        with m.graph.inserting_before(sites[1][0]):m.graph.call_function(operator.neg,(xnode(m),))
    elif kind=='input_mutation_before_first':
        m,sites,x=fixture(duplicate_q=True)
        with m.graph.inserting_before(sites[0][0]):m.graph.call_function(torch.ops.aten.add_.Tensor,(xnode(m),1))
    elif kind=='constant_mutation':
        weight=first.args[1].args[0]
        with m.graph.inserting_before(first):m.graph.call_function(torch.ops.aten.zero_.default,(weight,))
    elif kind=='constant_alias_mutation':
        weight=first.args[1].args[0]
        with m.graph.inserting_before(first):
            alias=m.graph.call_function(torch.ops.aten.view.default,(weight,[-1]))
            m.graph.call_function(torch.ops.aten.zero_.default,(alias,))
    elif kind=='constant_escape':
        output=list(m.graph.nodes)[-1];output.args=(tuple(output.args[0])+(first.args[1].args[0],),)
    elif kind=='unknown_after_span':
        with m.graph.inserting_before(list(m.graph.nodes)[-1]):m.graph.call_function(operator.neg,(xnode(m),))
    else:raise AssertionError(kind)
    m.graph.lint();m.recompile();r=P.rewrite(m,POLICY,columns_per_scale=2);assert not r['groups'] and count(m)==2,kind

def xnode(model):return next(n for n in model.graph.nodes if n.op=='placeholder')

def integer_reference_and_lowering():
    sys.path.insert(0,sys.argv[2])
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    from m2m.capture.pt2e_integerize import integerize_pt2e
    from merlin.targetgen._capture_integer_mm import exact_integer_mm
    m,_,x=fixture();original=run_pt2e_integer_reference(m,(x,));r=P.rewrite(m,POLICY,columns_per_scale=2)
    reference=run_pt2e_integer_reference(m,(x,));equal(original.output,reference.output)
    assert original.contraction_count==2 and reference.contraction_count==1
    m,receipt=integerize_pt2e(m,(x,));assert receipt['integer_mm_emitted']==1
    with torch.no_grad(),exact_integer_mm(torch):equal(reference.output,m(x))
    import m2m
    result=m2m.convert(m,(x,),backend='fx_importer',level='linalg-on-tensors')
    assert result.ok,result.diagnostics
    assert result.mlir_text.count('linalg.generic')>0
    import os
    artifact=Path(os.environ['TMPDIR'])/'projection-fixture.mlir'
    artifact.write_text(result.mlir_text)
    return {'reference_original':2,'reference_packed':1,'integer_mm_emitted':receipt['integer_mm_emitted'],'lowered':True}

def ordinary_quantizer():
    sys.path.insert(0,str(Path(sys.argv[1])/'src/merlin/targetgen'))
    sys.path.insert(0,sys.argv[2])
    import _recipe_quantizer as RQ
    from merlin.targetgen import quant_recipe as QR, quant_methodology as QM
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.left=torch.nn.Linear(5,4);self.right=torch.nn.Linear(5,6)
            self.tail1=torch.nn.Linear(4,3);self.tail2=torch.nn.Linear(6,2)
        def forward(self,x):return self.tail1(self.left(x)),self.tail2(self.right(x))
    path={'schema':'output_path_v1','accumulator':{'dtype':'i32'},
      'accumulator_stages':[{'kind':'preload','dtype':'i32','range':[-2**31,2**31-1],'rounding':'none'}],
      'readout':[{'kind':'convert','to':'f32','rounding':'half_even'},
       {'kind':'scale','dtype':'f32','rounding':'half_even','granularity':'per_command','columns_per_scale':2},
       {'kind':'requant','to':'i8','rounding':'half_even','clamp':[-128,127]}],
      'exits':{'accumulator':True,'requant':True}}
    methodology={'scale_granularity':'from_output_path','bias':'accumulator_preload_int32',
      'epilogue':'integer_domain','rounding':'from_output_path','clamp':'from_output_path',
      'calibration':{'activation':'mse','weight':'minmax'},
      'recovery':{'cross_layer_equalization':False,'bias_correction':False},
      'float_island':{'dtype':'fp32','ops':['softmax']}}
    base=QR.QuantRecipe(target='synthetic',unit='mesh',weight=QR.TensorSpec('int8','tensor',True,-127,127),
      activation=QR.TensorSpec('int8','tensor',True,-128,127),families=(QR.CONTRACTION,)).to_dict()|{'accumulator_dtype':'int32'}
    recipe=QM.derive(base,methodology,path)
    x=torch.randn(3,5)
    model=RQ.apply_recipe(Net().eval(),recipe,example_inputs=(x,),calibration_inputs=[(x,)],calibration_samples=1)
    before=run_pt2e_integer_reference(model,(x,));r=P.rewrite(model,POLICY,columns_per_scale=2)
    assert r['contractions_removed']==1,r
    after=run_pt2e_integer_reference(model,(x,));equal(before.output,after.output)
    assert before.contraction_count==4 and after.contraction_count==3
    return r

cases={'exact_grids':exact_grids,'all_i8_and_ties':all_i8_and_ties,'output_allocations':output_allocations,
 'duplicate_inputs':duplicate_inputs,'distinct_inputs':distinct_inputs,'odd_widths':odd_widths,
 'incompatible_physical_tail':incompatible_physical_tail,'compatible_physical_tail':compatible_physical_tail,
 'integer_reference_and_lowering':integer_reference_and_lowering,'ordinary_quantizer':ordinary_quantizer,
 'existing_attribute':existing_attribute,'narrow_dq_does_not_clamp_raw_codes':narrow_dq_does_not_clamp_raw_codes}
for k in ['negative_gain','nonfinite_gain','fractional_bias','bias_overflow','bias_i64_overflow','bias_nonfinite','wrong_dtype','unobserved_geometry',
 'raw_exit','other_liveout','different_input_grid','different_output_clamp','relu','mutable_span','random_span','unknown_span',
 'unobserved_input_geometry','wrong_input_geometry','input_mutation_before_first','constant_mutation',
 'constant_alias_mutation','constant_escape','unknown_after_span']:
    cases[k]=lambda k=k:mutation_case(k)
for k in ['ordinary','mutator','hook','foreign_context','fake_carrier','invalid_context','invalid_method']:
    cases['guard_'+k]=lambda k=k:framework_guard(k)
for k in ['readonly','mutable','unknown','captured','forward','global','call','class_call','budget','constant','alias','index','inside']:
    cases['context_'+k]=lambda k=k:closed_grad_context(k)
cases['causal_section_context']=causal_section_context
for k in ['fits','over','uniform']:
    cases['readout_cost_'+k]=lambda k=k:readout_cost_guard(k)
results={}
for name,fn in cases.items():
    try:results[name]={'passed':True,'detail':fn()}
    except Exception:results[name]={'passed':False,'error':traceback.format_exc()}
print('__RESULT__'+json.dumps(results))
"""


@pytest.fixture(scope="module")
def capture_results():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    m2m = os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("requires selected capture interpreter and model2MLIR source")
    result = subprocess.run(
        [python, "-B", "-c", _SCRIPT, str(repo_root()), m2m],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith("__RESULT__"))
    return json.loads(line[len("__RESULT__") :])


@pytest.mark.parametrize(
    "case",
    [
        "exact_grids",
        "all_i8_and_ties",
        "output_allocations",
        "duplicate_inputs",
        "distinct_inputs",
        "odd_widths",
        "incompatible_physical_tail",
        "compatible_physical_tail",
        "integer_reference_and_lowering",
        "ordinary_quantizer",
        "existing_attribute",
        "narrow_dq_does_not_clamp_raw_codes",
        "negative_gain",
        "nonfinite_gain",
        "fractional_bias",
        "bias_overflow",
        "bias_i64_overflow",
        "bias_nonfinite",
        "wrong_dtype",
        "unobserved_geometry",
        "raw_exit",
        "other_liveout",
        "different_input_grid",
        "different_output_clamp",
        "relu",
        "mutable_span",
        "random_span",
        "unknown_span",
        "unobserved_input_geometry",
        "wrong_input_geometry",
        "input_mutation_before_first",
        "constant_mutation",
        "constant_alias_mutation",
        "constant_escape",
        "unknown_after_span",
        "guard_ordinary",
        "guard_mutator",
        "guard_hook",
        "guard_foreign_context",
        "guard_fake_carrier",
        "guard_invalid_context",
        "guard_invalid_method",
        "context_readonly",
        "context_mutable",
        "context_unknown",
        "context_captured",
        "context_forward",
        "context_global",
        "context_call",
        "context_class_call",
        "context_budget",
        "context_constant",
        "context_alias",
        "context_index",
        "context_inside",
        "causal_section_context",
        "readout_cost_fits",
        "readout_cost_over",
        "readout_cost_uniform",
    ],
)
def test_capture_semantics_and_refusals(capture_results, case):
    assert capture_results[case]["passed"], capture_results[case].get("error")
