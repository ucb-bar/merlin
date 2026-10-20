"""Selected accumulator residuals have one bounded integer contraction readout."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import _integer_residual as R


def policy(mode="exact", **limits):
    return {
        "schema": R.SCHEMA,
        "coefficient_rounding": mode,
        "max_terms": 3,
        "max_weight_elements": 10000,
        "max_mac_ratio": 10,
        **limits,
    }


def plan(mode="exact", **kw):
    return R.derive_plan(
        activation_scale=0.5,
        weight_scales=[0.25] * 7,
        skip_scale=0.5,
        output_scale=0.5,
        input_channels=5,
        component_width=3,
        weight_max=127,
        block=policy(mode, **kw),
    )


def test_component_tail_and_cost_are_explicit():
    p = plan()
    assert p["coefficients"] == [4] * 7
    assert [(c["start"], c["stop"]) for c in p["components"]] == [(0, 3), (3, 6), (6, 7)]
    assert p["original_weight_elements"] == 35 and p["augmented_weight_elements"] == 54


def _coalesced(*, inputs=64, channels=256, width=16, **limits):
    block = R.check(policy("align_weight_grid", component_packing="coalesced", **limits))
    return R._components([1.0] * channels, [7] * channels, [0.0] * channels, 1.0, inputs, width, 127, block)


def test_explicit_coalescing_trades_fewer_calls_for_more_bounded_macs():
    selected = _coalesced(max_mac_ratio=2, max_weight_elements=32768)
    assert selected["physical_columns_per_scale"] == 16
    assert selected["component_columns_limit"] == 64
    assert len(selected["components"]) == 4
    assert selected["mac_ratio"] == 2
    assert selected["augmented_weight_elements"] == 32768
    physical = R._components(
        [1.0] * 256,
        [7] * 256,
        [0.0] * 256,
        1.0,
        64,
        16,
        127,
        R.check(policy("align_weight_grid", max_mac_ratio=2, max_weight_elements=32768)),
    )
    assert len(physical["components"]) == 16 and physical["mac_ratio"] == 1.25


def test_coalesced_tail_and_weight_budget_are_independent():
    selected = _coalesced(inputs=5, channels=7, width=3, max_mac_ratio=2.1)
    assert [(p["start"], p["stop"]) for p in selected["components"]] == [(0, 6), (6, 7)]
    assert selected["augmented_weight_elements"] == 72
    bounded = _coalesced(inputs=5, channels=7, width=3, max_mac_ratio=10, max_weight_elements=60)
    assert len(bounded["components"]) == 3 and bounded["augmented_weight_elements"] == 54
    with pytest.raises(R.ResidualError, match="budget"):
        _coalesced(inputs=5, channels=7, width=3, max_mac_ratio=1.1)


def test_coalescing_cannot_silently_change_a_fixed_grid_or_unknown_policy():
    with pytest.raises(R.ResidualError, match="aligned"):
        R.check(policy("exact", component_packing="coalesced"))
    with pytest.raises(R.ResidualError, match="packing"):
        R.check(policy("align_weight_grid", component_packing="guessed"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_terms", 0),
        ("max_terms", True),
        ("max_mac_ratio", float("inf")),
        ("max_mac_ratio", 0.5),
        ("max_weight_elements", 0),
    ],
)
def test_policy_rejects_unknown_or_unbounded_work(field, value):
    with pytest.raises(R.ResidualError):
        R.check(policy(**{field: value}))


def test_exact_vs_explicit_half_even_and_nonzero_residual():
    args = dict(
        activation_scale=1,
        weight_scales=[1],
        skip_scale=2.5,
        output_scale=1,
        input_channels=2,
        component_width=1,
        weight_max=127,
    )
    with pytest.raises(R.ResidualError, match="integer multiple"):
        R.derive_plan(**args, block=policy())
    assert R.derive_plan(**args, block=policy("half_even"))["coefficients"] == [2]
    with pytest.raises(R.ResidualError, match="drop"):
        R.derive_plan(**(args | {"skip_scale": 0.25}), block=policy("half_even"))


@pytest.mark.parametrize("kw", [{"max_mac_ratio": 1.1}, {"max_weight_elements": 36}])
def test_expansion_requires_both_selected_budgets(kw):
    with pytest.raises(R.ResidualError, match="budget"):
        plan(**kw)


SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch, torchao
import _integer_residual as R
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(23)
mode = sys.argv[2]
policy = dict(schema=R.SCHEMA,coefficient_rounding='align_weight_grid' if mode.startswith('align') else 'exact',
              max_terms=2,max_weight_elements=10000,max_mac_ratio=10)
if mode=='align_coalesced':policy['component_packing']='coalesced'
A,Q=torch.ops.aten,torch.ops.quantized_decomposed
g=torch.fx.Graph(); root=torch.nn.Module()
conv=mode in ('conv','conv_pool')
shape=(2,7,2,3) if conv else (3,7)
xshape=(2,5,2,3) if conv else (3,5)
root.register_buffer('weight',torch.randint(-3,4,(7,5,1,1) if conv else (7,5),dtype=torch.int8))
root.register_buffer('bias',torch.arange(7,dtype=torch.int32)-3)
x=g.placeholder('x'); x.meta['val']=torch.empty(xshape,dtype=torch.int8)
r=g.placeholder('r'); r.meta['val']=torch.empty(shape,dtype=torch.int8)
def dq(v,s,dtype=torch.int8):
    low,high=(-128,127) if dtype==torch.int8 else (-2**31,2**31-1)
    return g.call_function(Q.dequantize_per_tensor.default,(v,s,0,low,high,dtype))
xd=dq(x,.5); xd.meta['val']=torch.empty(xshape)
rd=dq(r,.5); rd.meta['val']=torch.empty(shape)
wd=dq(g.get_attr('weight'),.25); wd.meta['val']=torch.empty(root.weight.shape)
bd=dq(g.get_attr('bias'),.125,torch.int32)
n=g.call_function(A.conv2d.default if conv else A.linear.default,(xd,wd,bd)); n.meta['val']=torch.empty(shape)
add=g.call_function(A.add_.Tensor if mode=='inplace' else A.add.Tensor,(n,rd)); add.meta['val']=torch.empty(shape)
relu=g.call_function(A.relu_.default if mode=='inplace' else A.relu.default,(add,)); relu.meta['val']=torch.empty(shape)
q=g.call_function(Q.quantize_per_tensor.default,(relu,.5,0,-128,127,torch.int8))
out=dq(q,.5);g.output(out)
gm=torch.fx.GraphModule(root,g)
helper=R.AccumulatorResidual(policy)
assert helper.pattern(n)==[add,relu]
if mode=='two_branches':
    with g.inserting_before(add):
        other=g.call_function(A.linear.default,(xd,wd,bd));other.meta['val']=torch.empty(shape)
    add.args=(n,other)
    assert helper.pattern(other)==[] and helper.pattern(n)==[add,relu]
    assert helper.members[add.name]==n.name and len(helper.selected)==1
    helper.discard(n)
    assert helper.pattern(other)==[add,relu] and helper.pattern(n)==[]
    helper.close_selection()
    assert helper.pattern(other)==[add,relu]
    helper.discard(other)
    assert helper.pattern(n)==helper.pattern(other)==[] and not helper.members
    print(json.dumps({'matched':True}));sys.exit(0)
ep=dict(scale=dict(columns_per_scale=3),requant=dict(clamp=[-128,127]),activation=dict(domain='accumulator'))
if mode=='late_skip':add.prepend(rd)
elif mode=='fanout':
    with g.inserting_before(q): g.call_function(A.neg.default,(n,))
elif mode=='shape': r.meta['val']=torch.empty((4,7),dtype=torch.int8)
elif mode=='zero': rd.args=(r,.5,1,-128,127,torch.int8)
elif mode=='range': wd.args=(wd.args[0],.25,0,-128,120,torch.int8)
elif mode=='alpha': add.kwargs={'alpha':2}
elif mode=='bias': bd.args=(*bd.args[:-1],torch.int8)
elif mode=='mutation':
    with g.inserting_before(add):g.call_function(A.zero_.default,(xd,))
elif mode in ('align','align_coalesced'):
    helper.frozen[n.name]=(root.weight.float()*.25,root.bias.float()*.125)
if mode in ('fanout','shape','zero','range','alpha','bias','align_missing','mutation'):
    try: helper.rewrite_site(gm,n,relu,{},ep)
    except R.ResidualError: print(json.dumps({'refused':True}));sys.exit(0)
    raise AssertionError('mutation accepted')
row=helper.rewrite_site(gm,n,relu,{},ep)
xx=torch.randint(-128,128,xshape,dtype=torch.int8); rr=torch.randint(-128,128,shape,dtype=torch.int8)
actual=gm(xx,rr)
if mode in ('align','align_coalesced'):
    packed=getattr(gm,'_merlin_residual_0_0_weight')
    w=torch.cat([getattr(gm,f'_merlin_residual_0_{i}_weight')[:,:5].reshape(-1,5)
                 for i in range(len(row['components']))])
    b=torch.round(root.bias.double()*.125/row['accumulator_steps'][0]).long()
else: w=root.weight.reshape(7,5).long();b=root.bias.long()
if conv:
    acc=torch.einsum('nkhw,jk->njhw',xx.long(),w.long())+b.reshape(1,7,1,1)
    acc+=torch.tensor(row['coefficients']).reshape(1,7,1,1)*rr.long()
    gain=torch.tensor(row['readout_gains']).reshape(1,7,1,1)
else:
    acc=xx.long()@w.long().T+b+rr.long()*torch.tensor(row['coefficients'])
    gain=torch.tensor(row['readout_gains'])
expected=torch.clamp(torch.round(torch.relu(acc).float()*gain),-128,127).to(torch.int8).float()*.5
assert torch.equal(actual,expected),(actual,expected)
if mode=='conv_pool':
    import _integer_pool as P
    output=next(v for v in gm.graph.nodes if v.op=='output')
    with gm.graph.inserting_before(output):
        mean=gm.graph.call_function(A.mean.dim,(out,[2,3],False));mean.meta['val']=torch.empty((2,7))
        mq=gm.graph.call_function(Q.quantize_per_tensor.default,(mean,.5,0,-128,127,torch.int8))
        mdq=gm.graph.call_function(Q.dequantize_per_tensor.default,(mq,.5,0,-128,127,torch.int8))
    output.args=(mdq,)
    report=P.rewrite(gm,dict(schema=P.SCHEMA,max_window_elements=6,max_weight_elements=6,max_macs=84),epilogue=ep)
    integer_codes=(expected/.5).long()
    pool_expected=torch.round(integer_codes.sum((2,3)).float()*(1/6)).clamp(-128,127).to(torch.int8).float()*.5
    assert torch.equal(gm(xx,rr),pool_expected) and report['new_contractions']==1
contractions=[v for v in gm.graph.nodes if v.target in (A.linear.default,A.conv2d.default)]
assert len(contractions)==len(row['components'])+(1 if mode=='conv_pool' else 0)
assert row['augmented_weight_elements']==(84 if mode=='align_coalesced' else 54)
assert all(not v.target in (A.add.Tensor,A.add_.Tensor) for v in gm.graph.nodes)
assert all(v.target!=A.relu_.default for v in gm.graph.nodes)
if mode=='lowering':
    sys.path.insert(0,sys.argv[3])
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    from m2m.capture.pt2e_integerize import integerize_pt2e
    reference=run_pt2e_integer_reference(gm,(xx,rr))
    assert torch.equal(reference.output,expected)
    gm,receipt=integerize_pt2e(gm,(xx,rr))
    assert receipt['integer_mm_emitted']==3
    import m2m
    result=m2m.convert(gm,(xx,rr),backend='fx_importer',level='linalg-on-tensors')
    assert result.ok,result.diagnostics
    from pathlib import Path
    Path(sys.argv[4]).write_text(result.mlir_text)
    weight_args=[i for i,s in enumerate(result.exported_program.graph_signature.input_specs)
                 if s.kind.name in ('BUFFER','PARAMETER','CONSTANT_TENSOR')]
    Path(sys.argv[4]+'.weights.json').write_text(json.dumps(weight_args))
print(json.dumps({'matched':True,'row':row}))
"""


@pytest.mark.parametrize(
    "mode",
    [
        "linear",
        "inplace",
        "late_skip",
        "two_branches",
        "conv",
        "conv_pool",
        "align",
        "align_coalesced",
        "align_missing",
        "fanout",
        "shape",
        "zero",
        "range",
        "alpha",
        "bias",
        "mutation",
    ],
)
def test_actual_fx_integer_readout_or_closed_refusal(mode):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("selected capture interpreter required")
    result = subprocess.run(
        [python, "-B", "-c", SCRIPT, str(repo_root() / "src/merlin/targetgen"), mode],
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert row.get("matched") or row.get("refused")


def test_wide_exit_releases_all_residual_ownership():
    from types import SimpleNamespace

    helper = R.AccumulatorResidual(policy())
    helper.selected = {"a": {}, "b": {}}
    helper.members = {"sum_a": "a", "relu_a": "a", "sum_b": "b"}
    helper.frozen = {"a": (), "b": ()}
    helper.discard(SimpleNamespace(name="a"))
    assert helper.selected == {"b": {}} and helper.members == {"sum_b": "b"}
    assert helper.frozen == {"b": ()}


def test_missing_selected_node_is_a_closed_residual_refusal():
    helper = R.AccumulatorResidual(policy())
    helper.selected["selected"] = {"last": "last"}
    with pytest.raises(R.ResidualError, match="missing"):
        helper.rewrite_site(None, None, None, {"node": "selected"}, {})
    assert helper.rewrite_site(None, None, None, {"node": "ordinary"}, {}) is None
