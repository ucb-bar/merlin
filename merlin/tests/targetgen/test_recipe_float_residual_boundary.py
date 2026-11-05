"""Opt-in integer residual exit with unchanged floating-point normalization."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import quant_recipe as QR
from merlin.targetgen import _integer_operand_sum as sums
from merlin.tests.targetgen.test_recipe_shared_grid_sums import _base, _recipe


def test_boundary_policy_preserves_v1_and_requires_explicit_v2():
    base = _base()
    legacy = QR.with_shared_grid_operand_sums(base)
    expected = copy.deepcopy(base)
    expected["families"].append(QR.OPERAND_SUM)
    expected[sums.KEY] = {"schema": "shared_grid_operand_sum_v1"}
    expected["unquantized"] = {}
    expected["why"] = {QR.OPERAND_SUM: (
        "opted in: each closed residual's inputs and output share one observed activation grid; "
        "unit i8 load/readout multipliers preserve exact integer add, optional ReLU and saturation"
    )}
    expected["recipe_sha256"] = QR.digest(expected)
    assert legacy == expected
    narrowed = QR.with_shared_grid_operand_sums(base, output_boundary="dequantize")
    assert narrowed[sums.KEY] == {"schema": sums.FLOAT_BOUNDARY_SCHEMA, "output_boundary": "dequantize"}
    assert narrowed["recipe_sha256"] != legacy["recipe_sha256"]
    assert QR.with_shared_grid_operand_sums(narrowed, output_boundary="dequantize") == narrowed
    assert sums.KEY not in base
    with pytest.raises(ValueError, match="output boundary"):
        QR.with_shared_grid_operand_sums(base, output_boundary="integer_normalization")
    for bad in ({"schema": sums.FLOAT_BOUNDARY_SCHEMA},
                {"schema": sums.FLOAT_BOUNDARY_SCHEMA, "output_boundary": "integer"},
                {"schema": sums.SCHEMA, "output_boundary": "dequantize"}):
        with pytest.raises(sums.OperandSumError):
            sums.validate(narrowed | {sums.KEY: bad})


_SCRIPT = r'''
import copy, json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import torch, torchao
import _recipe_quantizer as RQ
from merlin.targetgen.quant_recipe import with_shared_grid_operand_sums
from m2m.capture import pt2e_integer_reference as IR
from m2m.capture.pt2e_integerize import integerize_pt2e
torch.set_num_threads(1)
torch.manual_seed(19)
base=json.loads(sys.argv[3])
class Block(torch.nn.Module):
    def __init__(self, alpha=1):
        super().__init__(); self.linear=torch.nn.Linear(7,7); self.alpha=alpha
    def forward(self,x):
        residual=torch.add(self.linear(x),x,alpha=self.alpha)
        norm=residual*torch.rsqrt(residual.square().mean(-1,keepdim=True)+1e-5)
        return norm,residual
x=torch.randn(1,3,7)*2
rows={}
for mode in ('legacy','boundary','unsupported_alpha'):
    model=Block(2 if mode=='unsupported_alpha' else 1).eval()
    recipe=with_shared_grid_operand_sums(base,output_boundary='quantized_readers' if mode=='legacy' else 'dequantize')
    q=RQ.apply_recipe(model,recipe,example_inputs=(x,),calibration_inputs=[(x,),(x*2,)])
    stats=q._recipe_quantization_stats
    row={'sum':stats['shared_grid_operand_sums'],'exits':stats['integer_epilogue']['by_exit']}
    if mode=='boundary':
        qd=torch.ops.quantized_decomposed
        add=next(n for n in q.graph.nodes if n.target==torch.ops.aten.add.Tensor and isinstance(n.args[1],torch.fx.Node))
        assert all(n.target==qd.dequantize_per_tensor.default and n.args[1]==1.0 for n in add.args[:2])
        quantize=next(iter(add.users)); dq=next(iter(quantize.users))
        assert quantize.target==qd.quantize_per_tensor.default and quantize.args[1]==1.0
        assert dq.target==qd.dequantize_per_tensor.default and dq.args[1]>0
        # Extract emitted residual and float readers; supply every signed i8 pair
        # independently of the observers and preceding contraction.
        graph=torch.fx.Graph(); a=graph.placeholder('a'); b=graph.placeholder('b')
        env={add.args[0].args[0]:a,add.args[1].args[0]:b}
        needed=set()
        def visit(n):
            if n in env or n in needed:return
            for v in n.all_input_nodes:visit(v)
            needed.add(n)
        visit(next(n for n in q.graph.nodes if n.op=='output'))
        for n in q.graph.nodes:
            if n in needed:env[n]=graph.node_copy(n,lambda v:env[v])
        section=torch.fx.GraphModule(q,graph)
        a=torch.arange(-128,128,dtype=torch.int16)[:,None].expand(256,256)
        b=torch.arange(-128,128,dtype=torch.int16)[None,:].expand(256,256)
        actual=section(a.to(torch.int8),b.to(torch.int8))
        residual=(a+b).clamp(-128,127).float()*dq.args[1]
        normalized=residual*torch.rsqrt(residual.square().mean(-1,keepdim=True)+1e-5)
        row['pairs_checked']=a.numel()
        row['float_boundary_exact']=bool(torch.equal(actual[0],normalized) and torch.equal(actual[1],residual))
        reference=IR.run_pt2e_integer_reference(q,(x,))
        integer,receipt=integerize_pt2e(copy.deepcopy(q),(x,))
        row['integerized_exact']=all(torch.equal(a,b) for a,b in zip(reference.output,integer(x)))
        row['float_rsqrt']=any(n.target==torch.ops.aten.rsqrt.default for n in q.graph.nodes)
    rows[mode]=row
print(json.dumps(rows))
'''


def test_actual_float_rms_boundary_and_independent_integer_reference():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    root = repo_root()
    result = subprocess.run(
        [python, "-B", "-c", _SCRIPT, str(root / "src/merlin/targetgen"),
         str(root / "src"), json.dumps(_recipe())], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    assert rows["legacy"]["sum"]["rewritten"] == 0
    assert rows["unsupported_alpha"]["sum"]["rewritten"] == 0
    actual = rows["boundary"]
    assert actual["sum"]["rewritten"] == 1
    assert actual["sum"]["target_execution_qualified"] is False
    assert actual["exits"]["requant"] == 1
    assert actual["pairs_checked"] == 65536
    assert actual["float_boundary_exact"] and actual["integerized_exact"] and actual["float_rsqrt"]


def test_v2_disjoint_attention_policy_preserved_and_overlaps_refused():
    base = QR.with_dynamic_contractions(
        _recipe(), scales=QR.STATIC_HEADS,
        planes={QR.SOFTMAX_OUTPUT: 2, QR.SOFTMAX_PARTNER: 2},
    )
    base = QR.with_site_agreement(base, cosine_min=0.9, rel_rmse_max=0.5, basis="public test")
    before = copy.deepcopy(base)
    combined = QR.with_shared_grid_operand_sums(base, output_boundary="dequantize")
    assert base == before
    assert combined["dynamic"] == base["dynamic"]
    with pytest.raises(sums.OperandSumError):
        QR.with_shared_grid_operand_sums(base)  # V1 restrictions unchanged.
    for change in (
        {"families": ["operand_sum"]}, {"families": ["contraction"]},
        {"families": ["activation_contraction", "operand_sum"]},
        {"scales": "non_reduction"}, {"planes": {"weight": 2}},
        {"planes": {"softmax_partner": 4}}, {"planes": {"unknown": 2}},
        {"planes": {"softmax_partner": True}}, {"schema": "unknown"},
        {"unknown": True}, {"smooth": "unknown"},
    ):
        bad = copy.deepcopy(combined)
        bad["dynamic"].update(change)
        with pytest.raises(sums.OperandSumError):
            sums.validate(bad)
    with pytest.raises(sums.OperandSumError):
        sums.validate(combined | {"static_planes": {"planes": 2}})


_MIXED_SCRIPT = r'''
import copy, json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import torch, torchao
import _recipe_quantizer as RQ
from merlin.targetgen import quant_recipe as QR
from m2m.capture import pt2e_integer_reference as IR
from m2m.capture.pt2e_integerize import integerize_pt2e
torch.set_num_threads(1); torch.manual_seed(29)
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear=torch.nn.Linear(32,32)
    def forward(self,x):
        p=torch.softmax(torch.matmul(x,x.transpose(-1,-2))*0.125,dim=-1)
        a=torch.matmul(p,x)
        residual=self.linear(a)+x
        return residual*torch.rsqrt(residual.square().mean(-1,keepdim=True)+1e-5)
x=torch.randn(1,17,32)
base=QR.with_dynamic_contractions(json.loads(sys.argv[3]),scales=QR.STATIC_HEADS,
    planes={QR.SOFTMAX_OUTPUT:2,QR.SOFTMAX_PARTNER:2})
recipe=QR.with_shared_grid_operand_sums(base,output_boundary='dequantize',
    residual_selection='distinct_contraction_frontiers')
before=copy.deepcopy(recipe)
q=RQ.apply_recipe(Block().eval(),recipe,example_inputs=(x,),calibration_inputs=[(x,),(x*2,)])
assert recipe==before
stats=q._recipe_quantization_stats
assert stats['shared_grid_operand_sums']['rewritten']==1, stats
assert stats['integer_epilogue']['by_exit']['requant']==1, stats
assert stats['dynamic_contractions']['rewritten']==2, stats
reference=IR.run_pt2e_integer_reference(q,(x,))
integer,receipt=integerize_pt2e(copy.deepcopy(q),(x,))
assert torch.equal(reference.output,integer(x))
assert any(n.target==torch.ops.aten.rsqrt.default for n in q.graph.nodes)
print('mixed attention and residual exact')
'''


def test_real_static_head_attention_composes_with_float_residual_exit():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    root = repo_root()
    result = subprocess.run(
        [python, "-B", "-c", _MIXED_SCRIPT, str(root / "src/merlin/targetgen"),
         str(root / "src"), json.dumps(_recipe())], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    assert "mixed attention and residual exact" in result.stdout


def test_distinct_frontier_selection_is_a_new_explicit_recipe():
    broad = QR.with_shared_grid_operand_sums(_recipe(), output_boundary="dequantize")
    before = copy.deepcopy(broad)
    selected = QR.with_shared_grid_operand_sums(
        broad, output_boundary="dequantize", residual_selection=sums.DISTINCT_FRONTIERS,
    )
    assert broad == before
    assert selected[sums.KEY]["selection"] == sums.DISTINCT_FRONTIERS
    assert selected["recipe_sha256"] != broad["recipe_sha256"]
    assert QR.with_shared_grid_operand_sums(
        selected, output_boundary="dequantize", residual_selection=sums.DISTINCT_FRONTIERS,
    ) == selected
    with pytest.raises(ValueError, match="residual selection"):
        QR.with_shared_grid_operand_sums(_base(), residual_selection=sums.DISTINCT_FRONTIERS)
    with pytest.raises(ValueError, match="residual selection"):
        QR.with_shared_grid_operand_sums(_base(), output_boundary="dequantize", residual_selection="model_names")


_FRONTIER_SCRIPT = r'''
import copy, json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import torch, torchao
import _recipe_quantizer as RQ
from merlin.targetgen import quant_recipe as QR
from m2m.capture import pt2e_integer_reference as IR
from m2m.capture.pt2e_integerize import integerize_pt2e
torch.set_num_threads(1); torch.manual_seed(31)
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first=torch.nn.Linear(8,8); self.second=torch.nn.Linear(8,8)
        self.third=torch.nn.Linear(8,8)
        self.register_buffer('c',torch.arange(8).float().cos())
        self.register_buffer('s',torch.arange(8).float().sin())
    def forward(self,x):
        p=self.first(x)
        coordinates=p*self.c+torch.cat((-p[...,4:],p[...,:4]),-1)*self.s
        residual=self.second(coordinates)+x
        norm=residual*torch.rsqrt(residual.square().mean(-1,keepdim=True)+1e-5)
        next_residual=self.third(norm)*7+residual
        return next_residual,coordinates
x=torch.randn(1,3,8)
model=Block().eval()
rows={}
for mode in ('broad','distinct'):
    kwargs={'residual_selection':'distinct_contraction_frontiers'} if mode=='distinct' else {}
    recipe=QR.with_shared_grid_operand_sums(json.loads(sys.argv[3]),output_boundary='dequantize',**kwargs)
    q=RQ.apply_recipe(copy.deepcopy(model),recipe,example_inputs=(x,),calibration_inputs=[(x,),(x*2,)])
    stats=q._recipe_quantization_stats
    rows[mode]=stats['shared_grid_operand_sums']['rewritten']
    if mode=='distinct':
        qd=torch.ops.quantized_decomposed
        scales=[]
        for n in q.graph.nodes:
            if n.target!=torch.ops.aten.add.Tensor or not all(isinstance(a,torch.fx.Node) for a in n.args):continue
            if not all(a.target==qd.dequantize_per_tensor.default and a.args[1]==1.0 for a in n.args):continue
            quantize=next(iter(n.users));dq=next(iter(quantize.users));scales.append(dq.args[1])
        # Existing PT2E implicit sharing ties this consecutive residual pair.
        # The source-selection ablation intentionally preserves that behavior.
        assert len(scales)==2 and len(set(scales))==1, scales
        reference=IR.run_pt2e_integer_reference(q,(x,))
        integer,receipt=integerize_pt2e(copy.deepcopy(q),(x,))
        assert all(torch.equal(a,b) for a,b in zip(reference.output,integer(x)))
# Source filter also permits independent projections with the same original
# input, while rejecting a repeated input without a contraction frontier.
class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__();self.a=torch.nn.Linear(8,8);self.b=torch.nn.Linear(8,8)
    def forward(self,x):return self.a(x)+self.b(x),x+x
from _integer_operand_sum import SharedGridSums
src=torch.export.export(Pair().eval(),(x,)).module()
selector=SharedGridSums(recipe)
selector.select([n for n in src.graph.nodes if n.target==torch.ops.aten.add.Tensor])
assert len(selector.selected)==1
assert rows=={'broad':3,'distinct':2},rows
print('coordinate sum retained; existing shared residual grid; exact integerized outputs')
'''


def test_real_coordinate_sum_exclusion_and_residual_chain_scales():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    root = repo_root()
    result = subprocess.run(
        [python, "-B", "-c", _FRONTIER_SCRIPT, str(root / "src/merlin/targetgen"),
         str(root / "src"), json.dumps(_recipe())], capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    assert "existing shared residual grid" in result.stdout
