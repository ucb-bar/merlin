"""Prospective residual arithmetic: one grid, exact sums, unchanged legacy recipes."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import _integer_operand_sum as sums
from merlin.targetgen import quant_recipe as QR


def _base() -> dict:
    return {
        "schema": QR.SCHEMA,
        "status": QR.DERIVED,
        "accumulator_dtype": "int32",
        "families": [QR.CONTRACTION],
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
        },
    }


def test_variant_is_explicit_content_bound_and_does_not_mutate_original():
    base = _base()
    before = copy.deepcopy(base)
    variant = QR.with_shared_grid_operand_sums(base)
    assert base == before and sums.KEY not in base
    assert variant["families"] == [QR.CONTRACTION, QR.OPERAND_SUM]
    assert variant["recipe_sha256"] == QR.digest(variant) != QR.digest(base)
    assert QR.with_shared_grid_operand_sums(variant) == variant


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "uint8"},
        {"dtype": "int16"},
        {"mode": "dynamic"},
        {"granularity": "channel"},
        {"symmetric": False},
        {"quant_min": -127},
        {"quant_min": True},
        {"quant_max": 255},
    ],
)
def test_variant_rejects_other_activation_contracts(change):
    recipe = _base()
    recipe["activation"].update(change)
    with pytest.raises(sums.OperandSumError):
        QR.with_shared_grid_operand_sums(recipe)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"status": "underivable"},
        {"accumulator_dtype": "int64"},
        {"dynamic": {"families": ["contraction"]}},
        {"static_planes": {"planes": 2}},
    ],
)
def test_variant_rejects_incompatible_recipe(change):
    with pytest.raises(sums.OperandSumError):
        QR.with_shared_grid_operand_sums(_base() | change)


def test_policy_fields_are_closed():
    recipe = QR.with_shared_grid_operand_sums(_base())
    recipe[sums.KEY]["rounding"] = "truncate"
    with pytest.raises(sums.OperandSumError):
        sums.validate(recipe)


@pytest.mark.parametrize("relu", [False, True])
def test_existing_native_unit_load_arithmetic_is_exact_for_every_i8_pair(relu):
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

    facet = {
        "operand_sum": {
            "operands": 2,
            "operand_dtype": "i8",
            "operand_rounding": "half_even",
            "operand_saturates": True,
            "scale_dtype": "f32",
        },
        "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu"]}],
        "scale": {"dtype": "f32", "granularities": ["tensor"]},
    }
    result = audit_i8_operand_sum(lhs_scale=1, rhs_scale=1, bound_lsb=0, relu=relu, facet=facet)
    assert result["status"] == "within_bound" and result["max_error_lsb"] == 0
    assert result["pairs_checked"] == 65536 and result["n_over_bound"] == 0


def test_unit_grid_uses_the_existing_generic_operand_sum_route():
    from merlin.tests.ir.test_compute_groups import _groups, _residual, _summing
    from merlin.xdsl_dialects.lowering import group_command

    (group,) = _groups(_residual(lhs_scale=1.0, rhs_scale=1.0, out_scale=1.0), _summing())
    program = group_command.program(group)
    assert group.placement != "host" and group.operand_sum["readout_factor"] == 1.0
    assert program.entry["op"] == "residual_add"
    assert program.entry["lhs_scale"] == program.entry["rhs_scale"] == 1.0
    assert program.entry["epilogue"] == ["relu"] and program.stored_operand is None


_SCRIPT = r"""
import copy, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import torch
import torchao
import _recipe_quantizer as RQ
import _integer_operand_sum as S
from merlin.targetgen.quant_recipe import with_shared_grid_operand_sums
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(3)
recipe = json.loads(sys.argv[3])
class Block(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.first = torch.nn.Linear(8,8)
        self.reader = torch.nn.Linear(8,4)
        self.mode = mode
    def forward(self,x):
        a = self.first(x)
        if self.mode == 'alpha':
            a = torch.add(a,x,alpha=2)
        elif self.mode == 'inplace':
            a += x
        else:
            a = a+x
        y = self.reader(a if self.mode == 'norelu' else torch.relu(a))
        return (y, a.sin()) if self.mode == 'fanout' else y

qd=torch.ops.quantized_decomposed
results={}
captured=[]
original_rewrite=S.SharedGridSums.rewrite
def observe_rewrite(self,gm):
    captured.append((copy.deepcopy(gm),copy.deepcopy(self.sites)))
    return original_rewrite(self,gm)
S.SharedGridSums.rewrite=observe_rewrite
for mode in ('plain','inplace','norelu','alpha','fanout'):
    model=Block(mode).eval()
    x=torch.randn(2,8)
    chosen=with_shared_grid_operand_sums(recipe)
    q=RQ.apply_recipe(model,chosen,example_inputs=(x,),calibration_inputs=[(x,),(x*2,)])
    stats=q._recipe_quantization_stats
    rows=stats['shared_grid_operand_sums']
    nodes=list(q.graph.nodes)
    add=next(n for n in nodes if n.target in (torch.ops.aten.add.Tensor,torch.ops.aten.add_.Tensor))
    results[mode]={'stats':rows,'exits':stats['integer_epilogue']['by_exit'],
       'left_float':stats['sums_and_means_left_in_float'],
       'sum_input_scales':[a.args[1] if a.target==qd.dequantize_per_tensor.default else None for a in add.args[:2]]}
    if mode == 'plain':
        from m2m.capture.pt2e_integerize import integerize_pt2e
        from m2m.capture import pt2e_integer_reference as IR
        reference = IR.run_pt2e_integer_reference(q,(x,))
        integer,receipt = integerize_pt2e(copy.deepcopy(q),(x,))
        results[mode]['integerized_exact'] = bool(torch.equal(reference.output,integer(x)))
    # Recheck selected nodes on the complete signed i8 domain through actual
    # converted FX code: replacing only the already observed input Q nodes
    # supplies every ordered pair, without another model forward/reference.
    if mode in ('plain','inplace','norelu'):
        unit_inputs=list(add.args[:2])
        graph=torch.fx.Graph()
        left=graph.placeholder('left');right=graph.placeholder('right')
        env={unit_inputs[0].args[0]:left,unit_inputs[1].args[0]:right}
        tail=next(n for n in add.users) if mode != 'norelu' else add
        quantize=next(n for n in tail.users)
        dequantize=next(n for n in quantize.users)
        for n in dict.fromkeys((unit_inputs[0],unit_inputs[1],add,tail,quantize,dequantize)):
            env[n]=graph.node_copy(n,lambda n:env[n])
        graph.output(env[dequantize])
        section=torch.fx.GraphModule(q,graph)
        a=torch.arange(-128,128,dtype=torch.int16)[:,None].expand(256,256)
        b=torch.arange(-128,128,dtype=torch.int16)[None,:].expand(256,256)
        actual=section(a.to(torch.int8),b.to(torch.int8))
        scale=dequantize.args[1]
        expected=(a+b).clamp(-128 if mode == 'norelu' else 0,127).to(torch.float32)*scale
        results[mode]['pairs_checked']=actual.numel()
        results[mode]['exact']=bool(torch.equal(actual,expected))
        # Mutating a common scale or zero point must refuse, never leave a
        # selected site silently executing different arithmetic.
        before,sites=captured[-1]
        s=S.SharedGridSums(chosen);s.sites=sites
        original_inputs=next(n for n in before.graph.nodes if n.name==add.name).args
        for label,index,value in [('scale',1,2.0),('zero',2,1),('nan',1,float('nan'))]:
            bad=copy.deepcopy(before)
            n=next(n for n in bad.graph.nodes if n.name==original_inputs[1].name)
            args=list(n.args);args[index]=value;n.args=tuple(args)
            try:original_rewrite(s,bad)
            except S.OperandSumError:results[mode][label+'_refused']=True
            else:results[mode][label+'_refused']=False
        if mode=='inplace':
            bad=copy.deepcopy(before)
            n=next(n for n in bad.graph.nodes if n.name==add.name)
            with bad.graph.inserting_after(n.args[0]):bad.graph.call_function(torch.ops.aten.sin.default,(n.args[0],))
            try:original_rewrite(s,bad)
            except S.OperandSumError:results[mode]['mutable_dq_alias_refused']=True
            else:results[mode]['mutable_dq_alias_refused']=False
        for label in ('alpha','mutable','shape','fanout'):
            bad=copy.deepcopy(before)
            n=next(n for n in bad.graph.nodes if n.name==add.name)
            if label=='alpha':n.kwargs={'alpha':2}
            if label=='mutable':
                n.target=(torch.ops.aten.add.Tensor if sites[0]['mutable'] else torch.ops.aten.add_.Tensor)
            if label=='shape':
                src=n.args[0].args[0].args[0]
                src.meta['val']=torch.zeros(1,1)
            if label=='fanout':
                with bad.graph.inserting_after(n):bad.graph.call_function(torch.ops.aten.sin.default,(n,))
            try:original_rewrite(s,bad)
            except S.OperandSumError:results[mode][label+'_refused']=True
            else:results[mode][label+'_refused']=False

# Absence of opt-in preserves the existing float-result annotations and raw
# contraction exit. Reuse the same random seed/model size, not a new policy.
model=Block('plain').eval(); x=torch.randn(2,8)
legacy=RQ.apply_recipe(model,recipe,example_inputs=(x,),calibration_inputs=[(x,)])
results['legacy']={'stats':legacy._recipe_quantization_stats.get('shared_grid_operand_sums'),
                  'exits':legacy._recipe_quantization_stats['integer_epilogue']['by_exit']}
class Cascade(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a=torch.nn.Linear(8,8);self.b=torch.nn.Linear(8,8);self.out=torch.nn.Linear(8,4)
    def forward(self,x):
        y=torch.relu(self.a(x)+x)
        z=torch.relu(self.b(y)+y)
        return self.out(z)
model=Cascade().eval();x=torch.randn(2,8)
cascade=RQ.apply_recipe(model,chosen,example_inputs=(x,),calibration_inputs=[(x,),(x*2,)])
results['cascade']={'stats':cascade._recipe_quantization_stats['shared_grid_operand_sums']}
for kind in ('input','view','unknown','fresh_fanout'):
    g=torch.fx.Graph();x=g.placeholder('x');y=g.placeholder('y')
    x.meta['val']=y.meta['val']=torch.zeros(2,8)
    if kind=='view':left=g.call_function(torch.ops.aten.view.default,(x,[2,8]))
    elif kind=='unknown':left=g.call_function(torch.sin,(x,))
    elif kind=='fresh_fanout':left=g.call_function(torch.ops.aten.clone.default,(x,))
    else:left=x
    left.meta['val']=torch.zeros(2,8)
    a=g.call_function(torch.ops.aten.add_.Tensor,(left,y));a.meta['val']=torch.zeros(2,8)
    if kind=='fresh_fanout':g.call_function(torch.ops.aten.sin.default,(left,))
    s=S.SharedGridSums(chosen);s.select([a]);results[kind+'_alias_declined']=not s.selected
print(json.dumps(results))
"""


def _recipe():
    from merlin.targetgen import quant_methodology as QM
    from merlin.tests.targetgen.test_recipe_integer_epilogue import _METHODOLOGY, _path

    base = _base() | {
        "weight": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -127,
            "quant_max": 127,
        }
    }
    return QM.derive(
        base,
        _METHODOLOGY
        | {
            "recovery": {
                "cross_layer_equalization": False,
                "bias_correction": False,
            }
        },
        _path(),
    )


def test_real_pt2e_residual_closure_and_exhaustive_integer_section():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    root = repo_root()
    result = subprocess.run(
        [python, "-B", "-c", _SCRIPT, str(root / "src/merlin/targetgen"), str(root / "src"), json.dumps(_recipe())],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    for mode in ("plain", "inplace", "norelu"):
        row = rows[mode]
        assert row["stats"]["annotated"] == row["stats"]["rewritten"] == 1
        assert row["sum_input_scales"] == [1.0, 1.0]
        assert row["pairs_checked"] == 65536 and row["exact"]
        assert row["scale_refused"] and row["zero_refused"] and row["nan_refused"]
        assert all(row[label + "_refused"] for label in ("alpha", "mutable", "shape", "fanout"))
        assert row["exits"]["requant"] == 1
    assert rows["plain"]["integerized_exact"]
    assert rows["inplace"]["mutable_dq_alias_refused"]
    assert rows["cascade"]["stats"]["rewritten"] == 2
    assert all(rows[k + "_alias_declined"] for k in ("input", "view", "unknown", "fresh_fanout"))
    assert rows["alpha"]["stats"]["rewritten"] == 0
    assert rows["fanout"]["stats"]["rewritten"] == 0
    assert rows["alpha"]["left_float"] == rows["fanout"]["left_float"] == 1
    assert rows["legacy"]["stats"] is None and rows["legacy"]["exits"]["requant"] == 0


def test_public_speed_profile_is_a_reproducible_unselected_extension():
    import hashlib

    import yaml

    directory = repo_root() / "examples/gemmini/protocol"
    profile = yaml.safe_load((directory / "resnet-speed.yaml").read_bytes())
    baseline_bytes = (directory / profile["baseline"]["recipe"]).read_bytes()
    candidate_bytes = (directory / profile["candidate"]["recipe"]).read_bytes()
    assert hashlib.sha256(baseline_bytes).hexdigest() == profile["baseline"]["file_sha256"]
    assert hashlib.sha256(candidate_bytes).hexdigest() == profile["candidate"]["file_sha256"]
    baseline, candidate = json.loads(baseline_bytes), json.loads(candidate_bytes)
    assert candidate == QR.with_shared_grid_operand_sums(baseline)
    assert candidate["recipe_sha256"] == profile["candidate"]["recipe_sha256"]
    assert profile["selected_by_default"] is False
    assert profile["measurements"]["float_quality"] is None
    assert profile["requirements"]["float_quality"]["cosine_min"] == 0.98
    assert profile["requirements"]["compiled_quantized_agreement"] == "../numerics/models.yaml#numerics.functional"
    assert candidate["epilogue"]["pool"]["padding"] == baseline["epilogue"]["pool"]["padding"] == "unknown"


def test_ordinary_capture_normalization_and_residual_group(tmp_path):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m:
        pytest.skip("the TorchAO capture interpreter and selected Model2MLIR checkout are required")
    root = repo_root()
    loader = tmp_path / "loader.py"
    loader.write_text("""import torch
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a=torch.nn.Conv2d(16,16,1)
        self.b=torch.nn.Conv2d(16,16,3,padding=1)
        self.c=torch.nn.Conv2d(16,16,1)
        self.reader=torch.nn.Conv2d(16,16,1)
    def forward(self,x):
        y=self.c(torch.relu(self.b(torch.relu(self.a(x)))))
        y+=x
        return self.reader(torch.relu_(y))
def get_model_and_inputs():
    torch.manual_seed(41)
    return Block().eval(),(torch.randn(1,16,8,8),)
def get_calibration_inputs(model,inputs):
    yield inputs
    yield (inputs[0]*2,)
""")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(QR.with_shared_grid_operand_sums(_recipe())))
    output = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            "-B",
            str(root / "src/merlin/targetgen/_m2m_capture_worker.py"),
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe),
            "--out",
            str(output),
            "--m2m-dir",
            m2m,
            "--materialize-bundle",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "M2M_SESSION_STEPS": "2"},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    meta = json.loads((output / "meta.json").read_bytes())
    assert meta["ok"] and meta["opaque"] == 0
    assert meta["frontend_trace"]["status"] == "complete"
    assert meta["integerization_receipt"]["golden_agreement"]["status"] == "passed"
    assert meta["quantization_stats"]["shared_grid_operand_sums"]["rewritten"] == 1
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.tests.ir.test_compute_groups import _summing
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command

    text, receipt = normalize_capture_mlir((output / "model.mlir").read_text())
    assert not receipt["remaining_opaque_detail"]
    groups = CG.form_groups(mq.parse(text), "synthetic", oracle=_summing())
    residuals = [g for g in groups if g.operand_sum]
    assert len(residuals) == 1
    program = group_command.program(residuals[0])
    assert program.entry["op"] == "residual_add" and program.entry["epilogue"] == ["relu"]
    assert program.entry["lhs_scale"] == program.entry["rhs_scale"] == 1.0
