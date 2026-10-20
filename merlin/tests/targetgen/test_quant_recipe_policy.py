"""Explicit successor arithmetic retains capability and independent reference gates."""

from __future__ import annotations

import copy
import json
import os
import subprocess

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe as Q
from merlin.targetgen import quant_recipe_policy as P


def inputs(model="tiny_llama"):
    root = repo_root() / "examples/gemmini"
    source = json.loads((root / f"protocol/recipes/{model}-native-speed.json").read_bytes())
    method = yaml.safe_load((root / "target/software-spec.yaml").read_bytes())["quantization"]["methodology"]
    epilogue = source["epilogue"]
    path = {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [
            {"dtype": "i32", "kind": "preload", "range": [-2147483648, 2147483647], "rounding": "none"},
            {"dtype": "i32", "kind": "accumulate", "operand_scale": "none"},
        ],
        "readout": [{"kind": k, **epilogue[k]} for k in epilogue["order"]],
        "exits": epilogue["exits"],
    }
    # Bind this synthetic path explicitly; production reopens its actual contract.
    source["methodology"]["output_path_sha256"] = M.digest(path)
    source["recipe_sha256"] = Q.digest(source)
    policy = {
        "schema": P.SCHEMA,
        "weight_granularity": "tensor",
        "float_quality": "report_only",
        "native_outputs": {k: v for k, v in epilogue["output_quantization"].items() if k != "schema"},
        "dynamic_activation_planes": 1 if "dynamic" in source else None,
        "integer_extensions": {
            key: source[key]
            for key in ("accumulator_residual", "integer_mean", "projection_input_reuse")
            if key in source
        },
    }
    return source, method, path, policy


@pytest.mark.parametrize("model", ["resnet50", "tiny_llama", "smolvla"])
def test_selected_scalar_native_and_single_dynamic_plane_preserve_other_arithmetic(model):
    source, method, path, policy = inputs(model)
    before = copy.deepcopy((source, method, path, policy))
    selected = P.select(source, method, path, policy=policy, model=model)
    assert (source, method, path, policy) == before
    assert selected["weight"]["granularity"] == "tensor" and "channels_per_scale" not in selected["weight"]
    assert selected["epilogue"] == source["epilogue"]
    for key in (
        "activation",
        "recovery",
        "framework_capture_policy",
        "software_numerical_engine",
        "accumulator_residual",
        "integer_mean",
        "projection_input_reuse",
        "families",
    ):
        assert selected.get(key) == source.get(key)
    if "dynamic" in source:
        assert set(selected["dynamic"]["planes"]) == set(source["dynamic"]["planes"])
        assert set(selected["dynamic"]["planes"].values()) == {1}
        assert Q.site_agreement_of(selected) is None and Q.site_agreement_of(source) is not None
        assert selected["dynamic"]["scales"] == source["dynamic"]["scales"]
    assert selected["recipe_sha256"] == Q.digest(selected) != Q.digest(source)
    assert selected["numerical_selection"]["source_recipe_sha256"] == source["recipe_sha256"]


def test_public_capability_derivation_uses_policy_without_private_recipe_or_model_identifier():
    source, method, path, policy = inputs("resnet50")
    base = Q.QuantRecipe(
        target="synthetic_target",
        unit="unit",
        weight=Q.TensorSpec("int8", "tensor", True, -127, 127),
        activation=Q.TensorSpec("int8", "tensor", True, -128, 127),
    ).to_dict() | {"accumulator_dtype": "int32"}
    base["recipe_sha256"] = Q.digest(base)
    selected = P.derive(base, method, path, policy=policy)
    assert selected["methodology"]["model"] is None
    assert selected["target"] == "synthetic_target"
    assert selected["weight"]["granularity"] == "tensor"
    assert selected["accumulator_residual"] == source["accumulator_residual"]
    assert selected["integer_mean"] == source["integer_mean"]
    assert Q.WINDOW_MEAN in selected["families"]


def test_successor_reporting_config_preserves_geometry_and_exact_quantized_comparison():
    from merlin.model_check.models import load, selected_float_policy, selected_host_math_policy

    root = repo_root() / "examples/gemmini/numerics"
    old, new = load(root / "models.yaml"), load(root / "models-speed-first.yaml")
    assert selected_float_policy(old) == selected_float_policy(new)
    for name in old.models:
        before, after = old.model(name), new.model(name)
        assert (before.chain, before.programs) == (after.chain, after.programs)
        assert all(rule["report_only"] for rule in after.judged["admission"])
        assert any(not rule.get("report_only") for rule in before.judged["admission"])
    assert selected_host_math_policy(new.model("tiny_llama"), new.full) == "portable_exp_trig"
    assert selected_host_math_policy(new.model("smolvla"), new.full) == "portable_neural_f32"


def test_complete_schedule_binds_selected_quantized_state_at_every_step():
    import numpy as np

    from merlin.model_check import chain as CH
    from merlin.model_check.models import load

    loaded = load(repo_root() / "examples/gemmini/numerics/models-speed-first.yaml")
    tiny = loaded.model("tiny_llama")
    run = CH.ChainRun()
    run.steps["prefill"] = [[np.array([1.0]), np.array([7.0]), np.array([32], np.int64)]]
    run.steps["decode"] = []
    tokens = np.arange(64, dtype=np.int64)
    for step in range(32):
        bound = CH.bind(tiny.program("decode"), tiny, {"tokens": tokens}, step, run)
        assert bound["in0"].item() == 32 + step
        assert bound["in1"].item() == 7 + step
        assert bound["in2"].item() == 32 + step
        run.steps["decode"].append([np.array([2.0]), bound["in1"] + 1, bound["in2"] + 1])
    assert sum(len(outputs) for steps in run.steps.values() for outputs in steps) == 99
    smol = loaded.model("smolvla")
    run = CH.ChainRun()
    run.steps["prefix_encode"] = [[np.array([True]), np.array([11.0])]]
    run.steps["flow_denoise"] = []
    for step in range(10):
        bound = CH.bind(smol.program("flow_denoise"), smol, {"noise": np.array([3.0])}, step, run)
        assert bound["in0"].item() is True
        assert bound["in1"].item() == 11 + step
        assert bound["in2"].item() == 3 + step
        assert bound["in3"].item() == pytest.approx(1 - step / 16)
        run.steps["flow_denoise"].append([bound["in2"] + 1, bound["in1"] + 1, bound["in3"] - 1 / 16])
    action = CH.bind(smol.program("action_decode"), smol, {}, 0, run)
    assert action["in0"].item() == 13
    run.steps["action_decode"] = [[action["in0"]]]
    assert sum(len(outputs) for steps in run.steps.values() for outputs in steps) == 33


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "foreign"},
        {"unknown": True},
        {"weight_granularity": "channel"},
        {"float_quality": "waive_integer_reference"},
        {"dynamic_activation_planes": True},
        {"dynamic_activation_planes": 2},
        {"native_outputs": {"operators": [[]], "reader_policy": "compute"}},
        {"native_outputs": {"operators": ["linear", "linear"], "reader_policy": "compute"}},
        {"native_outputs": {"operators": ["linear"], "reader_policy": "anything"}},
    ],
)
def test_policy_is_closed_and_has_no_implicit_precision_or_bypass(change):
    *_, policy = inputs()
    with pytest.raises(ValueError):
        P.validate(policy | change)


@pytest.mark.parametrize("field", ["recipe", "method", "path", "model", "capability", "calibration"])
def test_misbound_or_incompatible_selection_refuses(field):
    source, method, path, policy = inputs()
    model = "tiny_llama"
    if field == "recipe":
        source["weight"]["observer"] = "other"
    elif field == "method":
        method["calibration"]["activation"] = "minmax"
    elif field == "path":
        path["readout"][2]["columns_per_scale"] = 8
    elif field == "model":
        model = "another_workload"
    elif field == "capability":
        path["exits"]["requant"] = False
        source["methodology"]["output_path_sha256"] = M.digest(path)
        source["recipe_sha256"] = Q.digest(source)
    else:
        source["framework_capture_policy"]["calibration_count"] = 7
        source["recipe_sha256"] = Q.digest(source)
    with pytest.raises(ValueError):
        P.select(source, method, path, policy=policy, model=model)


@pytest.mark.parametrize("mutation", ["static_digits", "wrong_family", "unknown_smooth", "unknown_field"])
def test_single_dynamic_plane_never_reinterprets_other_plane_or_scale_semantics(mutation):
    source, method, path, policy = inputs()
    if mutation == "static_digits":
        source["static_planes"] = Q.static_planes_block(planes=2)
    elif mutation == "wrong_family":
        source["dynamic"]["families"] = [Q.CONTRACTION]
    elif mutation == "unknown_smooth":
        source["dynamic"]["smooth"] = "invented"
    else:
        source["dynamic"]["unproved"] = True
    # Preserve the authored block through methodology derivation so the refusal
    # exercises the selected policy rather than only the ancillary-change guard.
    for row in method["model_variants"]:
        if row["model"] == "tiny_llama" and "dynamic" in row["changes"]:
            row["changes"]["dynamic"] = copy.deepcopy(source["dynamic"])
    source["recipe_sha256"] = Q.digest(source)
    with pytest.raises(ValueError):
        P.select(source, method, path, policy=policy, model="tiny_llama")


_SCRIPT = r"""
import copy,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'src/merlin/targetgen'),sys.argv[2]]
import torch,torchao,_recipe_quantizer as R
from m2m.capture import pt2e_integer_reference as IR
from merlin.targetgen._capture_integer_mm import exact_integer_mm
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(119)
class Net(torch.nn.Module):
 def __init__(self):
  super().__init__();self.q=torch.nn.Linear(64,64);self.k=torch.nn.Linear(64,64);self.v=torch.nn.Linear(64,64)
 def forward(self,x):
  q,k,v=[layer(x).reshape(1,32,2,32).transpose(1,2) for layer in (self.q,self.k,self.v)]
  return torch.matmul(torch.softmax(torch.matmul(q,k.transpose(-2,-1))*.25,dim=-1),v).sin()
x=torch.randn(1,32,64);model=Net().eval();rows={}
for name in ('tiny_llama','smolvla'):
 recipe=json.loads((root/f'examples/gemmini/protocol/recipes/{name}-speed-first.json').read_bytes())
 q=R.apply_recipe(copy.deepcopy(model),recipe,example_inputs=(x,),calibration_inputs=[(x,),(x*.5,)])
 ref=IR.run_pt2e_integer_reference(q,(x,))
 with exact_integer_mm(torch):actual=q(x)
 assert torch.equal(ref.output,actual)
 stats=q._recipe_quantization_stats
 dynamic=stats['dynamic_contractions']['sites']
 assert len(dynamic)==2 and all(row['planes']==[1,1] and row['status']=='rewritten' for row in dynamic)
 assert stats['integer_epilogue']['by_exit']=={'requant':3,'accumulator':0}
 rows[name]={'integer_exact':True,'dynamic_sites':len(dynamic),'planes':[r['planes'] for r in dynamic],
             'native_exits':3,'scope':'synthetic section; no full-model quality or performance claim'}
print(json.dumps(rows))
"""


def test_real_speed_first_pipeline_and_independent_integer_reference(tmp_path):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("requires selected TorchAO and model2MLIR sources")
    result = subprocess.run(
        [python, "-B", "-c", _SCRIPT, str(repo_root()), m2m], capture_output=True, text=True, timeout=90, check=False
    )
    assert result.returncode == 0, result.stderr[-7000:]
    rows = json.loads(result.stdout.strip().splitlines()[-1])
    (tmp_path / "actual-pipeline.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert all(row["integer_exact"] for row in rows.values())
