"""Additional scalar recipes use existing derivation and leave per-16 recipes intact."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe as Q


def _module():
    path = repo_root() / "examples/gemmini/protocol/native_scalar_recipes.py"
    spec = importlib.util.spec_from_file_location("native_scalar_recipes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(model):
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
        "readout": [{"kind": key, **epilogue[key]} for key in epilogue["order"]],
        "exits": epilogue["exits"],
    }
    # The fixture's path tokens differ in spelling from the actual selected
    # target. Bind this fixture explicitly; production reopens the real path.
    source["methodology"]["output_path_sha256"] = M.digest(path)
    source["recipe_sha256"] = Q.digest(source)
    return source, method, path


@pytest.mark.parametrize("model", ["tiny_llama", "smolvla"])
def test_scalar_choice_changes_only_explicit_weight_policy_and_identity(model):
    source, method, path = _inputs(model)
    before = copy.deepcopy((source, method, path))
    selected = _module().scalar_recipe(source, method, path, model=model)
    assert (source, method, path) == before
    assert selected["weight"] == {
        k: v for k, v in (source["weight"] | {"granularity": "tensor"}).items() if k != "channels_per_scale"
    }
    assert selected["recipe_sha256"] == Q.digest(selected) != Q.digest(source)
    assert selected["projection_input_reuse"]["max_nonuniform_readout_blocks"] == 64
    for key in ("activation", "dynamic", "framework_capture_policy", "epilogue", "software_numerical_engine"):
        assert selected[key] == source[key]


@pytest.mark.parametrize("mutation", ["source", "method", "path"])
def test_misbound_derivation_refuses(mutation):
    source, method, path = _inputs("tiny_llama")
    if mutation == "source":
        source["weight"]["quant_min"] = -128
    elif mutation == "method":
        method["scale_granularity"] = "per_channel"
    else:
        path["readout"][2]["columns_per_scale"] = 8
    with pytest.raises(ValueError):
        _module().scalar_recipe(source, method, path, model="tiny_llama")


def test_generator_adds_files_without_rewriting_baselines(tmp_path):
    module = _module()
    _, _, path = _inputs("tiny_llama")
    root = repo_root() / "examples/gemmini"
    baseline = {}
    for model in ("tiny_llama", "smolvla"):
        source, _, _ = _inputs(model)
        data = json.dumps(source).encode()
        filename = tmp_path / f"{model}-native-speed.json"
        filename.write_bytes(data)
        baseline[filename] = data
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"output_epilogue": {"output_path": path}}))
    records = module.generate(tmp_path, spec=root / "target/software-spec.yaml", contract=contract)
    assert set(records) == {"tiny_llama", "smolvla"}
    assert all(filename.read_bytes() == data for filename, data in baseline.items())
    assert all((tmp_path / row["candidate"]).is_file() for row in records.values())


def test_existing_per_tensor_observer_has_one_scale_and_zero_point():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("requires the selected TorchAO capture interpreter")
    source = r"""
import sys
sys.path.insert(0, sys.argv[1])
import torch, _recipe_quantizer as R
spec=R._spec({'dtype':'int8','mode':'static','granularity':'tensor','symmetric':True,
 'observer':'minmax','quant_min':-127,'quant_max':127},is_weight=True,eps=1e-8)
assert spec.qscheme==torch.per_tensor_symmetric and spec.ch_axis is None
observer=spec.observer_or_fake_quant_ctr(dtype=spec.dtype,qscheme=spec.qscheme,
 quant_min=spec.quant_min,quant_max=spec.quant_max)
weight=torch.tensor([[-127.,2.,0.],[0.01,-0.02,0.03],[7.,8.,9.]])
observer(weight);scale,zero=observer.calculate_qparams()
assert scale.shape==(1,) and zero.shape==(1,)
assert scale.item()==1. and zero.item()==0
print('scalar observer PASS')
"""
    result = subprocess.run(
        [python, "-B", "-c", source, str(repo_root() / "src/merlin/targetgen")],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "scalar observer PASS"
