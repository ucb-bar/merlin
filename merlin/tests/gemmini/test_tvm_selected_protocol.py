"""Reject implicit protocol fallback and route the frozen contract to preparation."""

import importlib.util

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def preparation(monkeypatch):
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("selected_preparation_test", folder / "prepare_remote_tinyllama.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkout_protocol_cannot_replace_selected_contract(preparation):
    with pytest.raises(ValueError, match="exact 8ec99308"):
        preparation.load_selected_protocol(repo_root() / "examples/gemmini/protocol/experiment.yaml")


@pytest.mark.parametrize("phase", ["prefill", "decode", "references"])
def test_selected_paths_reach_worker(preparation, monkeypatch, tmp_path, phase):
    contract, models, recipe = (tmp_path / name for name in ("protocol.yaml", "models.yaml", "recipe.json"))
    monkeypatch.setattr(preparation, "load_selected_protocol",
                        lambda path: ({"models": {"tiny_llama": {}}}, models, {"tiny_llama": recipe}))
    monkeypatch.setattr(preparation, "check_selected_input", lambda *args: None)
    (tmp_path / "m2m").mkdir()
    monkeypatch.setenv("MERLIN_M2M_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(preparation.subprocess, "run", lambda command, **kwargs: calls.append(command))
    preparation.run_stage(phase, tmp_path, contract)
    command = calls[0]
    if phase == "references":
        assert command[command.index("--protocol") + 1] == str(contract)
        assert command[command.index("--models") + 1] == str(models)
    else:
        assert command[command.index("--recipe") + 1] == str(recipe)


def test_wrong_input_refuses_before_worker(preparation, monkeypatch, tmp_path):
    input_path = tmp_path / "prepared-inputs/tiny-v00.npz"
    input_path.parent.mkdir()
    input_path.write_bytes(b"unselected input")
    model = {"external_evaluation_input": {"sha256": "0" * 64}}
    monkeypatch.setattr(preparation, "load_selected_protocol",
                        lambda path: ({"models": {"tiny_llama": model}}, None, {}))
    monkeypatch.setattr(preparation.subprocess, "run", lambda *args, **kwargs: pytest.fail("worker started"))
    with pytest.raises(ValueError, match="Input bytes differ"):
        preparation.run_stage("prefill", tmp_path, tmp_path / "protocol.yaml")


@pytest.fixture
def frozen_contract():
    return (repo_root() / "out/artifacts/handoff/gemmini-firesim-elf-plan/v1/latest"
            / "run-contract/examples/gemmini/protocol/experiment.yaml")


def test_pinned_tinyllama_calibration_planes_and_complete_chain(preparation, frozen_contract):
    import json

    protocol, _, recipes = preparation.load_selected_protocol(frozen_contract)
    tiny = protocol["models"]["tiny_llama"]
    assert tiny["calibration"] == dict(source="WikiText2_train", prefill_samples=128,
                                      decode_samples=1024, decode_stride=1)
    assert tiny["host_math_policy"] == "portable_exp_trig"
    assert tiny["chain"] == dict(kind="causal", prefill_tokens=32, decode_steps=32)
    assert tiny["selected_chain_steps"] == 33
    assert tiny["declared_arrays_per_chain"] == 99
    assert tiny["quantization"]["attention"] == dict(scales="static_heads", softmax_output_planes=2,
                                                   partner_planes=2, ordinary_qk_planes=2)
    recipe = json.loads(recipes["tiny_llama"].read_text())
    assert recipe["dynamic"]["planes"] == dict(residual_operand=2, softmax_output=2, softmax_partner=2)
    assert tiny["stages"]["prefill"]["inputs"][0]["shape"] == [1, 32]
    assert tiny["stages"]["decode"]["inputs"][0]["shape"] == [1, 1]
    for stage in tiny["stages"].values():
        assert [(x["dtype"], x["shape"]) for x in stage["outputs"]] == [
            ("f32", [1, 32000]), ("f32", [2, 22, 1, 4, 64, 64]), ("i64", [22])]
    assert tiny["recurrence"]["teacher_forced"] is True
    assert tiny["recurrence"]["kv_and_positions"] == "prefill_outputs_then_previous_decode_outputs"


def test_pinned_quality_and_all_model_output_rosters(preparation, frozen_contract):
    import yaml

    protocol, models, _ = preparation.load_selected_protocol(frozen_contract)
    numerics = yaml.safe_load(models.read_text())["numerics"]
    assert protocol["profiles"]["functional_v1"]["fp_quality"] == "report_only"
    assert protocol["profiles"]["quality_v1"]["selected_as_phase1_exit_gate"] is False
    assert numerics["functional"] == dict(atol=2e-6, rtol=1e-5, equal_nan=False,
                                         finite_required=True, integer_boolean="exact",
                                         outputs="all_declared_every_step")
    totals = {}
    for name, model in protocol["models"].items():
        totals[name] = sum(stage["steps"] * len(stage["outputs"]) for stage in model["stages"].values())
        assert totals[name] == model["declared_arrays_per_chain"]
    assert totals == dict(resnet50=1, tiny_llama=99, smolvla=33)
    assert protocol["models"]["smolvla"]["architecture"]["actual_cameras"] == 1


@pytest.mark.parametrize("fault", ["chain", "recipe", "input_manifest"])
def test_changed_contract_dependency_is_rejected(preparation, frozen_contract, tmp_path, fault):
    import json
    import shutil

    destination = tmp_path / "contract"
    shutil.copytree(frozen_contract.parent.parent, destination)
    protocol = destination / "protocol/experiment.yaml"
    if fault == "chain":
        target = destination / "numerics/models.yaml"
        target.write_text(target.read_text().replace("prefill_tokens: 32", "prefill_tokens: 128"))
    elif fault == "recipe":
        target = destination / "protocol/recipes/tiny_llama.json"
        value = json.loads(target.read_text())
        value["dynamic"]["planes"]["residual_operand"] = 1
        target.write_text(json.dumps(value))
    else:
        target = destination / "numerics/MANIFEST.json"
        value = json.loads(target.read_text())
        value["models"]["tiny_llama"]["derived"]["visible/v00/inputs.npz"] = "0" * 64
        target.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        preparation.load_selected_protocol(protocol)
