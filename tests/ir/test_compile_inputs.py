import hashlib
import json
import shutil
import struct

import pytest
from corefixtures.compile_input_fixtures import SOURCE, _write_capture, _write_weighted_capture

from merlin.common.paths import repo_root
from merlin.compiler.frontend.inputs import load_compile_inputs, stage_compile_inputs


def test_c_parameter_binding_does_not_freeze_mutable_state(tmp_path):
    from merlin.compiler.lower.c.parameters import freeze_c_arguments

    capture, staged = (tmp_path / "capture", tmp_path / "compiler")
    _write_weighted_capture(capture)
    manifest = capture / "weights.safetensors.manifest.json"
    rows = json.loads(manifest.read_text())
    rows["0"]["kind"] = "buffer"
    raw = json.dumps(rows).encode()
    manifest.write_bytes(raw)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][manifest.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))
    stage_compile_inputs(capture, staged)
    with pytest.raises(ValueError, match="state interface"):
        freeze_c_arguments(staged, name="model", source_sha256=load_compile_inputs(staged).program_sha256)


@pytest.mark.parametrize("name", ["model.mlir"])
def test_changed_declared_input_refuses_without_publishing(tmp_path, name):
    capture, staged = (tmp_path / "capture", tmp_path / "compiler")
    _write_capture(capture)
    (capture / name).write_bytes((capture / name).read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="differs from receipt"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


def test_typed_stage_reader_preserves_identity_and_refuses_forged_abi(tmp_path):
    capture, staged = (tmp_path / "capture", tmp_path / "compiler")
    _write_capture(capture)
    manifest = stage_compile_inputs(capture, staged)
    loaded = load_compile_inputs(staged)
    assert loaded.entry == "forward"
    assert loaded.original_source_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert loaded.program_sha256 == hashlib.sha256(loaded.program.encode()).hexdigest()
    assert loaded.precision_policy is None
    signature = json.loads((staged / "signature.json").read_bytes())
    signature["inputs"][0]["shape"] = [7]
    raw = json.dumps(signature).encode()
    (staged / "signature.json").write_bytes(raw)
    manifest["members"]["signature.json"] = hashlib.sha256(raw).hexdigest()
    (staged / "compile-inputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inputs differ from parsed function"):
        load_compile_inputs(staged)


def _kernel_capture(directory):
    directory.mkdir()
    fixture = repo_root() / "tests/fixtures/ttir_stage"
    names = ("model.mlir", "source.ttir", "spec.json", "provenance.json")
    for name in names:
        shutil.copyfile(fixture / name, directory / name)
    (directory / "weights.safetensors").write_bytes(struct.pack("<Q", 8) + b"{}      ")
    (directory / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "input", "name": "x"}}))
    names += ("weights.safetensors", "weights.safetensors.manifest.json")
    receipt = {"schema": "t2m.capture-receipt.v2", "profile": "shared-functional-tensor",
        "materialized_abi": {"complete": True, "inputs": 1, "outputs": 1},
        "artifacts": {name: {"bytes": len((directory / name).read_bytes()),
            "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()} for name in names}}
    (directory / "capture_receipt.json").write_text(json.dumps(receipt))


def test_public_kernel_stage_keeps_source_and_requires_actual_backend_admission(tmp_path):
    from merlin import api
    from merlin.compiler.frontend.staged import prepare

    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _kernel_capture(capture)
    stage_compile_inputs(capture, stage)
    loaded = load_compile_inputs(stage)
    source, preparation = prepare(api.StagedProgram(stage), max_initialization_bytes=0)
    assert source.text == loaded.program == (stage / "program.mlir").read_text()
    assert loaded.signature["inputs"] == [{"shape": [1], "dtype": "f32"}]
    assert preparation.policy()["numerical_transforms"] == []
    assert (stage / "source/source.ttir").read_bytes() == (capture / "source.ttir").read_bytes()

    class Unadmitted:
        name = "unadmitted"

        def validate(self, request):
            pytest.fail("source policy must be admitted before backend validation")

        def compile(self, request):
            pytest.fail("source policy cannot authorize its own compilation")

    with pytest.raises(ValueError, match="preparation admission"):
        api.compile(api.StagedProgram(stage), backend=Unadmitted(), output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("member", ("program.mlir", "precision-policy.json"))
def test_resealed_kernel_stage_cannot_change_computation_or_numerical_policy(tmp_path, member):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _kernel_capture(capture)
    manifest = stage_compile_inputs(capture, stage)
    path = stage / member
    if member == "program.mlir":
        original = path.read_text()
        assert "arith.addf" in original
        path.write_text(original.replace("arith.addf", "arith.subf"))
    else:
        policy = json.loads(path.read_bytes())
        policy["numerical_transforms"] = ["reassociate"]
        path.write_text(json.dumps(policy))
    manifest["members"][member] = hashlib.sha256(path.read_bytes()).hexdigest()
    (stage / "compile-inputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reconstructed intake"):
        load_compile_inputs(stage)


def test_declaring_an_evaluator_file_does_not_make_it_a_compiler_member(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _kernel_capture(capture)
    manifest = stage_compile_inputs(capture, stage)
    path = stage / "source/golden.npy"
    path.write_bytes(b"forbidden evaluator values")
    manifest["members"]["source/golden.npy"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (stage / "compile-inputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reconstructed intake"):
        load_compile_inputs(stage)
