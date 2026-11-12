import hashlib
import json

import pytest
from corefixtures.compile_input_fixtures import SOURCE, _write_capture, _write_weighted_capture

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
