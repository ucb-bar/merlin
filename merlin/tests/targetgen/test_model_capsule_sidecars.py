"""Whole-model exports require both sidecars before touching their destination.

Synthetic capture artifacts only: no framework imports, capture subprocess, or hardware.
"""

from types import SimpleNamespace

import pytest
import yaml

from merlin.targetgen import capsule_source as source


@pytest.fixture
def capture(tmp_path, monkeypatch):
    loader = tmp_path / "loader.py"
    loader.write_text("raise AssertionError('synthetic loader must never execute')\n")
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"synthetic opaque weights and bias bytes")
    manifest = tmp_path / "weights.safetensors.manifest.json"
    manifest.write_bytes(b'{"0": "weight", "1": "bias"}\n')
    mlir = (
        f'builtin.module attributes {{prov.weights_file = "{weights}"}} {{\n'
        "  func.func @forward(%0: tensor<1xf32>) -> tensor<1xf32> {\n"
        "    return %0 : tensor<1xf32>\n  }\n}\n"
    )
    artifact = source.CapsuleArtifacts(
        op="model",
        dtype="f32",
        pytorch_src=loader.read_text(),
        linalg_mlir=mlir,
        inputs=[[1.0]],
        golden=[1.0],
        weights_path=str(weights),
        meta={
            "weights_manifest": str(manifest),
            "input_abi": [{"shape": [1], "dtype": "f32"}],
            "output_abi": [{"shape": [1], "dtype": "f32"}],
        },
    )

    def capture_loader(selected, dtype, **kwargs):
        assert selected == loader
        assert dtype == "f32"
        return artifact

    fake = SimpleNamespace(m2m_dir=tmp_path / "unused-model2mlir", capture_loader=capture_loader)
    binding = SimpleNamespace(
        operand_dtype="f32",
        target="synthetic",
        cap_dtype=lambda value: value,
        tiers=[],
        compare="tolerance_float",
        atol=0.0,
        rtol=0.0,
    )
    monkeypatch.setattr(source, "derived_recipe", lambda *args: None)
    monkeypatch.setattr(source, "model_accelerator_demand", lambda *args: (None, []))
    monkeypatch.setattr(source.subprocess, "run", lambda *args, **kwargs: pytest.fail("no subprocess permitted"))
    entry = {"kind": "model", "cat": "model", "name": "synthetic", "loader": str(loader)}
    output = tmp_path / "capsules"
    return SimpleNamespace(
        weights=weights,
        manifest=manifest,
        artifact=artifact,
        destination=output / "model/synthetic",
        write=lambda: source.write_model_capsule(entry, binding, output, source=fake),
    )


@pytest.mark.parametrize("missing", ["weights", "manifest"])
@pytest.mark.parametrize("existing", [False, True])
def test_missing_sidecar_refuses_before_destination_changes(capture, missing, existing):
    getattr(capture, missing).unlink()
    if existing:
        capture.destination.mkdir(parents=True)
        for name in ("capsule.weights.safetensors", "capsule.weights.safetensors.manifest.json", "capsule.yaml"):
            (capture.destination / name).write_bytes(b"existing bytes: " + name.encode())
        before = {p.name: p.read_bytes() for p in capture.destination.iterdir()}
    with pytest.raises(source.M2MUnavailable):
        capture.write()
    if existing:
        assert {p.name: p.read_bytes() for p in capture.destination.iterdir()} == before
    else:
        assert not capture.destination.exists()


@pytest.mark.parametrize("empty_weights", [False, True])
def test_success_preserves_sidecar_bytes_and_uses_relative_reference(capture, empty_weights):
    if empty_weights:
        capture.weights.write_bytes(b"")
    destination = capture.write()
    assert destination == capture.destination
    assert (destination / "capsule.weights.safetensors").read_bytes() == capture.weights.read_bytes()
    assert (destination / "capsule.weights.safetensors.manifest.json").read_bytes() == capture.manifest.read_bytes()
    assert (destination / "capsule.interface.mlir").read_text() == capture.artifact.linalg_mlir.replace(
        str(capture.weights), "capsule.weights.safetensors"
    )
    capsule = yaml.safe_load((destination / "capsule.yaml").read_text())
    attributes = capsule["operation"]["attributes"]
    assert attributes["weights"] == "capsule.weights.safetensors"
    assert attributes["weights_manifest"] == "capsule.weights.safetensors.manifest.json"
    assert list(destination.glob("*.safetensors")) == [destination / "capsule.weights.safetensors"]
