"""Trusted runtime bundles retain independent byte and ABI binding."""
import hashlib
import pytest

from merlin.compile.materialized_model_inputs import MaterializedModelInputs
from merlin.compile import baremetal_model


def binding(tmp_path):
    names = ["model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
             "inputs.npz", "input_order.json", "golden.npy", "goldens.npz", "output_order.json"]
    artifacts = {}
    for name in names:
        raw = name.encode()
        (tmp_path / name).write_bytes(raw)
        artifacts[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    return MaterializedModelInputs.bind(tmp_path, artifacts=artifacts,
                                       validator_identity={"worker_sha256": "a" * 64},
                                       source_identity={"content_sha256": "b" * 64, "artifacts": artifacts})


def test_bound_materialization_never_creates_original_issuer_receipt(tmp_path):
    selected = binding(tmp_path)
    assert selected.verify(tmp_path)["model.mlir"]["sha256"]
    assert not (tmp_path / "capture_receipt.json").exists()
    assert baremetal_model._saved_reference(tmp_path, None, execution=False, materialized_inputs=selected) is None


@pytest.mark.parametrize("name", ["model.mlir", "inputs.npz", "goldens.npz", "weights.safetensors"])
def test_changed_materialized_source_or_abi_is_rejected(tmp_path, name):
    selected = binding(tmp_path)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        selected.verify(tmp_path)


def test_reference_selection_without_binding_still_requires_capture_receipt(tmp_path):
    binding(tmp_path)
    with pytest.raises(baremetal_model.BaremetalModelError, match="not byte-verified"):
        baremetal_model._saved_reference(tmp_path, None, execution=False)
