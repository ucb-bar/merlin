"""The ordinary full-output reader consumes every receipt-bound quantized result."""
import hashlib
import json

import numpy as np
import pytest

from merlin.compile.baremetal_model import BaremetalModelError, _agreement, _full_references


def capture(tmp_path, *, change=None):
    (tmp_path / "model.mlir").write_text('''module {
      func.func @forward(%a: tensor<1x2xf32>, %b: tensor<1xf32>)
          -> (tensor<1x2xf32>, tensor<1xf32>) {
        return %a, %b : tensor<1x2xf32>, tensor<1xf32>
      }
    }''')
    abi = [{"shape": [1, 2], "dtype": "f32"}, {"shape": [1], "dtype": "f32"}]
    meta = {"output_abi": abi}
    ref = {"schema": "merlin.capture.integer_reference.v1", "output_abi": abi,
           "outputs": [[[1.25, -2.5]], [3.0]]}
    if change == "missing_result":
        ref["outputs"].pop()
    if change == "wrong_shape":
        ref["outputs"][1] = [[3.0]]
    if change == "actual_abi":
        meta["output_abi"] = list(reversed(abi))
        ref["output_abi"] = meta["output_abi"]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "integer-reference.json").write_text(json.dumps(ref))
    artifacts = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in tmp_path.iterdir()}
    (tmp_path / "capture_receipt.json").write_text(json.dumps({"artifacts": artifacts}))
    if change == "changed_reference":
        (tmp_path / "integer-reference.json").write_text(json.dumps({**ref, "outputs": [[[0, 0]], [0]]}))
    if change == "changed_metadata":
        (tmp_path / "meta.json").write_text(json.dumps({**meta, "unbound": True}))
    return tmp_path


def test_complete_quantized_reference_preserves_both_results(tmp_path):
    references = _full_references(capture(tmp_path), "integer-reference.json")
    assert [name for name, _ in references] == ["out0", "out1"]
    assert np.array_equal(references[0][1], np.array([[1.25, -2.5]], dtype=np.float32))
    assert np.array_equal(references[1][1], np.array([3.0], dtype=np.float32))


@pytest.mark.parametrize("change", ["missing_result", "wrong_shape", "actual_abi",
                                    "changed_reference", "changed_metadata"])
def test_complete_reference_rejects_omission_conversion_or_changed_identity(tmp_path, change):
    with pytest.raises(BaremetalModelError):
        _full_references(capture(tmp_path, change=change), "integer-reference.json")


def test_complete_mixed_reference_preserves_masks_and_exact_int64(tmp_path):
    (tmp_path / "model.mlir").write_text('''module {
      func.func @forward(%a: tensor<2xi1>, %b: tensor<1xi64>, %c: tensor<1xf32>)
          -> (tensor<2xi1>, tensor<1xi64>, tensor<1xf32>) {
        return %a, %b, %c : tensor<2xi1>, tensor<1xi64>, tensor<1xf32>
      }
    }''')
    abi = [{"shape": [2], "dtype": "i1"}, {"shape": [1], "dtype": "i64"},
           {"shape": [1], "dtype": "f32"}]
    (tmp_path / "meta.json").write_text(json.dumps({"output_abi": abi}))
    (tmp_path / "integer-reference.json").write_text(json.dumps({
        "schema": "merlin.capture.integer_reference.v1", "output_abi": abi,
        "outputs": [[True, False], [2**63 - 1], [1.25]]}))
    artifacts = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in tmp_path.iterdir()}
    (tmp_path / "capture_receipt.json").write_text(json.dumps({"artifacts": artifacts}))
    refs = _full_references(tmp_path, "integer-reference.json")
    assert [v.dtype for _, v in refs] == [np.dtype(bool), np.dtype(np.int64), np.dtype(np.float32)]
    assert refs[1][1].item() == 2**63 - 1
    tolerance = {"atol": 100, "rtol": 1}
    assert not _agreement(np.array([True, True]), refs[0][1], tolerance)["passed"]
    assert not _agreement(np.array([2**63 - 2], np.int64), refs[1][1], tolerance)["passed"]
    assert not _agreement(np.array([1., 0.]), refs[0][1], tolerance)["passed"]
