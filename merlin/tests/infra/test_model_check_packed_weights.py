"""Catalog references keep packed tensors outside MLIR and bind their exact bytes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects.builtin import i8

from merlin.common import mlir_query as mq
from merlin.llvmlower import blob_weights as BW
from merlin.model_check import catalog as CAT

REF = """module {
func.func @forward(%a: tensor<2x3xi8>, %w: tensor<3x4xi8>) -> tensor<2x4xi32> {
  %z = arith.constant dense<0> : tensor<2x4xi32>
  %y = linalg.matmul ins(%a, %w : tensor<2x3xi8>, tensor<3x4xi8>)
      outs(%z : tensor<2x4xi32>) -> tensor<2x4xi32>
  return %y : tensor<2x4xi32>
} }"""

DEV = """module {
func.func private @dev(tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi32>
func.func @forward(%a: tensor<2x3xi8>, %w: tensor<3x4xi8>) -> tensor<2x4xi32> {
  %y = func.call @dev(%a, %w) : (tensor<2x3xi8>, tensor<3x4xi8>) -> tensor<2x4xi32>
  return %y : tensor<2x4xi32>
} }"""


@pytest.fixture
def prepared(tmp_path):
    weights = np.arange(-6, 6, dtype=np.int8).reshape(3, 4)
    reference = tmp_path / "ref.mlir"
    reference.write_text(REF)
    dev_dir = tmp_path / "dev"
    dev_dir.mkdir()
    module = mq.parse(DEV)
    binder = BW.WeightArgs(module)
    call = next(op for op in module.walk() if op.name == "func.call")
    call.operands[1] = binder(weights, i8)
    binder.finish()
    binder.write(dev_dir)
    dev = dev_dir / "model.mlir"
    dev.write_text(str(module))
    sidecar = {"call_buffers": {"dev": [{"role": "input"}, {"role": "weight"}]}}
    catalog = CAT.ProgramCatalog("t", tmp_path, reference, dev, {}, sidecar, {"dev": "unused in this test"})
    return catalog, weights


def test_evaluate_and_record_calls_bind_stored_values_in_entry_order(prepared, tmp_path, monkeypatch):
    catalog, weights = prepared
    inputs = np.arange(-3, 3, dtype=np.int8).reshape(2, 3)
    seen = []

    def kernel(_interface, values, roles):
        assert roles == ("input", "weight")
        seen.append([v.copy() for v in values])
        return [values[0].astype(np.int32) @ values[1].astype(np.int32)]

    monkeypatch.setattr(CAT.KR, "call_outputs", kernel)
    store = CAT.BlobStore(tmp_path / "arrays")
    evaluated = CAT.evaluate(catalog, [inputs, weights], blobs=store)
    assert all(row["exact"] for row in evaluated.consistent())
    calls = CAT.record_calls(catalog, [inputs, weights], blobs=store)
    assert len(calls) == len(evaluated.calls) == 1
    for actual in seen:
        np.testing.assert_array_equal(actual[0], inputs)
        np.testing.assert_array_equal(actual[1], weights)


@pytest.mark.parametrize("change", ["missing", "storage_shape", "entry_shape", "entry_dtype"])
def test_packed_weight_plan_must_match_storage_and_prepared_abi(prepared, change):
    catalog, weights = prepared
    path = catalog.dev_prepared.parent / BW.PLAN_FILE
    if change == "missing":
        (path.parent / BW.VALUES_FILE).unlink()
    else:
        doc = json.loads(path.read_text())
        if change == "entry_dtype":
            doc["args"][0]["dtype"] = "i16"
            np.savez(path.parent / BW.VALUES_FILE, w0=weights.astype(np.int16))
        else:
            doc["args"][0]["shape"] = [4, 3]
        path.write_text(json.dumps(doc))
        if change == "entry_shape":
            np.savez(path.parent / BW.VALUES_FILE, w0=weights.reshape(4, 3))
    with pytest.raises((CAT.CatalogError, ValueError, FileNotFoundError)):
        CAT.evaluate(catalog, [np.ones((2, 3), np.int8), weights])


def test_catalog_preparation_selects_packed_storage_and_restores_outer_policy(tmp_path, monkeypatch):
    from merlin.compile import complete_group_routing
    from merlin.runtime.backends import zephyr_model

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "linalg.mlir").write_text(REF)
    seen = []

    def prepare(source, directory, **options):
        device = "complete_group_device" in options
        seen.append((device, BW.is_requested()))
        path = directory / "model.prepared.mlir"
        path.write_text(REF)
        if device:
            (directory / "device_signatures.json").write_text(json.dumps({"entries": {}}))
        return path

    monkeypatch.setattr(zephyr_model, "_prepare_model_mlir", prepare)
    monkeypatch.setattr(
        complete_group_routing, "routing",
        lambda *a, **k: SimpleNamespace(operand_dtype="int8", accum_dtype="i32", numeric_policy=None),
    )
    with BW.requested(False):
        CAT.prepare(stage, tmp_path / "catalog", target="t")
        assert not BW.is_requested()
    assert seen == [(False, False), (True, True)]
