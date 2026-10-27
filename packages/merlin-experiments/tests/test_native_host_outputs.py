"""W3 keeps every captured result and derive independently rechecks the receipt."""

import ctypes
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from merlin_experiments import model_qualification as Q
from merlin_experiments.phase0 import evidence

from merlin.targetgen.application_inventory import verify_capture_receipt


def _identity(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _case(tmp_path, monkeypatch, *, changed_output=None):
    from merlin.llvmlower import abi, lower
    from merlin.runtime import dispatch_runtime as runtime

    arrays = [
        np.array([1, -0.0], dtype=np.float32),
        np.array([2**63 - 1], dtype=np.int64),
        np.array(True),
        np.array([1 + 2j], dtype=np.complex64),
        np.array([1.5, np.inf], dtype=np.float32),  # BF16 archive values are lifted to f32.
        np.array([], dtype=np.float16),
        np.array([np.nan, np.inf], dtype=np.float32),
    ]
    dtypes = ["f32", "i64", "i1", "complex64", "bf16", "f16", "f32"]
    storage = [
        (a.view(np.uint32) >> 16).astype(np.uint16) if dtype == "bf16" else a.copy()
        for a, dtype in zip(arrays, dtypes, strict=True)
    ]
    bundle = tmp_path / "capture"
    bundle.mkdir()
    types = [
        "tensor<" + "".join(f"{n}x" for n in a.shape) + ("complex<f32>" if d == "complex64" else d) + ">"
        for a, d in zip(arrays, dtypes, strict=True)
    ]
    arguments = ", ".join(f"%arg{i}: {typ}" for i, typ in enumerate(types))
    values = ", ".join(f"%arg{i}" for i in range(len(types)))
    result_types = ", ".join(types)
    (bundle / "model.mlir").write_text(
        f"module {{ func.func @forward({arguments}) -> ({result_types}) {{ func.return {values} : {result_types} }} }}"
    )
    output_abi = [{"shape": list(a.shape), "dtype": d} for a, d in zip(arrays, dtypes, strict=True)]
    (bundle / "meta.json").write_text(json.dumps({"output_abi": output_abi}))
    (bundle / "weights.safetensors").write_bytes(b"fixture")
    (bundle / "weights.safetensors.manifest.json").write_text("{}")
    (bundle / "input_order.json").write_text("{}")
    np.savez(bundle / "inputs.npz", **{f"in{i}": a for i, a in enumerate(arrays)})
    np.savez(bundle / "goldens.npz", **{f"out{i}": a for i, a in enumerate(arrays)})
    np.save(bundle / "golden.npy", arrays[0])
    (bundle / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "materialized_abi": {"complete": True},
                "artifacts": {p.name: _identity(p) for p in bundle.iterdir()},
            }
        )
    )
    monkeypatch.setattr(runtime, "resolve_forward_args", lambda *_: storage)

    def fake_lower(model, destination, **kwargs):
        destination.mkdir()
        paths = {
            name: destination / filename
            for name, filename in (("ll_path", "model.ll"), ("host_so", "model.so"), ("audit_index", "audit.json"))
        }
        for path in paths.values():
            path.write_text("fixture")
        return SimpleNamespace(**paths)

    calls = []

    def host(buffers):
        calls.append(len(buffers))
        for index, ((pointer, shape), expected) in enumerate(zip(buffers[len(storage) :], storage, strict=True)):
            assert shape == list(expected.shape)
            value = expected.copy()
            if index == changed_output:
                value.flat[0] -= 1
            ctypes.memmove(pointer, value.ctypes.data, value.nbytes)

    monkeypatch.setattr(lower, "lower_model_file", fake_lower)
    monkeypatch.setattr(abi.HostModel, "load", lambda *_: host)
    request = {
        "bundle": str(bundle),
        "workflow": Q.inspect_workflow(bundle),
        "numeric_policy": {"atol": 0, "rtol": 0},
        "native_host_only": True,
        "package": None,
        "target": None,
        "evidence_bundle": None,
        "execute": False,
    }
    root = tmp_path / "qualification"
    root.mkdir()
    observed = Q._native_host_worker(request, root)
    assert calls == [2 * len(arrays)]
    return bundle, root, request, observed, arrays


def _qualification(root, request, observed):
    (root / "request.json").write_text(json.dumps(request))
    (root / "observations.json").write_text(json.dumps(observed))
    producer = Q.Path(Q.__file__)
    document = {
        "schema": Q.SCHEMA,
        "status": "completed",
        "worker_returncode": 0,
        "error": None,
        "native_host_numerical_verified": True,
        "whole_workload_validation_verified": False,
        "target": None,
        "request": request,
        "observations": observed,
        "compiler": {
            "route": "merlin.llvmlower.lower_model_file:host",
            "producer": {"path": str(producer), **_identity(producer)},
        },
        "artifacts": {
            p.relative_to(root).as_posix(): _identity(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "qualification.json"
        },
    }
    path = root / "qualification.json"
    path.write_text(json.dumps(document))
    return path


def test_native_worker_and_derive_recheck_every_mixed_output(tmp_path, monkeypatch):
    bundle, root, request, observed, arrays = _case(tmp_path, monkeypatch)
    assert observed["native_host_numerical_verified"] is True
    row = observed["native_lowerings"][0]
    assert len(row["outputs"]) == len(arrays)
    assert row["nonfinite_output_elements"] == 3
    qualification = _qualification(root, request, observed)
    applications = {
        "app": {
            "capture_source_path": str(bundle / "model.mlir"),
            "capture_sha256": _identity(bundle / "model.mlir")["sha256"],
            "capture_receipt": verify_capture_receipt(bundle / "model.mlir"),
        }
    }

    def read(path, *_args, **_kwargs):
        return path.read_bytes()

    result = evidence._native_baseline_observations({"app": qualification}, applications, read)
    assert result["app"]["output_abi"] == row["output_abi"]
    # Rehash the altered artifact and keep its claimed passing metrics. The
    # independent consumer must still reject a nonzero output's wrong int64 word.
    with np.load(root / "program-000/outputs.npz") as archive:
        altered = {key: archive[key] for key in archive.files}
    altered["out1"][0] -= 1
    np.savez(root / "program-000/outputs.npz", **altered)
    _qualification(root, request, observed)
    with pytest.raises(ValueError, match="numerical comparison"):
        evidence._native_baseline_observations({"app": qualification}, applications, read)


def test_native_worker_does_not_let_output_zero_mask_a_changed_integer_result(tmp_path, monkeypatch):
    _, _, _, observed, _ = _case(tmp_path, monkeypatch, changed_output=1)
    assert observed["native_host_numerical_verified"] is False
    assert observed["native_lowerings"][0]["outputs"][0]["matched"] is True
    assert observed["native_lowerings"][0]["outputs"][1]["matched"] is False


@pytest.mark.parametrize(
    "golden,actual",
    [
        (np.array([2**63 - 1], np.int64), np.array([2**63 - 2], np.int64)),
        (np.array([1 + 2j], np.complex64), np.array([1 + 3j], np.complex64)),
        (np.array([0x7FC00001], np.uint32).view(np.float32), np.array([0x7FC00002], np.uint32).view(np.float32)),
        (np.array([np.inf], np.float32), np.array([-np.inf], np.float32)),
    ],
)
def test_native_comparison_refuses_lossy_or_special_word_substitution(golden, actual):
    assert not Q._native_host_comparison(actual, golden, {"atol": 1, "rtol": 1})["matched"]


def test_native_references_refuse_incomplete_or_unbound_archive(tmp_path, monkeypatch):
    bundle, _, request, _, _ = _case(tmp_path, monkeypatch)
    metadata = json.loads((bundle / "meta.json").read_bytes())
    outputs = [(row["shape"], row["dtype"]) for row in metadata["output_abi"]]
    np.savez(bundle / "goldens.npz", out0=np.ones(2, np.float32))
    with pytest.raises(ValueError, match="reference bytes differ"):
        Q._native_host_references(bundle, outputs)
    receipt = json.loads((bundle / "capture_receipt.json").read_bytes())
    receipt["artifacts"]["goldens.npz"] = _identity(bundle / "goldens.npz")
    (bundle / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="leaves"):
        Q._native_host_references(bundle, outputs)
    receipt = json.loads((bundle / "capture_receipt.json").read_bytes())
    del receipt["artifacts"]["goldens.npz"]
    (bundle / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="every output"):
        Q._native_host_references(bundle, outputs)


def test_native_reference_retains_legacy_single_fp32(tmp_path):
    golden = np.array(1.5, np.float32)
    np.save(tmp_path / "golden.npy", golden)
    (tmp_path / "capture_receipt.json").write_text(
        json.dumps(
            {
                "artifacts": {"golden.npy": _identity(tmp_path / "golden.npy")},
            }
        )
    )
    (result,) = Q._native_host_references(tmp_path, [([], "f32")])
    assert result.shape == () and result.dtype == golden.dtype and result.tobytes() == golden.tobytes()
    assert Q._native_host_comparison(result, golden, {"atol": 0, "rtol": 0})["matched"]


def test_derive_reader_retains_legacy_finite_single_fp32_comparison(tmp_path):
    golden = np.array([1.5], np.float32)
    path = tmp_path / "output.npy"
    np.save(path, golden)
    raw = path.read_bytes()
    assert (
        Q._verify_native_host_outputs(
            tmp_path,
            [([1], "f32")],
            {"max_absolute_error": 0, "nonfinite_output_elements": 0},
            {"program-000/output.npy": raw},
            {"atol": 0, "rtol": 0},
            legacy_golden=raw,
        )
        == 0
    )
