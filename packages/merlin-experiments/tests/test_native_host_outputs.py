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
    monkeypatch.setattr(Q, "_native_host_arguments", lambda *_: storage)
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


def _argument_bundle(root, rows, arrays, kind="input"):
    """A minimal manifest bundle; argument storage and ABI come from the caller."""
    import struct

    root.mkdir()
    types = ["tensor<" + "".join(f"{n}x" for n in row["abi_shape"]) + row["abi_element"] + ">" for row in rows]
    signature = ", ".join(f"%arg{i}: {typ}" for i, typ in enumerate(types))
    (root / "model.mlir").write_text(f"module {{ func.func @forward({signature}) {{ func.return }} }}")
    manifest, extra, order = {}, {}, {}
    for i in range(len(rows)):
        name = f"x{i}" if kind == "input" else f"b_pair_{i:03}" if kind == "buffer" else f"c_lifted_tensor_{i:03}"
        manifest[str(i)] = {"kind": kind, "name": name}
        order[name] = i
        if kind != "input":
            extra[f"buf::pair.{i:03}" if kind == "buffer" else f"pair.{i:03}"] = arrays[i]
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
    (root / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (root / "input_order.json").write_text(json.dumps(order))
    np.savez(root / "inputs.npz", **{f"in{i}": array for i, array in enumerate(arrays)} if kind == "input" else {})
    np.savez(root / "extra.npz", **extra)
    return root


@pytest.mark.parametrize("kind", ["input", "buffer", "lifted"])
def test_saved_movement_complex_arguments_keep_every_component_word(tmp_path, kind):
    from merlin.common.paths import merlin_dir
    from merlin.runtime.dispatch_runtime import ForwardArgsCache, resolve_forward_args

    fixture = json.loads((merlin_dir() / "tests/data/movement_component_arguments.json").read_bytes())
    rows = fixture["arguments"]
    assert len(rows) == 23 and rows[0]["input_index"] == 2
    arrays = [np.frombuffer(bytes.fromhex(row["words_hex"]), row["dtype"]).reshape(row["shape"]) for row in rows]
    bundle = _argument_bundle(tmp_path / "capture", rows, arrays, kind)
    cache = ForwardArgsCache()
    for bound in (
        resolve_forward_args(bundle),
        resolve_forward_args(bundle, cache=cache),
        Q._native_host_arguments(bundle),
    ):
        for value, original, row in zip(bound, arrays, rows, strict=True):
            assert value.shape == tuple(row["abi_shape"]) and value.dtype == np.dtype("float32")
            assert value.flags.c_contiguous and value.tobytes() == original.tobytes()
            assert value[..., 0].tobytes() == original.real.tobytes()
            assert value[..., 1].tobytes() == original.imag.tobytes()


@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_component_binding_preserves_special_words_and_strided_values(tmp_path, precision, shape):
    from merlin.runtime.dispatch_runtime import resolve_forward_args

    words = (
        [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001]
        if precision == 32
        else [0, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000001, 0x7FF0000000000001]
    )
    count = int(np.prod(shape))
    original = (
        np.resize(np.array(words, dtype=f"uint{precision}"), 2 * count).view(f"complex{2 * precision}").reshape(shape)
    )
    if len(shape) == 2:
        original = original.T[:, ::-1]  # logical order differs from the underlying storage
    row = {"abi_shape": [*original.shape, 2], "abi_element": f"f{precision}"}
    bundle = _argument_bundle(tmp_path / "capture", [row], [original])
    for bound in (resolve_forward_args(bundle), Q._native_host_arguments(bundle)):
        (value,) = bound
        assert value.shape == tuple(row["abi_shape"]) and value.dtype == np.dtype(f"float{precision}")
        assert value.flags.c_contiguous and value.tobytes() == original.tobytes()
    # NPZ materializes strides. Also exercise the shared binder directly with the strided view.
    from merlin.runtime.dispatch_arguments import bind_forward_array
    from merlin.runtime.dispatch_runtime import _np_dtype, f32_to_bf16

    value = bind_forward_array(
        original, row["abi_shape"], row["abi_element"], _np_dtype=_np_dtype, f32_to_bf16=f32_to_bf16
    )
    assert value.flags.c_contiguous and value.tobytes() == original.tobytes()


@pytest.mark.parametrize(
    "shape,element",
    [
        ([2, 3], "f32"),
        ([3, 2, 2], "f32"),
        ([12], "f32"),
        ([2, 3, 1], "f32"),
        ([2, 3, 2], "f64"),
        ([2, 3, 2], "f16"),
        ([2, 3, 2], "bf16"),
        ([2, 3, 2], "i32"),
    ],
)
@pytest.mark.parametrize("kind", ["input", "buffer", "lifted"])
def test_complex_binding_refuses_shape_and_precision_guesses(tmp_path, shape, element, kind):
    import warnings

    from merlin.runtime.dispatch_placement import DispatchRuntimeError
    from merlin.runtime.dispatch_runtime import resolve_forward_args

    bundle = _argument_bundle(
        tmp_path / "capture", [{"abi_shape": shape, "abi_element": element}], [np.ones((2, 3), np.complex64)], kind
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # refusal must precede any lossy complex cast
        for resolve in (resolve_forward_args, Q._native_host_arguments):
            with pytest.raises(DispatchRuntimeError, match="component ABI shape and precision"):
                resolve(bundle)


def test_real_saved_movement_capture_arguments_match_the_complete_w3_abi():
    import os

    capture = os.environ.get("MERLIN_TEST_MOVEMENT_CAPTURE")
    if not capture:
        pytest.skip("set MERLIN_TEST_MOVEMENT_CAPTURE to the saved public movement capture")
    bundle = Q.Path(capture)
    inputs, _ = Q._native_host_signature(bundle / "model.mlir")
    bound = Q._native_host_arguments(bundle)
    assert len(bound) == len(inputs) == 183
    with np.load(bundle / "inputs.npz", allow_pickle=False) as archive:
        complex_count = 0
        for index, (value, (shape, dtype)) in enumerate(zip(bound, inputs, strict=True)):
            assert list(value.shape) == shape and str(value.dtype) == Q._NATIVE_HOST_STORAGE[dtype]
            assert value.flags.c_contiguous
            original = archive[f"in{index}"]
            if np.iscomplexobj(original):
                complex_count += 1
                assert value.tobytes() == original.tobytes()
                assert value[..., 0].tobytes() == original.real.tobytes()
                assert value[..., 1].tobytes() == original.imag.tobytes()
        assert complex_count == 23
