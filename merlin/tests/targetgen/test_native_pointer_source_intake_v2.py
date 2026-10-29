"""Prospective large storage stays bounded, complete and independently checked."""

import io
import json
import zipfile

import numpy as np
import pytest
import yaml
from test_native_pointer_source_binding import SOURCE, frozen  # noqa: F401

from merlin.runtime.tensor_payload import TensorPayload
from merlin.targetgen import golden_store as G
from merlin.targetgen import native_pointer_source_binding as P

SELECTED = dict(intake_protocol=P.LARGE_INTAKE_PROTOCOL)


def test_selected_storage_preserves_complete_independent_inputs(frozen):  # noqa: F811
    root, capture, cb, a, b = frozen
    inputs, record = P.bind_pointer_source(cb, root, capture, **SELECTED)
    assert record["protocol"] == P.PROTOCOL
    assert record["schema"] == "merlin.native_pointer_source_binding.v2"
    assert record["intake"]["logical_bytes"] == 28
    assert all(type(v) is TensorPayload for v in inputs.values())
    assert P.payload_matches_array(inputs["left"], a, shape=a.shape)
    assert P.payload_matches_array(inputs["right"], b, shape=b.shape)
    again, same = P.bind_pointer_source(cb, root, capture, **SELECTED)
    assert same == record and P.pointer_inputs_equal(inputs, again, **SELECTED)
    assert "intake" not in P.bind_pointer_source(cb, root, capture)[1]


@pytest.mark.parametrize("protocol", [True, "v2", {}, "merlin.native_pointer_source_intake.v3"])
def test_unknown_selection_refuses_before_source_loading(frozen, protocol):  # noqa: F811
    root, capture, cb, *_ = frozen
    with pytest.raises(ValueError, match="unknown prospective"):
        P.bind_pointer_source(cb, root, capture, intake_protocol=protocol)


def test_selected_input_corruption_and_source_substitution_refuse(frozen):  # noqa: F811
    root, capture, cb, a, b = frozen
    a[0, 1] += 1
    np.savez(capture / "inputs.npz", in0=a, in1=b)
    with pytest.raises(ValueError, match="independent values"):
        P.bind_pointer_source(cb, root, capture, **SELECTED)
    (capture / "model.mlir").write_text(SOURCE.replace("arith.addi", "arith.subi"))
    with pytest.raises(ValueError, match="interfaces differ"):
        P.bind_pointer_source(cb, root, capture, **SELECTED)


def test_native_i8_storage_requires_the_original_exact_typed_policy(frozen):  # noqa: F811
    root, capture, cb, a, _b = frozen
    text = """builtin.module { func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x2xi8>)
      -> tensor<2x3xi8> { func.return %a : tensor<2x3xi8> } }"""
    (root / "source.mlir").write_text(text)
    (capture / "model.mlir").write_text(text)
    cb["tensors"]["result"].update(dtype="i8", shape=[2, 3])
    golden = yaml.safe_load((root / "golden.yaml").read_text())
    golden["outputs"]["answer"] = a.tolist()
    (root / "golden.yaml").write_text(yaml.safe_dump(golden))
    with pytest.raises(ValueError, match="exact i32 output policy"):
        P.bind_pointer_source(cb, root, capture, **SELECTED)
    declaration = yaml.safe_load((root / "capsule.yaml").read_text())
    declaration["numeric_policy"]["dtype"] = "i8"
    (root / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    assert P.bind_pointer_source(cb, root, capture, **SELECTED)[1]["returns"][0]["dtype"] == "i8"
    with pytest.raises(ValueError, match="raw i32 outputs"):
        P.bind_pointer_source(cb, root, capture)


def _header_archive(path, shape, dtype="i8"):
    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(member, dict(descr=np.dtype(dtype).str, fortran_order=False, shape=shape))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("in0.npy", member.getvalue())


@pytest.mark.parametrize("shape", [(P.LARGE_MAX_ELEMENTS + 1,), (2**62, 8)])
def test_oversize_archive_header_refuses_before_allocation(frozen, monkeypatch, shape):  # noqa: F811
    root, capture, cb, *_ = frozen
    _header_archive(capture / "inputs.npz", shape)
    monkeypatch.setattr(G, "load_golden", lambda *_a, **_k: pytest.fail("reference allocation reached"))
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("array allocation reached"))
    with pytest.raises(ValueError, match="decoded array shape/storage bound"):
        P.bind_pointer_source(cb, root, capture, **SELECTED)


def test_truncated_archive_header_refuses_before_allocation(frozen, monkeypatch):  # noqa: F811
    root, capture, cb, *_ = frozen
    _header_archive(capture / "inputs.npz", (2, 3))
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("array allocation reached"))
    with pytest.raises(ValueError, match="complete decoded array payload bound"):
        P.bind_pointer_source(cb, root, capture, **SELECTED)


def test_large_namespace_and_working_estimate_refuse_before_allocation(frozen, monkeypatch):  # noqa: F811
    root, capture, cb, *_ = frozen
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("array allocation reached"))
    cb["tensors"]["result"]["shape"] = [9_000_000]
    with pytest.raises(ValueError, match="working storage"):
        P.bind_pointer_source(cb, root, capture, **SELECTED)
    with (root / "rogue").open("wb") as stream:
        stream.truncate(P.LARGE_MAX_PAYLOAD_BYTES + 1)
    with pytest.raises(ValueError, match="payload bound"):
        P.bind_pointer_source(cb, root, capture, **SELECTED)


def test_large_opaque_capture_payload_is_bounded_without_metadata_parsing(frozen):  # noqa: F811
    root, capture, cb, *_ = frozen
    with (root / "source-capture-golden.json").open("wb") as stream:
        stream.truncate((4 << 20) + 1)
    # Capture provenance may carry a full numerical payload; it is hashed and
    # bounded as opaque bytes, while the independent compact golden is parsed.
    assert P.preflight_large_intake(cb, root, capture, **SELECTED)["output_elements"] == 4
    with (root / "golden.yaml").open("wb") as stream:
        stream.truncate((4 << 20) + 1)
    with pytest.raises(ValueError, match="bounded compact format"):
        P.preflight_large_intake(cb, root, capture, **SELECTED)


@pytest.mark.parametrize("field,value", [("offset", 1), ("storage_shape", [100, 100]), ("strides", [1, 2])])
def test_hidden_or_changed_storage_is_rejected_in_preflight(frozen, field, value):  # noqa: F811
    root, capture, cb, *_ = frozen
    cb["tensors"]["left"][field] = value
    with pytest.raises(ValueError, match="exact dense logical storage"):
        P.preflight_large_intake(cb, root, capture, **SELECTED)


def _reference(array, key):
    return {G.REFERENCE: G.ARRAYS, "key": key, "dtype": array.dtype.str,
            "shape": list(array.shape), "sha256": G._digest(array)}


@pytest.mark.parametrize("m,k,n", [(12544, 64, 256), (12544, 256, 64)])
def test_actual_tall_storage_extents_use_compact_complete_transport(frozen, m, k, n):  # noqa: F811
    # All-zero independent inputs give an exact all-zero result without per-MAC test work.
    root, capture, cb, *_ = frozen
    text = SOURCE.replace("2x3", f"{m}x{k}").replace("3x2", f"{k}x{n}").replace("2x2", f"{m}x{n}")
    (root / "source.mlir").write_text(text)
    (capture / "model.mlir").write_text(text)
    arrays = dict(a=np.zeros((m, k), dtype=np.int8), b=np.zeros((k, n), dtype=np.int8),
                  out=np.zeros((m, n), dtype=np.int32))
    np.savez_compressed(capture / "inputs.npz", in0=arrays["a"], in1=arrays["b"])
    np.savez_compressed(root / "golden.npz", **arrays)
    declaration = yaml.safe_load((root / "capsule.yaml").read_text())
    declaration["inputs"][0]["shape"] = [m, k]
    declaration["inputs"][1]["shape"] = [k, n]
    (root / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    golden = yaml.safe_load((root / "golden.yaml").read_text())
    golden["outputs"]["answer"] = _reference(arrays["out"], "out")
    for name, key in [("I0", "a"), ("I1", "b")]:
        golden["oracle_provenance"]["inputs"][name] = dict(
            shape=list(arrays[key].shape), decoded=_reference(arrays[key], key))
    (root / "golden.yaml").write_text(yaml.safe_dump(golden))
    for name, shape in [("left", [m, k]), ("right", [k, n]), ("result", [m, n])]:
        cb["tensors"][name]["shape"] = shape
    with pytest.raises(ValueError, match="tensor element bound"):
        P.bind_pointer_source(cb, root, capture)
    inputs, record = P.bind_pointer_source(cb, root, capture, **SELECTED)
    assert record["intake"]["output_elements"] == m * n
    assert record["intake"]["working_bytes"] < P.LARGE_MAX_WORKING_BYTES
    assert P.payload_matches_array(inputs["left"], arrays["a"], shape=(m, k))
    compact = G.load_golden(root, compact_arrays=True)
    assert isinstance(compact["outputs"]["answer"], np.ndarray)
    assert not compact["outputs"]["answer"].flags.writeable
    report = P.compare_complete_integer_outputs(compact["outputs"], {"answer": arrays["out"]},
                                                dict(compare="exact_int"), golden_source="zero_fixture")
    assert report["status"] == "pass" and report["per_output"]["answer"]["n_elements"] == m * n


def test_complete_chunk_check_detects_last_cell_and_short_output():
    expected = np.zeros((P.CHUNK_ELEMENTS * 2 + 3,), dtype=np.int32)
    observed = expected.copy()
    observed[-1] = 1
    report = P.compare_complete_integer_outputs({"y": expected}, {"y": observed},
                                                dict(compare="exact_int"), golden_source="fixture")
    assert report["status"] == "fail" and report["mismatch_count"] == 1
    assert report["first_mismatch"]["index"] == len(expected) - 1
    assert json.loads(json.dumps(report)) == report
    report = P.compare_complete_integer_outputs({"y": expected}, {"y": observed[:-1]},
                                                dict(compare="exact_int"), golden_source="fixture")
    assert report["status"] == "fail"
    assert report["per_output"]["y"]["failure_class"] == "output_shape_mismatch"


def test_per_output_policy_cannot_omit_original_results():
    policy = dict(compare="per_output", schema="merlin.numeric_policy.per_output.v1",
                  outputs={"x": dict(compare="exact_int")})
    with pytest.raises(ValueError, match="complete original result policy roster"):
        P.compare_complete_integer_outputs({"x": [1], "y": [2]}, {"x": [1], "y": [2]},
                                           policy, golden_source="fixture")
