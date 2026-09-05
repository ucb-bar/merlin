"""The independent-W8A8-reference generator keeps its fail-closed refusals.

The value of ``golden_w8a8.independent.npy`` is entirely in what the tool REFUSES to write. A
reference computed from weights that are not the bundle's own manufactures failures (or passes)
that have nothing to do with the datapath under test, and it is indistinguishable from a real
one once it is sitting in the bundle. These tests pin the refusals, the never-overwrite naming,
and the capture-environment layering that a wrong reference would otherwise come in through.
"""
from __future__ import annotations

import importlib.util
import json
import struct

import numpy as np
import pytest

from merlin.common.paths import repo_root

_SCRIPT = repo_root() / "build_tools" / "scripts" / "make_w8a8_independent_golden.py"


def _load():
    spec = importlib.util.spec_from_file_location("_indep_golden", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load()


def _write_safetensors(path, tensors: dict[str, np.ndarray]) -> None:
    """Minimal safetensors writer, so the reader is tested against bytes and not against itself."""
    tag = {np.dtype(np.int8): "I8", np.dtype(np.uint8): "U8", np.dtype(np.int32): "I32",
           np.dtype(np.int64): "I64", np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}
    header, blob, offset = {}, bytearray(), 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": tag[arr.dtype], "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blob += raw
        offset += len(raw)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + bytes(blob))


# --------------------------------------------------------------------- the container reader
def test_safetensors_reader_round_trips_bytes(tool, tmp_path):
    tensors = {"w.int_data": np.arange(-6, 6, dtype=np.int8).reshape(3, 4),
               "w.scale": np.array([0.5, 0.25, 0.125], dtype=np.float32),
               "ids": np.array([[1, 2, 3]], dtype=np.int64)}
    path = tmp_path / "weights.safetensors"
    _write_safetensors(path, tensors)
    got = tool.read_safetensors(path)
    assert set(got) == set(tensors)
    for name, arr in tensors.items():
        assert got[name].dtype == arr.dtype
        assert np.array_equal(got[name], arr)


# --------------------------------------------------------------------- the consistency gate
def test_gate_accepts_bit_identical_quantized_weights(tool):
    a = {"w.int_data": np.arange(-8, 8, dtype=np.int8),
         "w.scale": np.array([0.5], dtype=np.float32)}
    b = {"w.int_data": np.arange(-8, 8, dtype=np.int8),
         "w.scale": np.array([0.5], dtype=np.float32)}
    report = tool.quantized_weight_diff(a, b)
    assert report["ok"] is True
    assert report["n_quantized"] == 1
    assert report["n_mismatched"] == 0


def test_gate_refuses_when_one_quantized_element_differs(tool):
    a = {"w.int_data": np.arange(-8, 8, dtype=np.int8)}
    flipped = np.arange(-8, 8, dtype=np.int8)
    flipped[3] += 1                      # a single element: the smallest real disagreement
    report = tool.quantized_weight_diff(a, {"w.int_data": flipped})
    assert report["ok"] is False
    assert report["n_mismatched"] == 1
    assert report["mismatched_examples"] == ["w.int_data"]


def test_gate_refuses_when_no_integer_tensor_is_shared(tool):
    """An empty intersection is an UNMEASURED comparison, not agreement.

    This is the failure mode that silently converts the tool into ``make_w8a8_golden.py``: if a
    rename or a scheme change stopped the two captures from sharing any integer tensor, a gate
    written as "no mismatches" would pass every time and write a reference nothing had checked.
    """
    a = {"w.scale": np.array([0.5], dtype=np.float32)}
    b = {"w.scale": np.array([0.5], dtype=np.float32)}
    assert tool.quantized_weight_diff(a, b)["ok"] is False
    # ... and likewise when both sides have integer tensors but under different names.
    a2 = {"lhs.int_data": np.zeros(4, dtype=np.int8)}
    b2 = {"rhs.int_data": np.zeros(4, dtype=np.int8)}
    assert tool.quantized_weight_diff(a2, b2)["ok"] is False


def test_gate_refuses_on_shape_change_without_reading_past_the_end(tool):
    a = {"w.int_data": np.zeros((2, 4), dtype=np.int8)}
    b = {"w.int_data": np.zeros((4, 2), dtype=np.int8)}
    assert tool.quantized_weight_diff(a, b)["ok"] is False


# --------------------------------------------------------------------- never overwrite
def test_default_output_never_targets_the_shipped_golden(tool):
    """Board runs in flight are graded against ``golden_w8a8.npy``; the tool writes beside it."""
    assert tool.DEFAULT_OUT_NAME != "golden_w8a8.npy"
    assert tool.DEFAULT_OUT_NAME.endswith(".npy")


def test_generate_writes_nothing_when_the_capture_interpreter_is_absent(tool, tmp_path,
                                                                       monkeypatch):
    bundle = tmp_path / "recaptures" / "somemodel_int8_full"
    bundle.mkdir(parents=True)
    monkeypatch.setattr(tool, "m2m_root", lambda: tmp_path / "absent_m2m")
    ok, message = tool.generate(bundle)
    assert ok is False
    assert "interpreter is absent" in message
    assert not (bundle / tool.DEFAULT_OUT_NAME).exists()


# --------------------------------------------------------------------- capture environment
def test_capture_environment_override_can_unset_a_smoke_default(tool, tmp_path, monkeypatch):
    """``capture.toml`` carries the SMOKE config for expensive models.

    gemma2_2b pins ``M2M_GEMMA_LAYERS=2`` while its ``_full`` bundle was captured with the
    variable unset (all 26 layers). Applying the file blindly builds a DIFFERENT model, so the
    override layer — including the empty value that unsets — is load-bearing, not cosmetic.
    """
    model_dir = tmp_path / "workloads" / "somemodel"
    model_dir.mkdir(parents=True)
    (model_dir / "capture.toml").write_text(
        '[env]\nM2M_LAYERS = "2"\nHF_HOME = "/cache"\n', encoding="utf-8")
    monkeypatch.delenv("M2M_LAYERS", raising=False)

    plain = tool.capture_environment(tmp_path, "somemodel")
    assert plain["M2M_LAYERS"] == "2" and plain["HF_HOME"] == "/cache"

    unset = tool.capture_environment(tmp_path, "somemodel", {"M2M_LAYERS": ""})
    assert "M2M_LAYERS" not in unset
    assert unset["HF_HOME"] == "/cache"        # untouched entries survive

    replaced = tool.capture_environment(tmp_path, "somemodel", {"M2M_LAYERS": "26"})
    assert replaced["M2M_LAYERS"] == "26"


def test_capture_python_falls_back_to_the_repo_venv(tool, tmp_path):
    """A workload with no ``capture.toml`` (small_llama) was captured under the shared venv."""
    (tmp_path / "workloads" / "somemodel").mkdir(parents=True)
    shared = tmp_path / ".venv" / "bin"
    shared.mkdir(parents=True)
    (shared / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    assert tool.capture_python(tmp_path, "somemodel") == shared / "python"


# --------------------------------------------------------------------- bundle naming
@pytest.mark.parametrize("bundle,model", [
    ("spectformer_int8_full", "spectformer"),
    ("small_llama_int8_consistent", "small_llama"),
    ("gemma2_2b_int8_full_seq8_sliced_tiledhead", "gemma2_2b"),
    ("lstmnetvit_int8_w8a8_consistent", "lstmnetvit"),
])
def test_bundle_model_name(tool, tmp_path, bundle, model):
    assert tool.bundle_model_name(tmp_path / bundle) == model


def test_bundle_model_name_rejects_a_non_int8_bundle(tool, tmp_path):
    with pytest.raises(ValueError):
        tool.bundle_model_name(tmp_path / "small_llama_fp32_consistent")


# --------------------------------------------------------------------- bundle inputs
def test_bundle_inputs_are_read_in_capture_order(tool, tmp_path):
    """``in10`` must sort after ``in2``; lexicographic order would silently permute arguments."""
    bundle = tmp_path / "b"
    bundle.mkdir()
    arrays = {f"in{i}": np.full((1,), i, dtype=np.float32) for i in range(12)}
    np.savez(bundle / "inputs.npz", **arrays)
    got = tool.load_bundle_inputs(bundle)
    assert [float(a[0]) for a in got] == list(range(12))
