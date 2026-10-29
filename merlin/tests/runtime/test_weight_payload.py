"""Actual bounded weight bytes, packing, mutation barriers and ordinary generation."""

import hashlib
import json
import os
import struct
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower import blob_weights, c_runtime
from merlin.llvmlower.weight_payload import CHUNK_BYTES, WeightPayload


@pytest.mark.parametrize("dtype,shape", [("int8", (3, 7)), ("float16", (5, 2)), ("int64", (2, 3))])
def test_slicing_preserves_file_array_zero_tails_and_partial_overlap(tmp_path, dtype, shape):
    raw = bytes(range(71))
    path = tmp_path / "weight.bin"
    path.write_bytes(raw)
    array = np.arange(np.prod(shape), dtype=dtype).reshape(shape, order="F")
    source = WeightPayload.file(path, 3, 67)
    source.extend(WeightPayload.zeros(9))
    source.extend(WeightPayload.array(array.nbytes, lambda: array))
    reference = raw[3:70] + bytes(9) + array.tobytes(order="C")
    assert bytes(source) == reference
    for begin, end in ((0, 0), (1, 68), (65, len(source)), (67, 76), (len(source) - 1, len(source))):
        chunks = list(source[begin:end].chunks(7))
        assert all(0 < len(x) <= 7 for x in chunks)
        assert b"".join(chunks) == reference[begin:end]
    selected = source[4:19]
    selected.extend(source[10:24])
    assert bytes(selected) == reference[4:19] + reference[10:24]


@pytest.mark.parametrize("change", ["before", "during", "replace", "truncate"])
def test_changed_file_never_completes_as_an_unchanged_borrowed_source(tmp_path, change):
    path = tmp_path / "weight.bin"
    path.write_bytes(bytes(range(79)))
    source = WeightPayload.file(path, 3, 73)
    before = path.stat()
    stream = source.chunks(7)
    if change != "before":
        assert bytes(next(stream)) == bytes(range(3, 10))
    if change == "replace":
        replacement = path.with_suffix(".new")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    else:
        path.write_bytes(b"x" * (6 if change == "truncate" else before.st_size))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="changed|truncated"):
        list(stream)


def test_bad_ranges_budgets_and_array_extent_refuse(tmp_path):
    path = tmp_path / "weight.bin"
    path.write_bytes(b"123")
    with pytest.raises(ValueError, match="exceeds"):
        WeightPayload.file(path, 1, 3)
    source = WeightPayload.file(path, 0, 3)
    for selection in (slice(-1, 2), slice(0, 4), slice(2, 1), slice(None, None, 2)):
        with pytest.raises(ValueError):
            source[selection]
    for budget in (0, -1, True):
        with pytest.raises(ValueError):
            list(source.chunks(budget))
    with pytest.raises(ValueError, match="disagrees"):
        bytes(WeightPayload.array(5, lambda: np.arange(3, dtype=np.int8)))


@pytest.mark.parametrize("chunks", [(b"123",), (b"12345",), (b"", b"1234")])
def test_faulty_reader_cannot_report_a_complete_span(chunks):
    payload = WeightPayload()._add(4, lambda *args: iter(chunks))
    with pytest.raises(ValueError, match="complete span|exceeded|invalid"):
        bytes(payload)


def test_sparse_source_streaming_does_not_allocate_its_complete_payload(tmp_path):
    path = tmp_path / "large.bin"
    size = 32 * CHUNK_BYTES + 3
    with path.open("wb") as stream:
        stream.truncate(size)
    source = WeightPayload.file(path, 1, size - 2)
    tracemalloc.start()
    try:
        digest = hashlib.sha256()
        count = 0
        for chunk in source.chunks():
            count += len(chunk)
            digest.update(chunk)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == size - 2 and peak < 4 * CHUNK_BYTES
    assert digest.digest() == hashlib.sha256(bytes(size - 2)).digest()


def test_compaction_uses_same_tied_weight_and_dead_descriptor_rules(tmp_path):
    path = tmp_path / "weights.bin"
    raw = bytes(range(33))
    path.write_bytes(raw)
    header = {name: {"dtype": "I8", "shape": [end - begin], "data_offsets": [begin, end]}
              for name, begin, end in (("a", 0, 7), ("b", 7, 20), ("c", 20, 33))}
    manifest = {"0": {"weight": "a"}, "1": {"weight": "b"}, "2": {"weight": "c"}, "3": {"weight": "c"}}
    expected_header, expected_bytes = blob_weights.compact(header, raw, manifest, frozenset({1, 2}))
    selected_header, selected = blob_weights.compact(
        header, WeightPayload.file(path, 0, len(raw)), manifest, frozenset({1, 2}))
    assert selected_header == expected_header and bytes(selected) == expected_bytes
    assert selected_header["b"]["data_offsets"] == [0, 0]


def _capture(tmp_path):
    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text('''module {
      func.func @forward(%w: tensor<3x5xi8>, %x: tensor<2x3xf32>) -> tensor<2x3xf32> {
        return %x : tensor<2x3xf32>
      }
    }''')
    header = json.dumps({"w": {"dtype": "I8", "shape": [3, 5], "data_offsets": [0, 15]}}).encode()
    (model / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(range(15)))
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": "w"}, "1": {"kind": "input", "name": "x"},
    }))
    np.savez(model / "inputs.npz", in0=np.arange(6, dtype=np.float32).reshape(2, 3))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "blob_weights.json").write_text(json.dumps({
        "args": [{"key": "tail", "shape": [2, 7], "dtype": "i16"}], "dead_args": [],
    }))
    np.savez(prepared / "blob_weights.npz", tail=np.arange(14, dtype=np.int16).reshape(2, 7, order="F"))
    return model, prepared


@pytest.mark.parametrize("materialize", [False, True])
def test_ordinary_cgen_storage_choices_emit_identical_bytes_and_headers(tmp_path, monkeypatch, materialize):
    model, prepared = _capture(tmp_path)
    original = (model / "weights.safetensors").read_bytes()
    # The generated append begins at an absolute blob boundary. Padding only
    # the appended suffix left this i16 payload at the odd byte offset 15.
    expected = (bytes(range(15)) + bytes(c_runtime.GENERATED_WEIGHT_ALIGNMENT - 15)
                + np.arange(14, dtype=np.int16).reshape(2, 7, order="F").tobytes(order="C"))
    reports = []
    for choice in ("materialized", "streamed"):
        monkeypatch.setenv("MERLIN_WEIGHT_PAYLOAD_STORAGE", choice)
        out = tmp_path / choice
        reports.append(c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared,
                                          materialize_weights=materialize))
        assert (out / "weights.bin").exists() == materialize
        if materialize:
            assert (out / "weights.bin").read_bytes() == expected
    assert reports[0] == reports[1]
    assert reports[0]["weights_sha256"] == hashlib.sha256(expected).hexdigest()
    assert reports[0]["weights_bytes"] == len(expected)
    for name in ("model_io.h", "model_gen.h", "model_call.c"):
        assert (tmp_path / "materialized" / name).read_bytes() == (tmp_path / "streamed" / name).read_bytes()
    assert (model / "weights.safetensors").read_bytes() == original


def test_bad_streamed_chunk_is_visible_in_real_output_instead_of_hidden_by_plan(tmp_path, monkeypatch):
    model, prepared = _capture(tmp_path)
    out = tmp_path / "generated"
    original = WeightPayload.chunks

    def wrong(self, *args, **kwargs):
        for data in original(self, *args, **kwargs):
            yield bytes([data[0] ^ 1]) + bytes(data[1:])

    monkeypatch.setenv("MERLIN_WEIGHT_PAYLOAD_STORAGE", "streamed")
    monkeypatch.setattr(WeightPayload, "chunks", wrong)
    info = c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared)
    actual = (out / "weights.bin").read_bytes()
    expected = (bytes(range(15)) + bytes(c_runtime.GENERATED_WEIGHT_ALIGNMENT - 15)
                + np.arange(14, dtype=np.int16).reshape(2, 7, order="F").tobytes(order="C"))
    assert info["weights_sha256"] == hashlib.sha256(actual).hexdigest()
    assert actual != expected  # Independent full-byte reference detects the faulty implementation.


def test_invalid_prepared_array_does_not_publish_replacement_blob(tmp_path):
    model, prepared = _capture(tmp_path)
    out = tmp_path / "generated"
    c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared)
    original = (out / "weights.bin").read_bytes()
    np.savez(prepared / "blob_weights.npz", tail=np.arange(14, dtype=np.int8).reshape(2, 7))
    with pytest.raises(ValueError, match="disagree"):
        c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared)
    assert (out / "weights.bin").read_bytes() == original


def test_corrupted_actual_write_is_rejected_before_atomic_publication(tmp_path, monkeypatch):
    model, prepared = _capture(tmp_path)
    out = tmp_path / "generated"
    c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared)
    before = (out / "weights.bin").read_bytes()
    original = Path.open

    class BrokenWrite:
        def __init__(self, path, stream):
            self.path, self.stream = path, stream

        def __enter__(self):
            return self.stream.__enter__()

        def __exit__(self, *args):
            result = self.stream.__exit__(*args)
            with original(self.path, "r+b") as output:
                value = output.read(1)
                output.seek(0)
                output.write(bytes([value[0] ^ 1]))
            return result

    def opening(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if path.name == "weights.bin" and path.parent.name.startswith("weights-") and mode == "wb":
            return BrokenWrite(path, stream)
        return stream

    monkeypatch.setattr(Path, "open", opening)
    with pytest.raises(ValueError, match="realized weight blob"):
        c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared)
    assert (out / "weights.bin").read_bytes() == before


def test_unknown_storage_control_refuses_before_writing(tmp_path, monkeypatch):
    model, prepared = _capture(tmp_path)
    monkeypatch.setenv("MERLIN_WEIGHT_PAYLOAD_STORAGE", "unknown")
    with pytest.raises(ValueError, match="STORAGE"):
        c_runtime.generate(model, tmp_path / "out", model / "inputs.npz", prepared_dir=prepared)
    assert not (tmp_path / "out").exists()
