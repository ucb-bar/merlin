"""Actual generated ABI weight alignment, aliases and original capture bytes."""

import json
import struct

import numpy as np
import pytest

from merlin.llvmlower import c_runtime


def capture(path, *, overlapping=False, incompatible=False):
    path.mkdir()
    if overlapping:
        payload = b"abc" + np.array([17, -31], "<i4").tobytes() + b"z"
        declarations = [("whole", "12xi8", "I8", [12], 0, 12),
                        ("wide", "2xi32", "I32", [2], 3, 11),
                        ("alias", "2xi32", "I32", [2], 3 if not incompatible else 4,
                         11 if not incompatible else 12)]
    else:
        payload = b"abcdef" + np.array([17, -31], "<i4").tobytes() + np.array(1.25, "<f8").tobytes()
        declarations = [("a", "3xi8", "I8", [3], 0, 3), ("b", "3xi8", "I8", [3], 3, 6),
                        ("wide", "2xi32", "I32", [2], 6, 14), ("alias", "2xi32", "I32", [2], 6, 14),
                        ("buffer", "f64", "F64", [], 14, 22)]
    args = [f"%{name}:tensor<{typ}>" for name, typ, *_ in declarations]
    args.append("%stub:tensor<1xi64>")
    (path / "model.mlir").write_text(
        "module {func.func @forward(" + ",".join(args) + ")->tensor<2xi32> {return %wide:tensor<2xi32>}}"
    )
    header = {name: dict(dtype=dtype, shape=shape, data_offsets=[start, end])
              for name, _, dtype, shape, start, end in declarations}
    encoded = json.dumps(header).encode()
    (path / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    manifest = {str(index): dict(kind="buffer" if name == "buffer" else "param", weight=name)
                for index, (name, *_) in enumerate(declarations)}
    manifest[str(len(declarations))] = dict(kind="param", weight="absent", stub=True)
    (path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    np.savez(path / "inputs.npz")
    return payload, declarations


def weight_offsets(path):
    return [int(line.split(",")[1].strip().removesuffix("L"))
            for line in (path / "model_gen.h").read_text().splitlines() if "{MERLIN_WEIGHT," in line]


@pytest.mark.parametrize("storage", ["streamed", "materialized"])
@pytest.mark.parametrize("overlapping", [False, True])
def test_generated_weight_rows_are_aligned_and_preserve_bytes_and_aliases(tmp_path, monkeypatch, storage, overlapping):
    model, out = tmp_path / "capture", tmp_path / "generated"
    payload, declarations = capture(model, overlapping=overlapping)
    original = (model / "weights.safetensors").read_bytes()
    monkeypatch.setenv("MERLIN_WEIGHT_PAYLOAD_STORAGE", storage)
    info = c_runtime.generate(model, out, model / "inputs.npz")
    offsets, actual = weight_offsets(out), (out / "weights.bin").read_bytes()
    widths = [c_runtime.DT_BYTES["i32" if typ == "2xi32" else "f64" if typ == "f64" else "i8"]
              for _, typ, *_ in declarations] + [8]
    assert all(offset % width == 0 for offset, width in zip(offsets, widths, strict=True))
    for offset, (_, _, _, _, start, end) in zip(offsets, declarations):
        assert actual[offset:offset + end - start] == payload[start:end]
    assert actual[offsets[-1]:offsets[-1] + 8] == bytes(8)
    wide, alias = (1, 2) if overlapping else (2, 3)
    assert offsets[wide] == offsets[alias]
    if overlapping:
        # The byte view and typed subview remain one physical backing. A write
        # through either descriptor must still be observable through the other.
        assert offsets[wide] - offsets[0] == 3
        mutable = bytearray(actual)
        mutable[offsets[wide]] ^= 71
        assert mutable[offsets[0] + 3] == mutable[offsets[wide]]
    assert (model / "weights.safetensors").read_bytes() == original
    assert info["weights_bytes"] == len(actual)
    assert actual[:len(payload)] == payload
    hashed = c_runtime.generate(model, tmp_path / "hash_only", model / "inputs.npz", materialize_weights=False)
    assert hashed == info
    assert not (tmp_path / "hash_only/weights.bin").exists()


def test_incompatible_overlapping_alignment_refuses_instead_of_breaking_aliases(tmp_path):
    model = tmp_path / "capture"
    capture(model, overlapping=True, incompatible=True)
    with pytest.raises(ValueError, match="overlapping weight.*alignment"):
        c_runtime.generate(model, tmp_path / "generated", model / "inputs.npz")


def test_old_unaligned_blob_cannot_be_reused_as_current_derived_bytes(tmp_path):
    model, out = tmp_path / "capture", tmp_path / "generated"
    raw, _ = capture(model)
    c_runtime.generate(model, out, model / "inputs.npz")
    previous = (out / "weights.bin").read_bytes()
    old = tmp_path / "old.bin"
    old.write_bytes(raw + bytes(8))
    with pytest.raises(ValueError, match="differs from this stage's derived bytes"):
        c_runtime.generate(model, out, model / "inputs.npz", weights_source=old)
    assert (out / "weights.bin").read_bytes() == previous
