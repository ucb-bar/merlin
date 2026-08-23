"""Offline bundle rewrites are value-preserving, and they say that they happened.

The second half is not bookkeeping. `gemma2_2b_int8_full_seq8_pretransposed` had its embedding table
specialized AND 183 weights physically pre-transposed, recorded neither, and still named the original
bundle's weights file -- so anything measured from it read as stock Gemma 2 2B.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from merlin.baselines import bundle_rewrite as br
from merlin.baselines.bundle import CaptureBundle

SOLE_USE = """module {
  func.func @forward(%0: tensor<4x8xi8>, %1: tensor<2x3xi8>) -> tensor<8x4xi8> {
    %2 = tensor.empty() : tensor<8x4xi8>
    %3 = linalg.transpose ins(%0:tensor<4x8xi8>) outs(%2:tensor<8x4xi8>) permutation = [1, 0]
    return %3 : tensor<8x4xi8>
  }
}
"""


def _write_safetensors(path, tensors: dict[str, np.ndarray]) -> None:
    header, off, blobs = {}, 0, []
    for name, a in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        header[name] = {"dtype": "I8", "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


def _read_safetensors(path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            s, e = spec["data_offsets"]
            f.seek(base + s)
            out[name] = np.frombuffer(f.read(e - s), dtype=np.int8).reshape(spec["shape"])
    return out


@pytest.fixture
def bundle(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.mlir").write_text(SOLE_USE)
    w = np.arange(32, dtype=np.int8).reshape(4, 8)
    other = np.arange(6, dtype=np.int8).reshape(2, 3)
    _write_safetensors(src / "weights.safetensors", {"w": w, "other": other})
    (src / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": "w", "dtype": "int8", "shape": [4, 8]},
        "1": {"kind": "param", "weight": "other", "dtype": "int8", "shape": [2, 3]},
    }))
    (src / "golden.npy").write_bytes(b"")
    return src, w, other


def test_the_weight_is_stored_transposed_and_the_transpose_is_gone(bundle, tmp_path):
    src, w, other = bundle
    dst = tmp_path / "dst"
    rec = br.hoist_weight_transposes(src, dst)

    got = _read_safetensors(dst / "weights.safetensors")
    assert np.array_equal(got["w"], w.T), "the stored weight must be exactly the transpose"
    assert np.array_equal(got["other"], other), "an untouched weight must be byte-identical"

    text = (dst / "model.mlir").read_text()
    assert "linalg.transpose" not in text, "the run-time transpose must be gone"
    assert "%0: tensor<8x4xi8>" in text, "the argument must carry the transposed type"

    man = json.loads((dst / "weights.safetensors.manifest.json").read_text())
    assert man["0"]["shape"] == [8, 4]
    assert man["1"]["shape"] == [2, 3], "an untouched weight's manifest entry must not move"
    assert rec.effect["weights_pre_transposed"] == 1
    assert rec.effect["bytes_moved_per_inference_before"] == 32
    assert rec.effect["bytes_moved_per_inference_after"] == 0


def test_the_rewrite_records_itself(bundle, tmp_path):
    src, _, _ = bundle
    dst = tmp_path / "dst"
    br.hoist_weight_transposes(src, dst)

    recs = br.read_rewrites(dst)
    assert len(recs) == 1
    assert recs[0].name == "hoist_weight_transposes"
    assert recs[0].source_bundle == "src", "the chain back to the capture must survive"
    assert "SOLE consumer" in recs[0].soundness
    assert CaptureBundle("m", "v", dst).rewrites.is_file()


def test_rewrites_chain_rather_than_replace(bundle, tmp_path):
    """A bundle rewritten twice must show BOTH, or the second erases the first's caveats."""
    src, _, _ = bundle
    br.record_rewrite(src, br.RewriteRecord(
        name="specialize_gather", source_bundle="orig", soundness="index tensor has one consumer",
        caveats=["valid for these 8 token ids ONLY"]))
    dst = tmp_path / "dst"
    br.hoist_weight_transposes(src, dst)

    names = [r.name for r in br.read_rewrites(dst)]
    assert names == ["specialize_gather", "hoist_weight_transposes"]
    assert "valid for these 8 token ids ONLY" in br.read_rewrites(dst)[0].caveats


def test_a_shared_weight_is_refused_not_silently_corrupted(tmp_path):
    """The soundness condition is the whole point: another reader would see transposed data."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.mlir").write_text("""module {
  func.func @forward(%0: tensor<4x8xi8>) -> (tensor<8x4xi8>, tensor<4x8xi8>) {
    %1 = tensor.empty() : tensor<8x4xi8>
    %2 = linalg.transpose ins(%0:tensor<4x8xi8>) outs(%1:tensor<8x4xi8>) permutation = [1, 0]
    %3 = tensor.empty() : tensor<4x8xi8>
    %4 = linalg.copy ins(%0 : tensor<4x8xi8>) outs(%3 : tensor<4x8xi8>) -> tensor<4x8xi8>
    return %2, %4 : tensor<8x4xi8>, tensor<4x8xi8>
  }
}
""")
    _write_safetensors(src / "weights.safetensors", {"w": np.arange(32, dtype=np.int8).reshape(4, 8)})
    (src / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": "w", "dtype": "int8", "shape": [4, 8]}}))
    with pytest.raises(ValueError, match="no hoistable"):
        br.hoist_weight_transposes(src, tmp_path / "dst")


def test_an_unrewritten_bundle_reports_no_rewrites(tmp_path):
    assert br.read_rewrites(tmp_path) == []


# --- the table type must be updated EVERYWHERE the value is typed, not just in the signature ------
# Retyping only the signature left `tensor.extract %0[...] : tensor<256000x2304xf32>` behind, i.e.
# IR that does not verify. Caught by comparing the applier's output byte-for-byte against the bundle
# it reproduces; kept here so it cannot come back.

GATHER_IR = """module {
  func.func @forward(%0: tensor<256000x2304xf32>, %1: tensor<1x8xi64>) -> tensor<1x8x2304xf32> {
    %2 = linalg.generic ins(%1 : tensor<1x8xi64>) {
      ^bb0(%id: i64, %out: f32):
        %665 = tensor.extract %0[%663, %664] : tensor<256000x2304xf32>
        linalg.yield %665 : f32
    }
    return %2 : tensor<1x8x2304xf32>
  }
}
"""


def test_retype_updates_every_occurrence_of_the_table_type():
    out = br._retype_arg(GATHER_IR, 0, [256000, 2304], [8, 2304], "f32")
    assert "tensor<256000x2304xf32>" not in out, "a stale table type leaves IR that will not verify"
    assert out.count("tensor<8x2304xf32>") == 2, "signature AND the extract must be retyped"


def test_retype_leaves_a_same_shaped_unrelated_value_alone():
    ir = GATHER_IR.replace(
        "    return %2", "    %9 = tensor.empty() : tensor<256000x2304xf32>\n    return %2")
    out = br._retype_arg(ir, 0, [256000, 2304], [8, 2304], "f32")
    assert "%9 = tensor.empty() : tensor<256000x2304xf32>" in out


def test_retype_refuses_rather_than_writing_unretyped_ir():
    with pytest.raises(ValueError, match="never appears with type"):
        br._retype_arg(GATHER_IR, 7, [256000, 2304], [8, 2304], "f32")


def test_token_boundaries_are_respected():
    assert br._mentions_ssa("foo %0[", "%0")
    assert br._mentions_ssa("%658 = x", "%658")
    assert not br._mentions_ssa("%6580 = x", "%658"), "%658 must not match %6580"
    assert br._mentions_ssa("a %658, %6580", "%658")
