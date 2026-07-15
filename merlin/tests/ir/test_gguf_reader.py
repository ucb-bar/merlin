"""Tests for the GGUF reader core (merlin.frontends.gguf_reader).

Synthesizes a tiny GGUF with GGUFWriter (no download) covering an F32 tensor and a Q8_0 tensor, then
checks metadata parsing, quant-format classification, and the dequantization reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.frontends import gguf_reader as gr

try:
    _gguf = gr._gguf()
except Exception:  # pragma: no cover - vendored gguf-py absent
    _gguf = None

pytestmark = pytest.mark.skipif(_gguf is None, reason="vendored gguf-py unavailable")


def _write_tiny_gguf(path):
    from gguf import GGMLQuantizationType as T
    from gguf import GGUFWriter, quants

    w = GGUFWriter(str(path), "llama")
    w.add_uint32("llama.block_count", 2)
    w.add_uint32("llama.embedding_length", 64)
    w.add_uint32("llama.attention.head_count", 4)
    w.add_uint32("llama.attention.head_count_kv", 2)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)

    embd = np.arange(64 * 8, dtype=np.float32).reshape(8, 64)
    w.add_tensor("token_embd.weight", embd)

    wq = np.random.RandomState(0).randn(4, 64).astype(np.float32)
    q = quants.quantize(wq, T.Q8_0)
    # For a pre-quantized tensor, pass the packed byte-array; gguf derives the logical shape from it.
    w.add_tensor("blk.0.attn_q.weight", q, raw_dtype=T.Q8_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return wq


def test_read_metadata_and_classification(tmp_path):
    ref_q = _write_tiny_gguf(tmp_path / "tiny.gguf")
    m = gr.read(tmp_path / "tiny.gguf")

    assert m.arch == "llama"
    assert m.metadata["block_count"] == 2
    assert m.metadata["embedding_length"] == 64
    assert m.metadata["head_count"] == 4 and m.metadata["head_count_kv"] == 2
    assert m.metadata["rms_eps"] == pytest.approx(1e-5, rel=1e-3)

    embd = m.tensor("token_embd.weight")
    assert embd is not None and embd.ggml_type == "F32"
    assert embd.fmt.name == "fp32" and not embd.is_quantized

    q = m.tensor("blk.0.attn_q.weight")
    assert q is not None and q.ggml_type == "Q8_0"
    assert q.fmt.name == "gguf_q8_0" and q.is_quantized

    assert m.quant_histogram() == {"F32": 1, "Q8_0": 1}
    assert m.unsupported() == []


def test_dequantize_reference_roundtrips(tmp_path):
    ref_q = _write_tiny_gguf(tmp_path / "tiny.gguf")
    m = gr.read(tmp_path / "tiny.gguf")
    q = m.tensor("blk.0.attn_q.weight")
    deq = q.dequantize()
    assert deq.shape == ref_q.shape
    cos = float((ref_q.reshape(-1) @ deq.reshape(-1)) / (np.linalg.norm(ref_q) * np.linalg.norm(deq)))
    assert cos > 0.999   # Q8_0 round-trip fidelity


def test_unsupported_types_reported_not_crashed(tmp_path):
    # A scalar F16 tensor classifies; an exotic type (if present) would land in unsupported() — here
    # we just confirm F16 maps and nothing raises on a mixed file.
    from gguf import GGUFWriter

    w = GGUFWriter(str(tmp_path / "f16.gguf"), "llama")
    w.add_string("general.architecture", "llama")
    w.add_tensor("output_norm.weight", np.ones(32, dtype=np.float16))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    m = gr.read(tmp_path / "f16.gguf")
    t = m.tensor("output_norm.weight")
    assert t.ggml_type == "F16" and t.fmt.name == "fp16"
