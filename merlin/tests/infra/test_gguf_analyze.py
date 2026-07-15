"""Tests for the GGUF capability probe (merlin.frontends.adapters.gguf.analyze).

Routes a GGUF checkpoint's quantized weights against a target's compute_units — a fast 'what can this
target run' probe built on the GGUF reader + the target-agnostic routing tooling. No download: a tiny
GGUF is synthesized with GGUFWriter.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.frontends import gguf_reader as gr
from merlin.frontends.adapters import gguf as gguf_adapter

try:
    _gguf = gr._gguf()
except Exception:  # pragma: no cover
    _gguf = None

pytestmark = pytest.mark.skipif(_gguf is None, reason="vendored gguf-py unavailable")


def _write(path):
    from gguf import GGMLQuantizationType as T
    from gguf import GGUFWriter, quants

    w = GGUFWriter(str(path), "llama")
    w.add_string("general.architecture", "llama")
    w.add_uint32("llama.block_count", 1)
    # two quantized matmul weights (Q8_0) + one unquantized norm (F32, skipped by weight_demands)
    for name in ("blk.0.attn_q.weight", "blk.0.ffn_gate.weight"):
        wq = np.random.RandomState(0).randn(4, 64).astype(np.float32)
        w.add_tensor(name, quants.quantize(wq, T.Q8_0), raw_dtype=T.Q8_0)
    w.add_tensor("output_norm.weight", np.ones(64, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def test_weight_demands_skip_norms(tmp_path):
    _write(tmp_path / "m.gguf")
    model = gr.read(tmp_path / "m.gguf")
    demands = gr.weight_demands(model)
    # the two Q8_0 matmul weights, not the F32 norm
    assert len(demands) == 2
    assert {d.in_fmt for d in demands} == {"gguf_q8_0"}


def test_analyze_reports_honest_gaps(tmp_path):
    _write(tmp_path / "m.gguf")
    # GGUF Q8_0 is a storage format neither target lists as a native compute dtype -> honest gap
    # (a dequant/convert step is required). The probe surfaces exactly that.
    for target in ("rvv", "gemmini_mx"):
        rep = gguf_adapter.analyze(tmp_path / "m.gguf", target=target)
        assert rep.arch == "llama"
        assert rep.n_weights == 2
        assert rep.quant_histogram.get("Q8_0") == 2
        assert rep.gaps.get("gguf_q8_0") == 2
        assert rep.routable == 0
        assert not rep.fully_routable
        assert rep.unsupported_types == []   # Q8_0 IS a known canonical format (just not target-native)
