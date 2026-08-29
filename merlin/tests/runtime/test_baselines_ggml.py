"""Board-free unit tests for the ggml / llama.cpp baseline arm (merlin.baselines.ggml).

These never touch a board or require llama.cpp to be built: they exercise the RVV-audit
symbol->region mapping, the compute-kernel classifier, llama-bench output parsing, the VLA
out-of-scope + non-convertible-model honesty gaps, and the not_run_is_not_pass contract — so the
gate stays green regardless of the ggml build state.
"""
from __future__ import annotations

import pytest

from merlin.baselines import ggml


# --- symbol -> region mapping -----------------------------------------------------------------

def _needs_correctness_bundle(model: str):
    """Skip unless THIS machine holds a capture bundle the ggml comparison can be graded against.

    The bundles are multi-GB and deliberately untracked, so which variant a checkout has is a
    property of the machine. `_CORRECTNESS_BUNDLE` names the variant whose golden a ggml GGUF forward
    can actually reproduce -- int8 for tiny_llama -- and a checkout holding only the fp32 capture
    resolves to None. The comparison then honestly reports UNCOMPARABLE, and asserting against that
    tests the absence of a dataset rather than the comparator. Widening the accepted variants here
    would instead ASSERT that a ggml forward reproduces the fp32 golden, which is a claim about
    numerics nobody has measured.
    """
    from merlin.baselines import ggml as _g
    return pytest.mark.skipif(_g._correctness_bundle(model) is None,
                              reason=f"no ggml-reproducible capture bundle for {model} on this machine")


def test_region_of_symbol():
    assert ggml._region_of_symbol("ggml_gemm_q4_K_16x1_q8_K") == "gemm"
    assert ggml._region_of_symbol("ggml_gemv_iq4_nl_16x1_q8_0") == "gemm"
    assert ggml._region_of_symbol("ggml_vec_dot_q8_K") == "gemm"
    assert ggml._region_of_symbol("ggml_compute_forward_flash_attn_ext_tiled") == "attention"
    assert ggml._region_of_symbol("ggml_compute_forward_rms_norm_f32") == "norm"
    assert ggml._region_of_symbol("ggml_compute_forward_silu_f32") == "elementwise"
    assert ggml._region_of_symbol("ggml_backend_cpu_buffer_type") == "other"


def test_is_kernel():
    assert ggml._is_kernel("ggml_gemm_q4_K_16x1_q8_K")
    assert ggml._is_kernel("ggml_vec_dot_iq2_s_q8_K")
    assert ggml._is_kernel("ggml_compute_forward_add")
    assert not ggml._is_kernel("ggml_backend_cpu_buffer_type")
    assert not ggml._is_kernel("std::vector")


# --- RVV audit: kernel-only + active-quant coverage -------------------------------------------

def test_audit_cpu_so_separates_kernel_and_active_quant(tmp_path, monkeypatch):
    # Synthetic .so: a vectorized q4_K GEMM inner kernel, a scalar q4_K vec_dot (fallback), and a
    # non-kernel dispatch symbol (excluded). Verifies whole-so vs kernel vs active-quant coverage
    # are each reported and that the fully-scalar q4_K vec_dot is labeled as a fallback.
    disasm = (
        "0000000000010000 <ggml_gemm_q4_K_16x1_q8_K>:\n"
        "   10000:\t02008557          \tvsetvli\ta0,a1,e32,m1,ta,ma\n"
        "   10004:\t0205f007          \tvfmacc.vv\tv8,v0,v4\n"
        "   10008:\t00b50533          \tadd\ta0,a0,a1\n"
        "   1000c:\t00008067          \tret\n"
        "0000000000010100 <ggml_vec_dot_q4_K_q8_K>:\n"      # active-quant but fully scalar
        "   10100:\t00b50533          \tadd\ta0,a0,a1\n"
        "   10104:\t02c58533          \tmul\ta0,a1,a2\n"
        "   10108:\t00008067          \tret\n"
        "0000000000010200 <ggml_backend_cpu_buffer_type>:\n"  # non-kernel dispatch (ignored)
        "   10200:\t00b50533          \tadd\ta0,a0,a1\n"
        "   10204:\t00008067          \tret\n"
    )
    import merlin.baselines.rvv_audit as ra
    monkeypatch.setattr(ra, "audit_binary", lambda p, **k: ra.classify_disasm(disasm))

    aud = ggml.audit_cpu_so(tmp_path / "fake-cpu.so")
    # kernel coverage counts only the two ggml_gemm/vec_dot kernels: 1 vector / (1+1+2) compute.
    assert aud.coverage_kernels is not None
    # active-quant coverage over q4_K/q8_K inner kernels: 1 vector / (1 vector + 3 scalar) = 0.25.
    assert aud.coverage_active_quant is not None
    assert 0.0 < aud.coverage_active_quant <= 1.0
    # the fully-scalar q4_K vec_dot is labeled as a scalar fallback in the gemm bucket.
    fbs = {f.symbol: f.region for f in aud.fallbacks}
    assert "ggml_vec_dot_q4_K_q8_K" in fbs
    assert fbs["ggml_vec_dot_q4_K_q8_K"] == "gemm"
    # gemm region coverage is present.
    assert "gemm" in aud.region_coverage
    # int8 (Q8_0 = q8_0+q8_K) and ternary (TQ2_0) per-quant path fields exist.
    assert hasattr(aud, "coverage_int8")
    assert hasattr(aud, "coverage_ternary")


def test_quant_path_coverage_selects_right_kernels():
    # Synthetic .so: a mixed q8_0 int8 dot + a fully-vector tq2_0 ternary dot. The int8 path
    # (q8_0+q8_K) and ternary path (tq2_0) must be scored independently, and an absent quant -> None.
    import merlin.baselines.rvv_audit as ra
    disasm = (
        "0000000000010000 <ggml_vec_dot_q8_0_q8_K>:\n"
        "   10000:\t02008557          \tvsetvli\ta0,a1,e32,m1,ta,ma\n"
        "   10004:\t0205f007          \tvfmacc.vv\tv8,v0,v4\n"
        "   10008:\t00b50533          \tadd\ta0,a0,a1\n"
        "   1000c:\t00008067          \tret\n"
        "0000000000010100 <ggml_vec_dot_tq2_0_q8_K>:\n"
        "   10100:\t02008557          \tvsetvli\ta0,a1,e32,m1,ta,ma\n"
        "   10104:\t0205f007          \tvmul.vv\tv8,v0,v4\n"
        "   10108:\t00008067          \tret\n"
    )
    rep = ra.classify_disasm(disasm)
    cov_int8 = ggml._quant_path_coverage(rep, "Q8_0")
    cov_tern = ggml._quant_path_coverage(rep, "tq2_0")
    assert cov_int8 is not None and 0.0 < cov_int8 <= 1.0
    assert cov_tern == 1.0                                    # tq2_0 dot fully vector here
    # tq1_0 kernels are absent from this dump -> None (not fabricated). (q4_K would match via the
    # shared q8_K activation dot, which is the CORRECT semantics, so we probe an absent quant.)
    assert ggml._quant_path_coverage(rep, "tq1_0") is None


def test_default_quants_int8_first():
    # int8 (Q8_0) must lead so it's the headline E2E the coordinator asked for.
    assert ggml.DEFAULT_QUANTS[0] == "Q8_0"
    assert "q4_K_M" in ggml.DEFAULT_QUANTS


# --- llama-bench output parsing ---------------------------------------------------------------

def test_parse_llama_bench_extracts_tps():
    # A representative llama-bench markdown table (pp = prompt/prefill, tg = token-gen).
    out = (
        "| model            |     size | test  |            t/s |\n"
        "| ---------------- | -------: | ----- | -------------: |\n"
        "| llama 1B Q4_K    | 636 MiB  | pp64  |  12.34 ± 0.05  |\n"
        "| llama 1B Q4_K    | 636 MiB  | tg32  |   3.21 ± 0.02  |\n"
    )
    tps = ggml._parse_llama_bench(out)
    assert tps.get("pp64") == 12.34
    assert tps.get("tg32") == 3.21


# --- honesty gaps: VLA out-of-scope + non-convertible LLMs -------------------------------------

def test_vla_models_are_out_of_scope_not_built():
    # Each VLA carries a PRECISE per-model reason (backbone-arch support vs captured-forward
    # reproducibility), not a blanket "out of scope". The two whose backbone IS a supported arch
    # (openvla=Llama-2, molmoact=Qwen2-style) say so explicitly; the rest are diffusion/action heads.
    for m in ("openvla", "rdt", "rdt2", "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla"):
        r = ggml.run_model(m, "fp32", write=False, run_board=False)
        assert r.status() == "not_built"
        assert r.gap_reason and len(r.gap_reason) > 40   # a specific, non-empty reason
        r.validate()
    # backbone-supported cases name the supported arch explicitly (honest scope boundary)
    assert "Llama-2" in ggml.run_model("openvla", "fp32", write=False, run_board=False).gap_reason
    assert "Qwen2" in ggml.run_model("molmoact", "fp32", write=False, run_board=False).gap_reason
    # diffusion/flow cases say they are not a causal LM
    for m in ("rdt", "rdt2", "xr0", "pi05", "smolvla", "groot_n1d7"):
        assert "causal LM" in ggml.run_model(m, "fp32", write=False, run_board=False).gap_reason


def test_small_llama_gguf_built_directly(monkeypatch):
    # small_llama's op surface IS a Llama block, so we build its GGUF DIRECTLY from the capture
    # bundle (gguf-py, HF-permuted Q/K). It is therefore `built`; off-board it stops at `not_run`
    # (board skipped) with the board-unavailable reason — never a fabricated pass.
    monkeypatch.setattr(ggml, "ggml_available", lambda: True)
    monkeypatch.setattr(ggml, "ggml_cpu_so", lambda: None)  # skip the audit path cleanly
    r = ggml.run_model("small_llama", "fp32", write=False, run_board=False)
    assert r.built is True
    assert r.status() == "not_run"          # built, but board skipped off-board
    assert "gguf-py llama-arch build" in r.notes and "HF-permuted Q/K" in r.notes
    assert r.gap_reason                      # board-unavailable reason present
    r.validate()


def test_small_llama_gguf_writer_is_llama_arch():
    # The direct GGUF builder emits a valid `llama`-arch GGUF with the right hparams + none-vocab.
    import sys
    p = ggml.build_small_llama_gguf()
    assert p.is_file() and p.stat().st_size > 0
    sys.path.insert(0, str(ggml._LLAMA_SRC / "gguf-py"))
    import gguf  # noqa: PLC0415
    r = gguf.GGUFReader(str(p))
    kv = {f.name: f for f in r.fields.values()}
    assert kv["general.architecture"].contents() == "llama"
    assert kv["llama.embedding_length"].contents() == 128
    assert kv["llama.attention.head_count"].contents() == 4
    assert kv["tokenizer.ggml.model"].contents() == "none"
    names = {t.name for t in r.tensors}
    assert "token_embd.weight" in names and "output.weight" in names
    assert "blk.0.ffn_gate.weight" in names  # SwiGLU present


def test_bitvla_inputs_embeds_out_of_gguf_shape(monkeypatch):
    monkeypatch.setattr(ggml, "ggml_available", lambda: True)
    monkeypatch.setattr(ggml, "ggml_cpu_so", lambda: None)
    r = ggml.run_model("bitvla", "fp32", write=False, run_board=False)
    assert r.status() == "not_built"
    assert "inputs_embeds" in r.gap_reason or "no llama.cpp arch" in r.gap_reason
    r.validate()


def test_ggml_not_built_when_toolchain_absent(monkeypatch):
    monkeypatch.setattr(ggml, "ggml_available", lambda: False)
    r = ggml.run_model("tiny_llama", "fp32", write=False, run_board=False)
    assert r.status() == "not_built"
    assert r.gap_reason
    r.validate()


# --- correctness gate: comparable for tiny_llama, uncomparable elsewhere ----------------------

@_needs_correctness_bundle("tiny_llama")
def test_tiny_llama_offboard_cos_none_board_gated(monkeypatch, tmp_path):
    # tiny_llama now HAS a real-checkpoint correctness bundle, so the "uncomparable" note is NOT
    # emitted; but off-board (run_board=False) the logits-dump gate is skipped, so cos stays None
    # -> never a fabricated pass. The board branch is where cos gets filled.
    monkeypatch.setattr(ggml, "ggml_available", lambda: True)
    monkeypatch.setattr(ggml, "ggml_cpu_so", lambda: None)
    fake_gguf = tmp_path / "tiny_llama-f16.gguf"
    fake_gguf.write_bytes(b"\x00" * 16)
    monkeypatch.setattr(ggml, "convert_to_gguf", lambda m, **k: fake_gguf)
    r = ggml.run_model("tiny_llama", "fp32", write=False, run_board=False)
    assert r.cos is None
    assert r.passed is False
    assert "UNCOMPARABLE" not in r.notes   # a comparable bundle exists -> no uncomparable claim
    r.validate()



@_needs_correctness_bundle("tiny_llama")
def test_correctness_bundle_resolution():
    # tiny_llama resolves to the real full checkpoint (int8_full); small_llama resolves to its
    # fp32 capture (the GGUF is built from the SAME weights, so its golden IS reproducible); VLA
    # models have no ggml-reproducible golden.
    cb = ggml._correctness_bundle("tiny_llama")
    assert cb is not None and cb.golden.is_file() and "full" in cb.root.name
    sm = ggml._correctness_bundle("small_llama")
    assert sm is not None and sm.golden.is_file() and sm.inputs.is_file()
    assert ggml._correctness_bundle("openvla") is None


def test_compare_logits_to_golden(tmp_path):
    # A synthetic logits blob equal to golden -> cos ~1.0, rel ~0 (the comparator is sound).
    import struct

    import numpy as np
    gold = np.random.RandomState(0).randn(1, 4, 16).astype(np.float32)
    gp = tmp_path / "golden.npy"
    np.save(gp, gold)
    lp = tmp_path / "logits.f32"
    with open(lp, "wb") as f:
        f.write(struct.pack("ii", 16, 4))
        gold.astype(np.float32).ravel().tofile(f)
    cos, rel = ggml._compare_logits_to_golden(lp, gp)
    assert cos == pytest.approx(1.0, abs=1e-5)
    assert rel == pytest.approx(0.0, abs=1e-5)


def test_default_models_llm_subset_first():
    assert ggml.DEFAULT_MODELS[:3] == ("tiny_llama", "small_llama", "bitvla")
    assert set(ggml.VLA_OUT_OF_SCOPE) == {
        "openvla", "rdt", "rdt2", "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla"}
