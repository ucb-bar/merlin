"""Board-free unit tests for the ggml / llama.cpp baseline arm (merlin.baselines.ggml).

These never touch a board or require llama.cpp to be built: they exercise the RVV-audit
symbol->region mapping, the compute-kernel classifier, llama-bench output parsing, the VLA
out-of-scope + non-convertible-model honesty gaps, and the not_run_is_not_pass contract — so the
gate stays green regardless of the ggml build state.
"""
from __future__ import annotations

from merlin.baselines import ggml


# --- symbol -> region mapping -----------------------------------------------------------------

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
    for m in ("openvla", "rdt", "rdt2", "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla"):
        r = ggml.run_model(m, "fp32", write=False, run_board=False)
        assert r.status() == "not_built"
        assert "out of ggml scope" in r.gap_reason
        r.validate()


def test_small_llama_no_gguf_arch(monkeypatch):
    # Even with the toolchain "available", small_llama has no HF/GGUF arch -> honest not_built gap.
    monkeypatch.setattr(ggml, "ggml_available", lambda: True)
    monkeypatch.setattr(ggml, "ggml_cpu_so", lambda: None)  # skip the audit path cleanly
    r = ggml.run_model("small_llama", "fp32", write=False, run_board=False)
    assert r.status() == "not_built"
    assert "no HF/GGUF architecture" in r.gap_reason or "no GGUF" in r.gap_reason
    r.validate()


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


# --- correctness is uncomparable: never a fabricated pass -------------------------------------

def test_tiny_llama_correctness_uncomparable(monkeypatch, tmp_path):
    # With the toolchain available and a (mocked) GGUF, tiny_llama runs but correctness vs our
    # golden is uncomparable -> cos stays None, so it can NEVER be a pass (not_run_is_not_pass).
    monkeypatch.setattr(ggml, "ggml_available", lambda: True)
    monkeypatch.setattr(ggml, "ggml_cpu_so", lambda: None)
    fake_gguf = tmp_path / "tiny_llama-f16.gguf"
    fake_gguf.write_bytes(b"\x00" * 16)
    monkeypatch.setattr(ggml, "convert_to_gguf", lambda m, **k: fake_gguf)
    r = ggml.run_model("tiny_llama", "fp32", write=False, run_board=False)
    assert r.cos is None
    assert r.passed is False
    assert "UNCOMPARABLE" in r.notes
    r.validate()


def test_default_models_llm_subset_first():
    assert ggml.DEFAULT_MODELS[:3] == ("tiny_llama", "small_llama", "bitvla")
    assert set(ggml.VLA_OUT_OF_SCOPE) == {
        "openvla", "rdt", "rdt2", "molmoact", "groot_n1d7", "xr0", "pi05", "smolvla"}
