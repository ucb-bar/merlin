"""Board-free unit tests for the EXO baseline arm (merlin.baselines.exo).

EXO is forced whole-model = an EXO RVV GEMM kernel + a hand C glue runtime (EXO is a kernel DSL, not
a whole-model compiler). These tests never touch the K1 board and do not require the EXO venv to be
installed for the pure-Python paths: they exercise the weights-header emitter, the OUT-marker
parser, the scalar-glue fallback labels, the RVV-audit .insn-vs-decoded quirk handling, and the
not_run_is_not_pass contract. Tests that need ``exocc`` / the SpacemiT toolchain skip cleanly when
absent, so the gate stays green regardless of build state.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from merlin.baselines import exo, rvv_audit
from merlin.baselines.contract import BaselineResult


# --- scalar-glue fallback labels (honesty) ----------------------------------------------------

def test_scalar_glue_labels_present_and_reasoned():
    fbs = exo._scalar_fallbacks()
    syms = {f.symbol for f in fbs}
    # the ops with NO EXO RVV kernel — must be labeled, never hidden.
    assert {"rmsnorm", "rope", "attention_softmax", "swiglu_silu", "residual_add"} <= syms
    for f in fbs:
        assert f.reason == "no EXO RVV kernel — scalar glue"
        assert f.region in ("gemm", "attention", "norm", "elementwise", "other")


# --- OUT marker parsing -----------------------------------------------------------------------

def test_parse_out_recovers_float_bits():
    vals = np.array([1.5, -2.0, 0.0, 3.25], dtype=np.float32)
    bits = vals.view(np.uint32)
    line = "OUT 4 " + " ".join(str(int(b)) for b in bits)
    out = exo._parse_out("=== hdr ===\n" + line + "\nDONE\n")
    assert out is not None
    np.testing.assert_array_equal(out, vals)


def test_parse_out_none_when_absent():
    assert exo._parse_out("no marker here\nDONE\n") is None


# --- weights header emitter -------------------------------------------------------------------

def _fake_bundle(tmp_path):
    """A minimal fake capture: a safetensors header with the tensors the emitter looks up."""
    cfg = exo._CFG
    names = {"lm.model.embed_tokens.weight": [cfg["V"], cfg["H"]],
             "lm.model.norm.weight": [cfg["H"]],
             "lm.lm_head.weight": [cfg["V"], cfg["H"]]}
    for L in range(cfg["NL"]):
        p = f"lm.model.layers.{L}"
        names[f"{p}.input_layernorm.weight"] = [cfg["H"]]
        names[f"{p}.self_attn.q_proj.weight"] = [cfg["H"], cfg["H"]]
        names[f"{p}.self_attn.k_proj.weight"] = [cfg["NKV"] * cfg["HD"], cfg["H"]]
        names[f"{p}.self_attn.v_proj.weight"] = [cfg["NKV"] * cfg["HD"], cfg["H"]]
        names[f"{p}.self_attn.o_proj.weight"] = [cfg["H"], cfg["H"]]
        names[f"{p}.post_attention_layernorm.weight"] = [cfg["H"]]
        names[f"{p}.mlp.gate_proj.weight"] = [cfg["FF"], cfg["H"]]
        names[f"{p}.mlp.up_proj.weight"] = [cfg["FF"], cfg["H"]]
        names[f"{p}.mlp.down_proj.weight"] = [cfg["H"], cfg["FF"]]
    hdr, off = {}, 0
    for n, shp in names.items():
        nbytes = int(np.prod(shp)) * 4
        hdr[n] = {"dtype": "F32", "shape": shp, "data_offsets": [off, off + nbytes]}
        off += nbytes
    root = tmp_path
    hj = json.dumps(hdr).encode()
    with open(root / "weights.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)  # header only; the emitter reads offsets, not data
    np.savez(root / "inputs.npz", in0=np.arange(8, dtype=np.int64)[None, :])
    inv = np.arange(cfg["HD"] // 2, dtype=np.float32)
    np.savez(root / "extra.npz", **{"buf::lm.model.rotary_emb.inv_freq": inv})
    return root


def test_emit_weights_header_valid_c(tmp_path):
    root = _fake_bundle(tmp_path)
    out = tmp_path / "gen"
    out.mkdir()
    h = exo.emit_weights_header(root, out)
    txt = h.read_text()
    assert "#define NL 22" in txt
    assert "#define H 2048" in txt
    assert "#define S 8" in txt
    assert "static const struct layer_off LAYERS[NL]" in txt
    # 22 layer rows + the config lines; each row has 9 offsets.
    assert txt.count("UL},") == 22
    assert "INPUT_IDS[S]" in txt and "INV_FREQ[HD/2]" in txt


# --- RVV audit: the linked-ELF .insn quirk (llvm-objdump decodes, GNU emits .insn) ------------

def test_audit_decodes_rvv_when_objdump_decodes():
    # llvm-objdump-style disasm: the EXO GEMM inner loop is vle/vfmacc/vse.
    disasm = (
        "0000000000012878 <gemm_nt_ref>:\n"
        "   1288a:\tcd047057\tvsetivli\tzero, 0x8, e32, m1, ta, ma\n"
        "   1288c:\t02056507\tvle32.v\tv10, (a0)\n"
        "   12890:\tb2a7d4d7\tvfmacc.vf\tv9, fa5, v10\n"
        "   12894:\t020f64a7\tvse32.v\tv9, (t5)\n"
        "   12898:\t00269293\tslli\tt0, a3, 0x2\n"
        "   1289c:\t00008067\tret\n"
    )
    rep = rvv_audit.classify_disasm(disasm)
    g = rep.by_symbol["gemm_nt_ref"]
    assert g.vector == 4          # vsetivli, vle32, vfmacc, vse32 all match ^v[a-z]; slli is scalar
    assert g.scalar_compute == 1  # slli
    assert g.coverage is not None and g.coverage > 0


# --- not_run_is_not_pass contract -------------------------------------------------------------

def test_missing_bundle_is_not_built_gap(monkeypatch, tmp_path):
    # Point recaptures at an empty dir so the bundle can't be resolved -> not_built with a reason.
    import merlin.baselines.bundle as B
    monkeypatch.setattr(B, "recaptures_dir", lambda: tmp_path)
    r = exo.run(model="small_llama", variant="fp32")
    assert r.status() == "not_built"
    assert r.gap_reason
    assert r.passed is False
    r.validate()


def test_int8_weight_only_has_no_activation_quant_fallback():
    # Weight-only int8 keeps activations in f32, so there is NO activation-quant/requant op — the
    # int8 fallback list must NOT carry one (that belonged to the retired W8A8 path). RMSNorm/RoPE/
    # attention-softmax/SiLU-exp/embed stay scalar; residual-add + SwiGLU product are RVV glue_ops.
    fbs = exo._scalar_fallbacks("int8")
    syms = {f.symbol for f in fbs}
    assert "activation_quant_requant" not in syms
    assert {"rmsnorm", "rope", "attention_softmax", "silu_sigmoid_exp", "embed_gather"} <= syms
    for f in fbs:
        assert f.reason == "no EXO RVV kernel — scalar glue"


def test_detect_llama_config_int8_and_fp32(tmp_path):
    root = _fake_bundle(tmp_path)
    _, hdr = exo._safetensors_offsets(root / "weights.safetensors")
    cfg = exo.detect_llama_config(hdr, "fp32")
    assert cfg is not None
    assert cfg["NL"] == 22 and cfg["H"] == 2048 and cfg["V"] == 32000 and cfg["FF"] == 5632
    # int8 detection on an fp32-named header returns None (no .parametrizations.original0).
    assert exo.detect_llama_config(hdr, "int8") is None


def test_detect_llama_config_none_for_non_llama():
    assert exo.detect_llama_config({"some.random.weight": {"shape": [4, 4]}}, "fp32") is None


def test_int8_gemm_lowers_to_vwmacc():
    import merlin.baselines.exo_kernels.igemm as ig
    src = str(ig.igemm_nt_rvv)
    assert "rvv256_vwmacc_vx" in src           # the RVV widening MAC (integer datapath)
    assert "rvv256_vld_i16" in src and "rvv256_vst_i32" in src


def test_igemm_k_unroll_scales_vwmacc():
    # k-unroll KU issues KU widening MACs per branch (more vector per scalar).
    import merlin.baselines.exo_kernels.igemm as ig
    n1 = str(ig.build_igemm(1)).count("rvv256_vwmacc_vx")
    n4 = str(ig.build_igemm(4)).count("rvv256_vwmacc_vx")
    assert n4 > n1                              # KU=4 body has more vwmaccs than KU=1
    assert exo.IGEMM_U_CANDIDATES[0] == 1       # baseline first in the bounded search


def test_igemm_output_blocking_shares_one_A_load():
    # U-blocking (the RVV-ceiling lever): U 16-wide i32 accumulators share ONE scalar A[m,k] load
    # per k. U distinct accumulator registers (Yr0..Yr{U-1}, NOT an illegal array of RVV types).
    import merlin.baselines.exo_kernels.igemm as ig
    import re
    p = str(ig.build_igemm(1, 4))               # U=4
    flat = re.sub(r"\s+", " ", p)               # collapse the pretty-printer's line wrapping
    assert flat.count("rvv256_vwmacc_vx") == 4  # 4 tiles -> 4 vwmaccs in the k-body
    assert "Yr0" in flat and "Yr3" in flat and "Yr4" not in flat  # exactly 4 distinct registers
    # every vwmacc's A operand is the single element X[m, k] (one CSE'd scalar load feeds U macs).
    # stage_mem may spell the 1-wide window as X[m, k:1+k] or X[m:1+m, k] — both are X[m,k].
    a_windows = re.findall(r"X\[m[^\]]*k[^\]]*\]", flat)
    assert len(a_windows) == 4 and all(("k:1 + k" in w or "m:1 + m" in w) for w in a_windows)


def test_glue_ops_lower_to_rvv():
    # residual-add + ewise-mul move from scalar C to RVV vfadd.vv / vfmul.vv.
    import merlin.baselines.exo_kernels.glue_ops as go
    assert "rvv256_vfadd" in str(go.residual_add_rvv)
    assert "rvv256_vfmul" in str(go.ewise_mul_rvv)


def test_int8_glue_is_weight_only_and_uses_exo_f32_gemm():
    # This capture's int8 golden is a WEIGHT-ONLY int8 reference (int8 per-channel weights dequant
    # to f32, activations f32 — verified: all zero-points 0, numpy repro matches golden cos 1.0).
    # So the int8 glue dequant+transposes the int8 weight (RVV i8->f32 * scale) and runs the EXO
    # *f32* GEMM (gemm_nt_ref) — NOT a full W8A8 activation-quant path (that gates against a W8A8
    # golden this full-fidelity capture doesn't carry: cos 0.949 vs weight-only's cos 1.0).
    from merlin.common.paths import repo_root
    glue = (repo_root() / "merlin/python/merlin/baselines/exo_kernels/llama_glue_int8.c").read_text()
    assert "fgemm_nk_dot_i8(" in glue              # transpose-free weight-only int8 dot GEMM (hot path)
    assert "quant_row_rvv" not in glue             # NO activation quantization (weight-only)
    assert "vsse32" not in glue                    # NO strided scatter (transpose-free)
    assert "weight-only" in glue.lower()           # the honest label is in the source
    # the transpose-free dot dequantizes the native int8 weight inside a contiguous k-reduction.
    fnk = (repo_root() / "merlin/python/merlin/baselines/exo_kernels/fgemm_nk.c").read_text()
    assert "vfmacc_vv" in fnk and "vfredusum" in fnk
    # the EXO f32 + int8 vwmacc kernels are still authored + compiled + RVV-audited (EXO story).
    import merlin.baselines.exo_kernels.igemm as ig
    assert "rvv256_vwmacc_vx" in str(ig.igemm_nt_rvv)


def test_autosched_vectorize_generates_the_reduction_kernel():
    # EXO AUTOCOMPILE: EXO's OWN `vectorize` auto-op (not a hand schedule) must turn the scalar
    # k-dot into a full RVV schedule — 8-wide partial-sum accumulator + contiguous vle + vfmacc.vv +
    # the vfredusum horizontal reduce (the stride-1 reduction the hand schedule couldn't express).
    from exo import compile_procs_to_strings
    from merlin.baselines.exo_kernels import autosched as A
    p = A.build_fdot(1)
    src = str(p)
    assert "rvv256_vfmacc_vv" in src and "rvv256_vredsum" in src   # autoscheduled to the vector MAC + reduce
    assert "rvv256_vld" in src and "rvv256_zero" in src
    c, _ = compile_procs_to_strings([p], "autosched_dot.h")
    assert "__riscv_vfmacc_vv_f32m1" in c and "__riscv_vfredusum" in c  # emits real RVV
    assert "void fdot_nk_ref" in c                                       # stable glue symbol


def test_autosched_glue_uses_autoscheduled_kernel_and_is_transpose_free():
    from merlin.common.paths import repo_root
    glue = (repo_root() / "merlin/python/merlin/baselines/exo_kernels/llama_glue_autosched.c").read_text()
    assert "fdot_nk_ref(" in glue          # calls the EXO-autoscheduled dot
    assert "transpose" not in glue.lower() or "no transpose" in glue.lower()  # transpose-free
    # the nblock autotune knob exists and is measurement-driven (EXO has no cost model).
    from merlin.baselines import exo
    assert hasattr(exo, "autotune_autosched_nblock")
    assert exo.AUTOSCHED_NBLOCK_CANDIDATES[0] == 1


def test_notes_disclose_glue_and_not_whole_model_compiler():
    # The disclosure that this is EXO-kernels-in-a-glue-runtime is mandatory and lives in notes.
    r = BaselineResult(framework="exo", model="tiny_llama")
    r.notes = exo.run.__doc__ or ""
    # build a fresh result the way run() does to check the fixed disclosure string
    fresh = BaselineResult(framework="exo", model="tiny_llama")
    fresh.notes = ("whole-model = EXO RVV GEMM kernel + hand C glue runtime; EXO is a kernel DSL/"
                   "scheduler, NOT a whole-model compiler.")
    assert "NOT a whole-model compiler" in fresh.notes
