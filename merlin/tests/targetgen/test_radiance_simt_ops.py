"""The SIMT fork-free path must derive operands + emit a reference kernel for the non-GEMM ops
(attention scores Q@K^T, row RMSNorm), and must not crash on a block-scaled (mxfp8) golden whose
``oracle_provenance.inputs`` carries non-tensor entries (E8M0 block-scale code arrays).

Regression for radiance capping at 3/6: ``muon_harness.args_from_cb`` and
``muon_codegen_mlir.emit_kernel_mlir`` only handled a single-matmul GEMM, so ATTENTION_QK / RMSNORM
capsules failed with "no canonical_inputs"; and ``capsule_golden.canonical_input_raws`` did ``.get``
on a list block-scale entry and raised ``AttributeError``. Target-agnostic (nothing here names a target
except as a kernel-symbol parameter).
"""
from __future__ import annotations

from merlin.runtime.backends import muon_codegen_mlir as MC
from merlin.runtime.backends import muon_harness as MH
from merlin.targetgen import capsule_golden as CG


def _attn_cb():
    return {
        "target": "t",
        "tensors": {
            "Q": {"role": "input", "shape": [16, 32], "dtype": "f16"},
            "K": {"role": "input", "shape": [16, 32], "dtype": "f16"},
        },
        "commands": [{"opcode": "ATTENTION_QK", "operands": {"q": "Q", "k": "K", "dst": "Y0"},
                      "attributes": {"output_dtype": "f32"}}],
    }


def _rms_cb():
    return {
        "target": "t",
        "tensors": {
            "X": {"role": "input", "shape": [16, 16], "dtype": "f32"},
            "G": {"role": "weight", "shape": [1, 16], "dtype": "f32"},
        },
        "commands": [{"opcode": "RMSNORM", "operands": {"src": "X", "gamma": "G", "dst": "Y0"},
                      "attributes": {"output_dtype": "f32", "eps": 1e-05}}],
    }


def test_attention_qk_emits_transposed_matmul_kernel():
    mlir = MC.emit_kernel_mlir(_attn_cb(), target="t")
    # signature = inputs in command order, no weight: (Q, K, Y0)
    assert "llvm.func @t_kernel(%Q: !llvm.ptr, %K: !llvm.ptr, %Y0: !llvm.ptr)" in mlir
    # K is indexed by ROW (the transpose): its base offset is n*D, not d*N
    assert "%nD = llvm.mul %ni, %cD" in mlir
    assert "llvm.return" in mlir


def test_rmsnorm_emits_reduce_then_scale_kernel():
    mlir = MC.emit_kernel_mlir(_rms_cb(), target="t")
    # weight-first ABI: (G, X, Y0)
    assert "llvm.func @t_kernel(%G: !llvm.ptr, %X: !llvm.ptr, %Y0: !llvm.ptr)" in mlir
    # rsqrt via hardware sqrt (no libcall) + the eps constant
    assert "llvm.intr.sqrt" in mlir
    assert "1.000000e-05" in mlir


def test_args_from_cb_attention_operands_and_output_shape():
    cb = _attn_cb()
    cb["canonical_inputs"] = {"Q": {"shape": [16, 32], "values": [0.5] * 512},
                              "K": {"shape": [16, 32], "values": [0.25] * 512}}
    ins, outs = MH.args_from_cb(cb)
    assert [a.name for a in ins] == ["Q", "K"]
    assert ins[0].values[0] == 0.5 and len(ins[0].values) == 512
    assert (outs[0].name, outs[0].rows, outs[0].cols) == ("Y0", 16, 16)   # (Qrows, Krows)


def test_args_from_cb_rmsnorm_is_weight_first():
    cb = _rms_cb()
    cb["canonical_inputs"] = {"X": {"shape": [16, 16], "values": [1.0] * 256},
                              "G": {"shape": [1, 16], "values": [2.0] * 16}}
    ins, outs = MH.args_from_cb(cb)
    assert [a.name for a in ins] == ["G", "X"]   # gamma (weight) first
    assert (ins[0].rows, ins[0].cols) == (1, 16)
    assert (outs[0].name, outs[0].rows, outs[0].cols) == ("Y0", 16, 16)


def test_canonical_input_raws_skips_non_tensor_blockscale_entries(tmp_path):
    """An mxfp8 golden records block-scale code arrays (lists) alongside tensor specs; the raws/values
    extractors must skip them, not ``.get`` on a list."""
    gy = tmp_path / "golden.yaml"
    gy.write_text(
        "oracle_provenance:\n"
        "  inputs:\n"
        "    A0: {shape: [1, 2], decoded: [0.5, 0.25], fp8_raw_hex: ['38', '34']}\n"
        "    SA_e8m0_codes:\n"
        "    - [125, 130]\n"
        "    scale_example: {SA0: 125, as_scale: 0.25}\n",
        encoding="utf-8")
    raws = CG.canonical_input_raws({}, tmp_path)
    vals = CG.canonical_input_values({}, tmp_path)
    assert set(raws) == {"A0"} and raws["A0"] == bytes([0x38, 0x34])
    assert set(vals) == {"A0"} and vals["A0"]["values"] == [0.5, 0.25]
