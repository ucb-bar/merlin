"""The merlin_iface grammar's whole-op mnemonics (op classes with no residency decomposition).

`parse_interface_mlir` learned `rmsnorm` / `attention_qk` (mnemonics the frozen parser previously
dropped -> 0 commands -> a hard parse failure, or, when fused with a matmul, a SILENT drop that
mis-graded). These tests assert the mnemonics parse to the command-buffer opcode + operand keys the
target codegen reads, schema-validate, reach the reference emitter, and that an unsupported FUSED
combination fails loud instead of silently emitting one half.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.contract import schemas
from merlin.targetgen.contract.interface_emit import parse_interface_mlir

_RMSNORM = """
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xf32>
  %G = merlin_iface.tensor {name = "G", role = "weight"} : tensor<1x16xf32>
  %Y0 = merlin_iface.rmsnorm %X, %G {name = "Y0", eps = 1.000000000e-05 : f64, output_dtype = "f32"} : (tensor<16x16xf32>, tensor<1x16xf32>) -> tensor<16x16xf32>
}
"""

_ATTN_QK = """
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %Q = merlin_iface.tensor {name = "Q", role = "input"} : tensor<16x32xf16>
  %K = merlin_iface.tensor {name = "K", role = "input"} : tensor<16x32xf16>
  %Y0 = merlin_iface.attention_qk %Q, %K {name = "Y0", output_dtype = "f32"} : (tensor<16x32xf16>, tensor<16x32xf16>) -> tensor<16x16xf32>
}
"""


def test_rmsnorm_parses_to_the_opcode_and_operand_keys_codegen_reads():
    cb = parse_interface_mlir(_RMSNORM)
    cmds = cb["commands"]
    assert len(cmds) == 1
    c = cmds[0]
    assert c["opcode"] == "RMSNORM"
    assert c["operands"] == {"src": "X", "gamma": "G", "dst": "Y0"}
    assert c["attributes"]["eps"] == pytest.approx(1e-5)
    assert c["attributes"]["output_dtype"] == "f32"
    schemas.validate_command_buffer(cb)


def test_attention_qk_parses_to_the_opcode_and_operand_keys_codegen_reads():
    cb = parse_interface_mlir(_ATTN_QK)
    c = cb["commands"][0]
    assert c["opcode"] == "ATTENTION_QK"
    assert c["operands"] == {"q": "Q", "k": "K", "dst": "Y0"}
    schemas.validate_command_buffer(cb)


def test_named_ops_do_not_disturb_the_residency_grammar():
    # a plain matmul buffer still parses to exactly its residency commands (no named-op misfire)
    mm = """
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<16x16xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>
  %Wp = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc = merlin_iface.matmul %A, %Wp : (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y = merlin_iface.commit %acc {name = "Y", output_dtype = "i8"} : (!merlin_iface.acc<i32>) -> tensor<16x16xi8>
}
"""
    cb = parse_interface_mlir(mm)
    assert [c["opcode"] for c in cb["commands"]] == ["RES_PACK", "MATMUL_RESIDENT", "COMMIT"]


def test_named_ops_reach_the_reference_emitter():
    muon = pytest.importorskip("merlin.runtime.backends.base")
    codegen = muon.get_backend("muon").muon_codegen_mlir
    for text in (_RMSNORM, _ATTN_QK):
        cb = parse_interface_mlir(text)
        mlir = codegen.emit_kernel_mlir(cb, target="t")
        assert "llvm.func @t_kernel(" in mlir


def test_vector_map_elementwise_emits_add_and_mul():
    """The transcendental-free elementwise core (VECTOR_MAP add/mul) emits a valid single-loop kernel."""
    muon = pytest.importorskip("merlin.runtime.backends.base")
    codegen = muon.get_backend("muon").muon_codegen_mlir
    for combine, fop in (("add", "llvm.fadd"), ("mul", "llvm.fmul")):
        cb = {
            "abi_version": "0.1", "target": "t",
            "tensors": {"A": {"shape": [16, 16], "dtype": "f32", "role": "input"},
                        "B": {"shape": [16, 16], "dtype": "f32", "role": "input"}},
            "commands": [{"opcode": "VECTOR_MAP", "operands": {"lhs": "A", "rhs": "B", "dst": "Y"},
                          "attributes": {"combine": combine}}],
            "outputs": ["Y"],
        }
        mlir = codegen.emit_kernel_mlir(cb, target="t")
        assert "llvm.func @t_kernel(%A: !llvm.ptr, %B: !llvm.ptr, %Y: !llvm.ptr)" in mlir
        assert fop in mlir


def test_vector_map_row_broadcast_bias_add_emits():
    """A length-n rhs broadcasts over the m rows of an (m,n) lhs (standalone bias-add / per-channel scale)."""
    muon = pytest.importorskip("merlin.runtime.backends.base")
    codegen = muon.get_backend("muon").muon_codegen_mlir
    cb = {
        "abi_version": "0.1", "target": "t",
        "tensors": {"A": {"shape": [16, 16], "dtype": "f32", "role": "input"},
                    "B": {"shape": [16], "dtype": "f32", "role": "bias"}},
        "commands": [{"opcode": "VECTOR_MAP", "operands": {"lhs": "A", "rhs": "B", "dst": "Y"},
                      "attributes": {"combine": "add"}}],
    }
    mlir = codegen.emit_kernel_mlir(cb, target="t")
    assert "llvm.func @t_kernel(%A: !llvm.ptr, %B: !llvm.ptr, %Y: !llvm.ptr)" in mlir
    assert "llvm.fadd" in mlir


def test_vector_map_rejects_incompatible_shapes():
    muon = pytest.importorskip("merlin.runtime.backends.base")
    codegen = muon.get_backend("muon").muon_codegen_mlir
    cb = {
        "abi_version": "0.1", "target": "t",
        "tensors": {"A": {"shape": [16, 16], "dtype": "f32", "role": "input"},
                    "B": {"shape": [8], "dtype": "f32", "role": "input"}},
        "commands": [{"opcode": "VECTOR_MAP", "operands": {"lhs": "A", "rhs": "B", "dst": "Y"},
                      "attributes": {"combine": "add"}}],
    }
    with pytest.raises(Exception, match="equal-shape or a row-broadcast"):
        codegen.emit_kernel_mlir(cb, target="t")


def test_fused_named_op_plus_matmul_fails_loud_not_silent():
    """A rmsnorm fused with a matmul must NOT silently emit one half; the reference emitter raises."""
    muon = pytest.importorskip("merlin.runtime.backends.base")
    codegen = muon.get_backend("muon").muon_codegen_mlir
    cb = {
        "abi_version": "0.1", "target": "t",
        "tensors": {
            "X": {"shape": [16, 16], "dtype": "f32", "role": "input"},
            "G": {"shape": [1, 16], "dtype": "f32", "role": "weight"},
            "W": {"shape": [16, 16], "dtype": "f32", "role": "weight"},
        },
        "commands": [
            {"opcode": "RMSNORM", "operands": {"src": "X", "gamma": "G", "dst": "H"},
             "attributes": {"eps": 1e-5}},
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wp"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "H", "rhs": "Wp", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y"},
             "attributes": {"output_dtype": "f32"}},
        ],
    }
    with pytest.raises(Exception) as ei:
        codegen.emit_kernel_mlir(cb, target="t")
    assert "fused" in str(ei.value).lower()
