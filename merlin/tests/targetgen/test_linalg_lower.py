"""The reference backend's linalg-on-tensors -> command-buffer lowering (:mod:`merlin.targetgen.linalg_lower`).

Reader (granted) parses; this lowering (reference-only) maps the inventory to the command-buffer opcodes
the reference emitter supports, fail-closing on the rest. These tests drive the real transcendental-free
elementwise capsules end to end: parse -> lower -> schema-valid command buffer -> emit a valid kernel.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract import schemas
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.linalg_lower import LinalgLowerError, lower_linalg_to_cb

_CAPS = repo_root() / "merlin" / "contract" / "capsules" / "radiance" / "model_slices"


def _parse(rel: str) -> dict:
    return parse_linalg_mlir((_CAPS / rel / "capsule.interface.mlir").read_text(encoding="utf-8"))


def test_equal_shape_resadd_lowers_to_vector_map():
    cb = lower_linalg_to_cb(_parse("RP18_resadd_bf16_pt"), target="t")
    schemas.validate_command_buffer(cb)
    (cmd,) = cb["commands"]
    assert cmd["opcode"] == "VECTOR_MAP" and cmd["attributes"]["combine"] == "add"
    # two equal-shape 2-D inputs, one 2-D output
    shapes = {k: tuple(v["shape"]) for k, v in cb["tensors"].items()}
    inputs = [k for k, v in cb["tensors"].items() if v["role"] == "input"]
    assert len(inputs) == 2 and shapes[inputs[0]] == shapes[inputs[1]]


def test_row_broadcast_bias_add_lowers_to_vector_map_with_bias_role():
    cb = lower_linalg_to_cb(_parse("RP16_bias_add_fp32_pt"), target="t")
    schemas.validate_command_buffer(cb)
    (cmd,) = cb["commands"]
    assert cmd["opcode"] == "VECTOR_MAP" and cmd["attributes"]["combine"] == "add"
    bias = [v for v in cb["tensors"].values() if v["role"] == "bias"]
    assert len(bias) == 1 and len(bias[0]["shape"]) == 1   # the length-n broadcast row


def test_arg_times_constant_lowers_to_a_scalar_vector_map():
    # embed-scale: X * splat(4.0) -> a scalar VECTOR_MAP with the constant baked in (no rhs operand)
    cb = lower_linalg_to_cb(_parse("RP12_embed_scale_fp32_pt"), target="t")
    schemas.validate_command_buffer(cb)
    (cmd,) = cb["commands"]
    assert cmd["opcode"] == "VECTOR_MAP" and cmd["attributes"]["combine"] == "mul"
    assert cmd["attributes"]["scalar"] == 4.0 and "rhs" not in cmd["operands"]
    inputs = [v for v in cb["tensors"].values() if v["role"] == "input"]
    assert len(inputs) == 1   # only the activation is a runtime operand; the scale is compiled in


def test_single_matmul_with_bias_lowers_to_residency_commands():
    cb = lower_linalg_to_cb(_parse("RP15_fused_matmul_bias_bf16_pt"), target="t")
    schemas.validate_command_buffer(cb)
    assert [c["opcode"] for c in cb["commands"]] == ["RES_PACK", "MATMUL_RESIDENT", "COMMIT"]
    roles = {v["role"] for v in cb["tensors"].values()}
    assert {"input", "weight", "bias", "output"} <= roles
    commit = cb["commands"][-1]
    assert commit["attributes"]["epilogue"] == ["bias_add"] and commit["operands"].get("bias")


@pytest.mark.parametrize("rel", ["RP18_resadd_bf16_pt", "RP16_bias_add_fp32_pt",
                                 "RP15_fused_matmul_bias_bf16_pt", "RP12_embed_scale_fp32_pt"])
def test_lowered_capsule_emits_a_valid_kernel(rel):
    muon = pytest.importorskip("merlin.runtime.backends.base")
    codegen = muon.get_backend("muon").muon_codegen_mlir
    cb = lower_linalg_to_cb(_parse(rel), target="t")
    mlir = codegen.emit_kernel_mlir(cb, target="t")
    assert "llvm.func @t_kernel(" in mlir


def test_chained_matmul_lowers_and_computes_a_at_w1_at_w2():
    import numpy as np

    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.runtime.simulator import simulate

    cb = lower_linalg_to_cb(_parse("RP17_k_chain_fp16_pt"), target="t")
    schemas.validate_command_buffer(cb)
    assert [c["opcode"] for c in cb["commands"]] == \
        ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "RES_PACK", "MATMUL_RESIDENT", "COMMIT"]
    assert sum(v["role"] == "weight" for v in cb["tensors"].values()) == 2  # two weights, one input
    # the chained structure computes A@W1@W2 (checked with the integer simulator)
    for t in cb["tensors"].values():
        t["dtype"] = "i8" if t["role"] in ("input", "weight") else "i32"
    env = materialize_inputs(cb)
    a, w1, w2 = (np.array(env[n].to_list(), dtype=np.int64) for n in ("arg0", "arg1", "arg2"))
    assert simulate(cb)["outputs"]["out"] == ((a @ w1) @ w2).tolist()


@pytest.mark.parametrize("rel", ["RP10_gemv_batched_fp16_pt",   # batched matmul
                                 "RP14_patch_embed_bf16_pt"])    # conv via im2col
def test_unsupported_matmul_shapes_fail_closed(rel):
    # a clear error, not a wrong cb, for compositions the reference emitter cannot build yet
    with pytest.raises(LinalgLowerError):
        lower_linalg_to_cb(_parse(rel), target="t")
