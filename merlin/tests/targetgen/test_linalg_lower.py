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


@pytest.mark.parametrize("rel", ["RP18_resadd_bf16_pt", "RP16_bias_add_fp32_pt"])
def test_lowered_elementwise_emits_a_valid_kernel(rel):
    muon = pytest.importorskip("merlin.runtime.backends.base")
    codegen = muon.get_backend("muon").muon_codegen_mlir
    cb = lower_linalg_to_cb(_parse(rel), target="t")
    mlir = codegen.emit_kernel_mlir(cb, target="t")
    assert "llvm.func @t_kernel(" in mlir and "llvm.fadd" in mlir


def test_unsupported_pattern_fails_closed():
    # a matmul-family capsule is not (yet) a pattern this lowering builds -> a clear error, not a wrong cb
    with pytest.raises(LinalgLowerError):
        lower_linalg_to_cb(_parse("RP15_fused_matmul_bias_bf16_pt"), target="t")
