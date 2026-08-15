"""The reference backend's self-contained MX (block-scaled fp8/fp6/fp4) kernel emission.

MX operands (quantized codes + corpus-seeded E8M0 block scales) live only in the capsule golden — they are
NOT reconstructable from the decoded-float workload (the scales are a function of the capsule-name salt) —
so this reference path bakes them from the golden's ``operand_codes`` bundle. Public-capsule known-good
baseline; masked for hidden capsules. These tests assert the emitter recognizes an MX command buffer and
bakes a compilable co-model kernel with the right datatype for each format.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_golden as CG
from merlin.targetgen.capsule_common import load_capsule
from merlin.common.paths import repo_root

_ISA = repo_root() / "merlin" / "contract" / "capsules" / "radiance" / "isa"
_CONTRACT = repo_root() / "merlin" / "contract"


def _mx():
    muon = pytest.importorskip("merlin.runtime.backends.base")
    return muon.get_backend("muon").muon_mx_codegen


def test_is_mx_cb_detects_microscaling_operands():
    mx = _mx()
    assert mx.is_mx_cb({"tensors": {"A": {"dtype": "f8E4M3FN"}, "B": {"dtype": "f8E4M3FN"}}})
    assert mx.is_mx_cb({"tensors": {"A": {"dtype": "f6E3M2FN"}}})
    assert mx.is_mx_cb({"tensors": {"A": {"dtype": "f4E2M1FN"}}})
    assert not mx.is_mx_cb({"tensors": {"A": {"dtype": "f32"}, "B": {"dtype": "bf16"}}})


@pytest.mark.parametrize("name,datatype,ain", [
    ("R5_mx_tile_mxfp8", "GemmDatatype::FP8", "A_in[128][128]"),
    ("R6_mx_tile_mxfp6", "GemmDatatype::FP6", "A_in_hw[64][128]"),
    ("R7_mx_tile_mxfp4", "GemmDatatype::FP4", "A_in_hw[64][128]"),
])
def test_emit_mx_kernel_bakes_the_format_specific_datatype_and_layout(name, datatype, ain):
    if not (_ISA / name).is_dir():
        pytest.skip(f"{name} capsule not present")
    mx = _mx()
    cap = load_capsule(str(_ISA / name), contract=str(_CONTRACT))
    cd = cap.get("__dir__", str(_ISA / name))
    ops = CG.mx_operands(cap, cd)
    assert ops is not None, "golden should carry an MX operand bundle"
    k = mx.emit_mx_kernel(ops, "Y0")
    assert datatype in k                       # per-format co-model datatype
    assert ain in k                            # fp8 = byte/elem, fp6/fp4 = nibble-packed A_in_hw
    assert "mxgemm<CFG>" in k and "int main" in k   # self-contained co-model driver
    assert "OUT" in k                          # OUT-protocol result print (via vx_putchar)


def test_batched_mx_packs_block_diagonal_single_tile():
    """A B-way batched MX matmul packs into ONE block-diagonal tile (batch b in K-group b) that reproduces
    every batch — so the single-tile emitter handles it. The assembled tile is numerically identical to the
    per-batch matmuls at fp8-decode level (verified against the golden's per-batch codes)."""
    name = "R11_gemv_batched_mx"
    slic = repo_root() / "merlin" / "contract" / "capsules" / "radiance" / "model_slices" / name
    if not slic.is_dir():
        pytest.skip(f"{name} capsule not present")
    mx = _mx()
    cap = load_capsule(str(slic), contract=str(_CONTRACT))
    ops = CG.mx_operands(cap, cap.get("__dir__", str(slic)))
    assert ops is not None and ops.get("batched")
    bd = mx._assemble_batched(ops)
    b, m, n, h = ops["B"], ops["M"], ops["N"], ops["H"]
    assert bd["M"] == b * m and bd["K"] == b * h and bd["N"] == n
    assert len(bd["A_bytes"]) == bd["M"] * bd["K"] and len(bd["SA"]) == b
    k = mx.emit_mx_kernel(ops, "Y0")               # emits through the single-tile path
    assert "mxgemm<CFG>" in k and "int main" in k


def test_fp6_bakes_the_lut_palette():
    name = "R6_mx_tile_mxfp6"
    if not (_ISA / name).is_dir():
        pytest.skip(f"{name} capsule not present")
    mx = _mx()
    cap = load_capsule(str(_ISA / name), contract=str(_CONTRACT))
    cd = cap.get("__dir__", str(_ISA / name))
    k = mx.emit_mx_kernel(CG.mx_operands(cap, cd), "Y0")
    assert "A_lut[64][3]" in k and "B_lut[64][3]" in k    # fp6 packs a 16-entry palette to 96-bit LUT slots
