"""Per-dtype expert-GEMM CCA fixtures: the CCA must lift a REAL XNNPACK RVV ukernel of each dtype
(f32 / int8 / f16) without error and read the datapath facet the dtype dictates.

The fixtures are genuine `llvm-objdump` disassembly of the vendored XNNPACK RVV microkernels
(cross-compiled with the SpacemiT K1 clang; see `data/cca_asm/AGENT.md` for provenance). This pins
that `cca.lift_asm` generalizes past the single f32 GEMM it started with:

  * the int8 `qd8-f32-qc8w` ukernel is a WIDENING int8xint8->i32 MAC (`vwmacc`), so the CCA must read
    `compute.widening=True` / `accumulator_dtype=i32`;
  * the f16 `7x4v-rvvfp16arith` ukernel accumulates NATIVELY in f16 (NOT a widening `vfwmacc` to
    f32), which is numerically non-comparable to our f32-accumulate datapath -- the expert-wall
    caveat this test documents so the comparison is never read as apples-to-apples.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.kernels import cca
from merlin.kernels.decode import rvv

_DATA = merlin_dir() / "tests" / "data" / "cca_asm"

F32 = _DATA / "xnnpack_f32_gemm_rvv.objdump"
INT8 = _DATA / "xnnpack_qd8_gemm_rvv.objdump"
F16 = _DATA / "xnnpack_f16_gemm_rvv.objdump"


def _lift(fixture: Path):
    return cca.lift_asm(rvv.decode_text(fixture.read_text()), op="matmul", source="expert")


@pytest.mark.parametrize("fixture", [F32, INT8, F16], ids=lambda p: p.stem)
def test_each_dtype_fixture_lifts_without_error(fixture: Path):
    """Every per-dtype expert fixture decodes + lifts to a CCA (regression: a new dtype's asm must
    not throw in the lifter). The lift populates the compute + vector facets from the real stream."""
    assert fixture.is_file(), f"missing fixture {fixture}"
    c = _lift(fixture)
    assert c.op == "matmul"
    assert c.compute is not None and c.vector is not None
    assert c.vector.sew in (16, 32)             # a real vector vtype was read, not None


def test_int8_fixture_shows_widening_mac():
    """The int8 W8A8 datapath is a WIDENING MAC: int8xint8 products accumulated in i32. The CCA must
    read that from the `vwmacc`-family MAC in the stream (not a guessed rule)."""
    stream = rvv.decode_text(INT8.read_text())
    assert stream.count("vwmacc") > 0, "int8 qd8 ukernel must contain a vwmacc-family widening MAC"
    c = _lift(INT8)
    assert c.compute.widening is True
    assert c.compute.accumulator_dtype == "i32"


def test_f16_fixture_is_native_f16_accumulate_not_widening():
    """fp16 NUMERICS CAVEAT (see AGENT.md): the `rvvfp16arith` ukernel accumulates NATIVELY in f16
    (a non-widening e16 `vfmacc.vv`), NOT a widening `vfwmacc` to f32. So its accumulator dtype is
    f16 and `widening` is False -- numerically non-comparable to our f32-accumulate datapath, unlike
    the int8 path which widens to i32. This asymmetry is why an expert-wall comparison against the
    f16 fixture must carry the caveat rather than be read as apples-to-apples."""
    stream = rvv.decode_text(F16.read_text())
    assert stream.count("vfwmacc") == 0, "the 7x4v-rvvfp16arith ukernel is NOT a widening MAC"
    assert stream.count("vfmacc", "vfmadd") > 0
    c = _lift(F16)
    assert c.compute.accumulator_dtype == "f16"
    assert c.compute.widening is False
    assert c.vector.sew == 16
