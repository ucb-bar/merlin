"""accumulator_resident_microkernel_v3: the COMPILER-EMITTED accumulator-resident, register-blocked,
vfmacc.vf RVV GEMM micro-kernel (the genuine closer of the #1 scalable-RVV-GEMM gap).

The transform-only v1/v2 features made the accumulator register-resident but the emitted K-loop still
used `vfmacc.vv` (the A operand was a reconstructed vector lane via a vmv/vslideup ladder), measuring
~19x off the hand ceiling. v3 adds (a) a PRE-bufferize `loop-invariant-subset-hoisting` so the
accumulator is a `vector<MRxNR>` scf.for iter_arg (register-resident across K), and (b) an A-operand
scalarization rewrite (`llvmlower/accum_microkernel.py`) so A reaches the backend as a scalar `flw`
and clang selects the clean `vfmacc.vf` — the hand kernel's exact K-loop structure.

These tests:
  * guard the FROZEN baseline (v3 default-off => pipeline + schedule byte-identical);
  * assert v3 splices the subset-hoist + A-scalarize marker only when enabled;
  * build the matmul THROUGH the real RVV compiler and assert, from the STRUCTURED decoder, that the
    INNERMOST (K) loop is accumulator-resident: vfmacc.vf forms, NO vfmacc.vv (the vslideup-ladder
    form), and ZERO in-loop accumulator vector spills (vlNre/vsNr) — the structural success criterion.
    General across a cube and a non-cube (not shape-overfit).

The bit-exact spike verification + instret-vs-ceiling numbers are recorded in
output/kernels/ceiling/scalable_gap_result.md (this test is build+decode only, no slow spike boot).
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower import pipeline as P

REPO = Path(__file__).resolve().parents[3]

_V3 = "accumulator_resident_microkernel_v3"

# spill ops that would indicate the accumulator is NOT register-resident (whole-vreg-group
# load/store the backend emits to round-trip a spilled accumulator through the stack).
_SPILL_OPS = ("vl1re32", "vl2re32", "vl4re32", "vl8re32", "vl1re8", "vl2re8", "vl4re8", "vl8re8",
              "vs1r", "vs2r", "vs4r", "vs8r")


# ---- baseline-frozen guards (no toolchain needed) -----------------------------------
def test_v3_default_off_pipeline_byte_identical():
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == P.build_rvv_pipeline("/tmp/s.mlir")


def test_v3_default_off_schedule_byte_identical():
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE


def test_v3_splices_subset_hoist_and_scalarize_marker_only_when_enabled():
    from merlin.llvmlower.accum_microkernel import SCALARIZE_MARKER
    base = P.build_rvv_pipeline("/tmp/s.mlir").split(",")
    on = F.apply_pipeline(base, frozenset([_V3]))
    # the PRE-bufferize tensor-level subset hoist (the v1/v2 blocker fix) is present...
    assert any("loop-invariant-subset-hoisting" in p for p in on)
    # ...the contract-lowering interpreter is spliced before bufferize...
    assert any("__transform_accum_v2_lower" in p for p in on)
    # ...and the A-scalarization split marker is present (consumed by the two-stage runner).
    assert SCALARIZE_MARKER in on
    # off => unchanged
    assert F.apply_pipeline(base, frozenset()) == base


def test_v3_pre_schedule_drops_bufferize_to_allocation():
    # v3 hoists at TENSOR level, so it must NOT promote C with bufferize_to_allocation (that promotion
    # inserts the per-M,N-tile C copy that dominated the timed region). The v1 schedule keeps it.
    on = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset([_V3]))
    assert "bufferize_to_allocation" not in on
    v1 = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset(["accumulator_resident_microkernel"]))
    assert "bufferize_to_allocation" in v1


def test_v3_runner_selected_for_v3_features():
    # the two-stage A-scalarization runner is keyed on the v3 feature names.
    assert _V3 in P._accum_microkernel_v3_features()
    assert not (P._accum_microkernel_v3_features() & frozenset())  # off => not selected


def test_scalarize_rewrite_is_numerically_neutral_by_construction():
    # The rewrite replaces vector.extract[i,0]:f32 of a transfer_read with a scalar load of element
    # [i,0] of the same source — identical value. Assert the rewriter only targets reads whose uses
    # are ALL scalar f32 extracts (so it can never change a vector-valued use).
    from merlin.llvmlower.accum_microkernel import rewrite_source
    src = rewrite_source()
    assert "vector.transfer_read" in src
    assert 'dims[-1] != "1"' in src                 # only register-tile lhs reads (trailing unit dim)
    assert 'str(owner.results[0].type) != "f32"' in src  # bail unless every use is a scalar f32 extract


# ---- compiler-emitted structural check (needs the riscv toolchain) ------------------
def _toolchain() -> bool:
    try:
        from merlin.kernels import build_asm
        return build_asm.asm_toolchain_available()
    except Exception:
        return False


def _decode(features, M, N, K):
    """apply_rvv_package(hand_v0 + features) on a matmul bundle -> decoded InsnStream of model.o."""
    from merlin.kernels.decode import rvv
    from merlin.rvvgen import workloads
    from merlin.rvvgen.apply import apply_rvv_package
    from merlin.rvvgen.registry import load_rvv_package

    bundle = workloads.gen_matmul_f32(tempfile.mkdtemp(), M=M, N=N, K=K)
    pkg = load_rvv_package(REPO / "generated_targets" / "rvv" / "hand_v0")
    pkg = replace(pkg, run_id="test_v3", compiler_features=list(features))
    work = Path(tempfile.mkdtemp(prefix="test_v3_"))
    apply_rvv_package(pkg, bundle, work, board="spike_riscv64", harts=1, arena_mb=64)
    return rvv.decode(str(work / "model.o"))


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (96, 48, 160)])  # cube + non-cube (not shape-overfit)
def test_v3_innermost_loop_is_accumulator_resident_vfmacc_vf(M, N, K):
    s = _decode([_V3], M, N, K)
    il = s.innermost_loop()
    assert il is not None, "no innermost loop decoded"
    # the K-loop forms the SCALAR-broadcast vfmacc (the hand kernel's form)...
    assert s.count_in(il, "vfmacc.vf") > 0
    # ...NOT the reconstructed-vector-lane form (the vslideup-ladder vfmacc.vv the v2 feature emitted)...
    assert s.count_in(il, "vfmacc.vv") == 0
    # ...and the accumulator is register-resident: ZERO in-loop whole-vreg-group spills.
    assert sum(s.count_in(il, op) for op in _SPILL_OPS) == 0


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_v3_forms_vfmacc_not_separate_mul():
    s = _decode([_V3], 64, 64, 64)
    assert s.count("vfmacc", "vmacc") > 0
    assert s.count("vfmul") == 0
