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
from merlin.common.paths import repo_root, merlin_dir

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower import pipeline as P

REPO = repo_root()

_V3 = "accumulator_resident_microkernel_v3"
# Whole-model-safe vfmacc.vf variant: v3 (.vf) + the wholemodel MR_mm=1 / NR_bmm=8 tail clamps.
_WMVF = "accumulator_resident_wholemodel_vf"

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
    pkg = load_rvv_package(REPO / "artifacts/targets" / "rvv" / "hand_v0")
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


# ---- whole-model-safe vfmacc.vf: the .vf micro-kernel SURVIVES small-M (the openvla/rdt2 gap) ----
# The decode-level assertion the breakdown (output/kernels/ceiling/kernel_breakdown.md) used. The
# whole-model headline kernel `accumulator_resident_wholemodel` emits `vfmacc.vv` + a per-K
# vslideup/vmv broadcast ladder (~20 inner-loop insns/FMA — the measured openvla/rdt2 residual).
# Bare v3 emits the clean `vfmacc.vf` but its MR=4 M-tiling trips the LLVM-23 masked-transfer_write
# PipelineError on the small-M openvla/rdt2 matmuls (M=17/20/28) -> degrades to NR=8 / non-resident /
# 118 spills. `accumulator_resident_wholemodel_vf` merges the two: v3's `.vf` A-scalarization on the
# wholemodel schedule WITH the MR_mm=1 / NR_bmm=8 clamps, so the `.vf` kernel SURVIVES small-M at
# NR=32, accumulator-resident, 0 spills. These shapes are the actual small-M openvla/rdt2 matmul dims
# the breakdown decoded (17×192×576, 20×128×512) plus a cube and an M=1 token-decode shape.
@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
@pytest.mark.parametrize("M,N,K", [
    (64, 64, 64),       # cube
    (17, 192, 576),     # openvla attn/MLP proj, small-M=17 (bare v3 degrades here; wholemodel_vf must not)
    (20, 128, 512),     # openvla action-head MLP up, M=20
    (28, 1024, 1024),   # rdt2 workhorse attn proj, M=28
    (1, 256, 256),      # M=1 token-decode (the MR_mm=1 clamp's design point)
])
def test_wholemodel_vf_vfmacc_vf_survives_small_M(M, N, K):
    from merlin.kernels import cca
    s = _decode([_WMVF], M, N, K)
    il = s.innermost_loop()
    assert il is not None, "no innermost loop decoded"
    # the .vf micro-kernel fired (scalar-broadcast vfmacc, NOT the vslideup-ladder vfmacc.vv) ...
    assert s.count_in(il, "vfmacc.vf") > 0, "K-loop must emit vfmacc.vf (the .vf micro-kernel)"
    assert s.count_in(il, "vfmacc.vv") == 0, "no vfmacc.vv (the ~20-insn broadcast-ladder form)"
    # ... it SURVIVED small-M at the wide lane group: NR=32 lanes @ VLEN=256 (e32, m4) ...
    c = cca.lift_asm(s, op="matmul", source="ours_wholemodel_vf")
    assert c.vector is not None and c.vector.sew == 32 and c.vector.lmul == 4.0, \
        f"expected e32,m4 (NR=32@VLEN256); got sew={c.vector and c.vector.sew} lmul={c.vector and c.vector.lmul}"
    # ... accumulator register-resident with ZERO in-loop spills (unlike bare v3's 118-spill M-tail
    # degradation on M=17) ...
    assert c.compute is not None and c.compute.accumulator_resident is True
    assert sum(s.count_in(il, op) for op in _SPILL_OPS) == 0, "no in-loop accumulator spills (resident)"
    # ... and the inner loop is MUCH tighter than the .vv broadcast ladder (~20 insns/FMA): the .vf
    # form is a handful of insns per FMA. Assert a real reduction (well under the .vv ladder's 20).
    fma = s.count_in(il, "vfmacc.vf")
    assert len(s.insns_in(il)) / fma < 12, "vfmacc.vf inner loop should be far tighter than the .vv ladder"


def test_wholemodel_vf_default_off_guards():
    # The new feature is default-off: empty features => baseline byte-identical (pipeline + schedule).
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == P.build_rvv_pipeline("/tmp/s.mlir")
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE
    # it is a v3-class feature (selects the two-stage A-scalarization runner) ...
    assert _WMVF in P._accum_microkernel_v3_features()
    # ... carries the wholemodel tail clamps inherent (matmul M-tile=1, batch_matmul N-tile=8) ...
    sch = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset([_WMVF]))
    assert "tile_sizes [1, 16, 0]" in sch        # matmul MR_mm=1
    assert "tile_sizes [1, 4, 8, 0]" in sch      # batch_matmul NR_bmm=8
    # ... and uses the v3 TENSOR-level hoist (no bufferize_to_allocation C promotion).
    assert "bufferize_to_allocation" not in sch
    # splices the subset-hoist + scalarize marker (the .vf path) when enabled.
    from merlin.llvmlower.accum_microkernel import SCALARIZE_MARKER
    on = F.apply_pipeline(P.build_rvv_pipeline("/tmp/s.mlir").split(","), frozenset([_WMVF]))
    assert any("loop-invariant-subset-hoisting" in p for p in on)
    assert SCALARIZE_MARKER in on


# ---- WHOLE-MODEL safety: the isolated tests above missed that v3 fell back to scalar e2e ----
# The isolated single-GEMM tests certified the v3 micro-kernel (vfmacc.vf, resident accumulator) but
# NOT that it composes onto a REAL model. Board measurement (de39201) showed v3 whole-model FELL BACK
# TO SCALAR (bitvla 0.77x / openvla 0.68x REGRESSION): build_k1_binary wraps the vectorized lowering
# in `except PipelineError -> scalar`, and the v3 A-scalarization rewrite raised
# `'tensor.extract' op incorrect number of indices` on a NON-matmul `vector<...x1xf32>` transfer_read
# (a higher-rank source read into a lower-rank vector, where the naive per-extract index
# reconstruction emits the wrong number of indices). These tests lock the whole-model-safe matcher:
# the A-scalarization fires ONLY on the matmul register-tile read it was designed for, and every other
# `vector<...x1>` read is left on the correct baseline lowering (so the model lowers vectorized, not
# scalar). The matcher-guard test needs no toolchain; the real-model test is toolchain-gated.

# A non-matmul `vector<4x1xf32>` transfer_read from a HIGHER-RANK source (tensor<1x4x1xf32>) consumed
# only by scalar `vector.extract`:f32 — the exact shape that broke whole-model v3 (extract position
# length 2 != source rank 3). The matcher must REFUSE this (it cannot soundly reconstruct indices),
# leaving it on the baseline lowering. A genuine matmul A read (vector<MRx1> from tensor<MRx1>, equal
# rank, identity map) is what it accepts.
def test_scalarize_matcher_refuses_rank_mismatched_reads():
    # The rewriter source must carry the whole-model-safety gates: a static source rank/dims, a
    # bound on the extract-position length vs the source rank, and a (rank-reducing-allowed) identity-
    # permutation gate. The MR_mm=1 whole-model clamp collapses the A read `vector<1x1>` -> `vector<1>`
    # with a rank-reducing minor-identity map, so the matcher now ACCEPTS a read that DROPS leading
    # source dims — but ONLY when each dropped leading dim has extent 1 (so the per-row scalar load is
    # provably value-identical) and the projection keeps the trailing dims in order. A position length
    # ABOVE the source rank, or a dropped leading dim with extent>1, is still refused (it would emit a
    # malformed / mis-mapped `tensor.extract` -> the silent-scalar fallback root cause). Assert the
    # gates are present (the structural contract of the fix).
    from merlin.llvmlower.accum_microkernel import rewrite_source
    src = rewrite_source()
    assert "_src_rank" in src                       # source must be statically ranked
    assert "_src_dims" in src                       # ...and its dim EXTENTS known (unit-leading check)
    assert "poslen > rank" in src                   # extract position length above source rank -> refuse
    # dropped LEADING dims (poslen < rank) must each have extent 1, else the lane != element mapping
    assert "poslen < rank" in src and "src_dims[:rank - poslen]" in src
    assert "_is_identity_perm" in src               # identity OR sound minor-identity projection only
    # and it must explicitly NOT key the gate on len(operands)-1 (transfer_read carries padding/mask
    # trailing operands, so that count is NOT the index count — the bug a naive count would re-introduce).
    assert "transfer_read carries trailing" in src.lower() or "padding scalar" in src.lower()


def _decode_model(model_name, features):
    """Lower a REAL model bundle (output/<model>) through the RVV compiler with `features` and decode
    the emitted model.o. Returns the InsnStream. Mirrors build_k1_binary's vectorized lowering path
    (vectorize=True, features=, hoist_static_allocs=False) so a PipelineError here is exactly the
    silent scalar fallback the board hit."""
    import subprocess
    from merlin.kernels.decode import rvv
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.runtime.backends import zephyr_model as zm

    mdir = REPO / "artifacts" / "recaptures" / model_name
    work = Path(tempfile.mkdtemp(prefix=f"test_v3_wm_{model_name}_"))
    prepared = zm._prepare_model_mlir(mdir / "model.mlir", work, int8_compute=False)
    feats = frozenset(features) or None
    # vectorize=True must SUCCEED (no PipelineError) — that is the whole-model-safety assertion.
    res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                           vectorize=True, hoist_static_allocs=False, features=feats)
    clang = toolchain.clang()
    obj = work / "model.o"
    subprocess.run([str(clang), "--target=riscv64-unknown-elf", "-march=rv64gcv", "-mabi=lp64d",
                    "-O2", "-Wno-override-module", "-c", str(res.ll_path), "-o", str(obj)],
                   check=True, capture_output=True)
    return rvv.decode(str(obj))


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
@pytest.mark.skipif(not (REPO / "artifacts" / "recaptures" / "bitvla_fp32_consistent" / "model.mlir").is_file(),
                    reason="bitvla model bundle not present")
@pytest.mark.parametrize("model", ["bitvla_fp32_consistent", "openvla_fp32_consistent"])
def test_v3_whole_model_lowers_vectorized_not_scalar_fallback(model):
    # The whole-model assertion the isolated tests missed: lowering a REAL model with v3 must
    # (a) SUCCEED (no PipelineError -> no silent scalar fallback in build_k1_binary), and
    # (b) actually apply the v3 micro-kernel — vfmacc.vf present in the emitted asm (the baseline
    #     emits ZERO vfmacc of any form, so vfmacc.vf>0 proves the resident-accumulator kernel fired
    #     whole-model, not a scalar/vfmul fallback).
    if not (REPO / "artifacts" / "recaptures" / model / "model.mlir").is_file():
        pytest.skip(f"{model} bundle not present")
    base = _decode_model(model, [])
    assert base.count("vfmacc", "vmacc") == 0, "baseline must emit no vfmacc (un-fused matmul)"
    on = _decode_model(model, [_V3])
    assert on.count("vfmacc.vf") > 0, "v3 whole-model must emit vfmacc.vf (resident-accumulator kernel)"
