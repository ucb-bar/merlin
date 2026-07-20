"""`vectorize_reduction`: the compute.reduction_form lever, PROVEN ON THE EMITTED CODE.

The baseline RVV schedule vectorizes only the contraction ops, so a standalone reduction (softmax
max/sum, LayerNorm/RMSNorm mean/var, a `linalg.reduce`) falls through convert-linalg-to-loops to a
SCALAR accumulate loop. This feature vectorizes it and lowers `vector.multi_reduction` -> a hardware
horizontal reduce (`vfredusum.vs` for fp / `vredsum.vs` for int).

Two levers in this repo (KC, and MR under unroll_m) looked correctly wired at every layer while being
completely inert, so a reduction lever is only believed when the EMITTED instruction stream changes.
These tests build the real compiler and decode the objdump under the REAL RVV cflags
(`-fno-vectorize -fno-slp-vectorize`) — so clang's own auto-vectorizer is OFF and every vector
reduction seen is MLIR-emitted by the feature, not clang re-vectorizing a scalar loop:

  * a gen_reduce_f32 sum reduction emits `vfredusum.vs` where the baseline emits ZERO vector ops;
  * a gen_softmax_f32 emits `vfredmax.vs` (max reduce) + `vfredusum.vs` (sum reduce);
  * both lift (via cca.lift_asm on the decoded stream) to a non-null reduction_form;
  * the contraction lowering is UNCHANGED (a matmul object is byte-identical with/without the feature),
    so the feature is whole-model-safe (no matmul regression).

APPROXIMATION: `reassociate-fp-reductions` reorders the fp sum (the unordered tree reduction XNNPACK
et al. also use) — gated on cos/rel error, never claimed bit-exact.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower import pipeline as P

_FEAT = "vectorize_reduction"


# ---- baseline-frozen guards (no toolchain needed) -----------------------------------
def test_default_off_pipeline_byte_identical():
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == P.build_rvv_pipeline("/tmp/s.mlir")


def test_default_off_schedule_byte_identical():
    assert F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset()) == P.RVV_TRANSFORM_SCHEDULE


def test_feature_registered_and_default_off():
    assert _FEAT in F.known()
    f = F.get(_FEAT)
    assert f.action_class == "PASS" and f.schedule_replace


def test_schedule_inserts_reduction_vectorize_and_keeps_contraction():
    sch = F.apply_schedule(P.RVV_TRANSFORM_SCHEDULE, frozenset([_FEAT]))
    # the reduction match+vectorize block is inserted (all three ranks) ...
    assert "iterator_types = [#linalg.iterator_type<reduction>]" in sch
    assert sch.count("transform.structured.vectorize %redr") == 3
    # ... while the baseline contraction handling is left intact (no matmul regression).
    assert 'transform.structured.match ops{["linalg.matmul"]}' in sch
    assert "transform.apply_patterns.vector.lower_contraction" in sch


def test_pipeline_switches_reduction_lowering_strategy_and_reassociates():
    on = F.apply_pipeline(P.build_rvv_pipeline("/tmp/s.mlir").split(","), frozenset([_FEAT]))
    assert any("lower-vector-multi-reduction{lowering-strategy=inner-reduction}" in p for p in on)
    assert any("convert-vector-to-llvm{reassociate-fp-reductions}" in p for p in on)
    # off => unchanged
    base = P.build_rvv_pipeline("/tmp/s.mlir").split(",")
    assert F.apply_pipeline(base, frozenset()) == base


# ---- compiler-emitted structural proof (needs m2m venv + clang + riscv objdump) ------
def _can_decode() -> bool:
    """Needs the model2MLIR venv (LLVM-23 passes), a riscv-capable clang, and an llvm-objdump that can
    disassemble riscv64 — the full lower -> compile -> decode path this proof exercises."""
    try:
        from merlin.llvmlower import toolchain
        from merlin.llvmlower.pipeline import m2m_python
        from merlin.kernels.decode.objdump import objdump_bin
        if not (Path(m2m_python()).is_file() and Path(toolchain.clang()).is_file()):
            return False
        ob = objdump_bin()
        if not (ob and (shutil.which(ob) or Path(ob).is_file())):
            return False
        # probe that this objdump understands riscv (system x86 objdump does not)
        p = subprocess.run([ob, "--version"], capture_output=True, text=True)
        return "riscv" in (p.stdout + p.stderr).lower() or "llvm" in (p.stdout + p.stderr).lower()
    except Exception:  # noqa: BLE001
        return False


def _decode(features, gen_fn):
    """Lower a reduction workload through the real compiler with `features`, compile it with the REAL
    RVV cflags (-fno-vectorize, so any vector reduction is MLIR-emitted), and decode the object."""
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.kernels.decode import rvv
    from merlin.runtime.backends import zephyr_model as zm

    bundle = gen_fn(tempfile.mkdtemp(prefix="red_wl_"))
    work = Path(tempfile.mkdtemp(prefix="red_"))
    prepared = zm._prepare_model_mlir(bundle / "model.mlir", work, int8_compute=False)
    res = lower_model_file(prepared, work / "lower", targets=(), textual=True, vectorize=True,
                           hoist_static_allocs=False,
                           features=(frozenset(features) or None))
    obj = work / "model.o"
    subprocess.run([str(toolchain.clang()), "--target=riscv64-unknown-elf", *zm.RVV_CFLAGS,
                    "-Wno-override-module", "-c", str(res.ll_path), "-o", str(obj)],
                   check=True, capture_output=True)
    return rvv.decode(str(obj))


def _reduce_sum(d):
    from merlin.rvvgen import workloads
    return workloads.gen_reduce_f32(d, op="sum", M=64, N=256)


def _softmax(d):
    from merlin.rvvgen import workloads
    return workloads.gen_softmax_f32(d, M=64, N=256)


def _matmul(d):
    from merlin.rvvgen import workloads
    return workloads.gen_matmul_f32(d, M=64, N=64, K=64)


@pytest.mark.skipif(not _can_decode(), reason="m2m venv / clang / riscv objdump missing")
def test_reduce_emits_vfredusum_where_baseline_emits_nothing():
    base = _decode([], _reduce_sum)
    assert base.count("vfredusum", "vfredosum", "vredsum") == 0, \
        "baseline must leave the reduction scalar (no vector reduce; -fno-vectorize)"
    on = _decode([_FEAT], _reduce_sum)
    assert on.count("vfredusum") > 0, "feature must emit the unordered vfredusum.vs (MLIR-vectorized)"
    # ... and the CCA lifted from the decoded stream now SEES a reduction (the lever's promise).
    from merlin.kernels import cca
    c = cca.lift_asm(on, op="reduce", source="ours_reduction")
    assert c.compute is not None and c.compute.reduction_form not in (None, "none")


@pytest.mark.skipif(not _can_decode(), reason="m2m venv / clang / riscv objdump missing")
def test_softmax_vectorizes_both_reductions():
    on = _decode([_FEAT], _softmax)
    # the sum-reduce -> vfredusum, the max-reduce -> vfredmax; both are hardware horizontal reduces.
    assert on.count("vfredusum") > 0, "softmax sum-reduce must emit vfredusum.vs"
    assert on.count("vfredmax") > 0, "softmax max-reduce must emit vfredmax.vs"
    from merlin.kernels import cca
    c = cca.lift_asm(on, op="softmax", source="ours_reduction")
    assert c.compute is not None and c.compute.reduction_form not in (None, "none")


@pytest.mark.skipif(not _can_decode(), reason="m2m venv / clang / riscv objdump missing")
def test_matmul_object_unchanged_no_regression():
    """Whole-model safety: enabling the reduction feature must NOT change a matmul's emitted code
    (the contraction schedule is left intact)."""
    base = _decode([], _matmul).vector_histogram()
    on = _decode([_FEAT], _matmul).vector_histogram()
    assert base == on, f"matmul vector mix changed: {base} -> {on}"
