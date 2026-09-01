"""`erase_self_copy` is a property of the RECIPE, not of the dtype — proved on the EMITTED CODE.

Two levers in this repo (KC, and MR under unroll_m) looked correctly wired at every layer while being
completely inert, so a lever here is only believed when the emitted code changes. These tests lower
the real compiler and read the resulting LLVM IR:

  * on the v3 micro-kernel recipe the bufferizer leaves a per-tile `memref.copy %x, %x` that survives
    as an opaque `@memrefCopy` rank-generic runtime call — for int8 AND f32 alike;
  * `erase_self_copy` removes that call from the emitted IR — for int8 AND f32 alike, and an MR>1
    recipe now `implies` it, so naming the recipe is enough (see `_tile_epilogue_hygiene`). The test
    below still removes the implication to show the defect reappearing, because "no copy in the IR"
    on its own cannot tell a working implication from a lowering that never had a copy;
  * on the `accumulator_resident_wholemodel_vf` recipe there is NO self-copy to begin with, so the
    feature is INERT there — which is why "turn erase_self_copy on for int8" was the wrong framing.
    The lowering is byte-identical with and without it (measured: the K1 object is the same 6,864
    bytes, and 128^3 timing moves 0.03%, i.e. noise).

MEASURED consequence on the live K1, int8 GEMM kernel region, correctness-gated (min of 3):
    64^3    732,447 -> 425,039 retired ins    14,721 ->  10,301 ticks
   128^3  4,230,288 -> 3,002,346 retired ins   85,760 ->  69,428 ticks
   256^3 27,423,003 -> 22,519,524 retired ins 567,986 -> 503,434 ticks
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_V3 = "accumulator_resident_microkernel_v3"
_WMVF = "accumulator_resident_wholemodel_vf"
_ERASE = "erase_self_copy"


def _m2m() -> bool:
    """The lowering runs in the model2mlir venv; skip cleanly when it is not installed."""
    try:
        from merlin.llvmlower.pipeline import m2m_python
        return Path(m2m_python()).is_file()
    except Exception:  # noqa: BLE001
        return False


def _lower_ir(features: list[str], *, int8: bool, M: int = 64, N: int = 64, K: int = 64) -> str:
    """Lower a matmul through the real compiler and return the emitted LLVM IR text."""
    from merlin.llvmlower.lower import lower_model_file
    from merlin.mining import workloads
    from merlin.runtime.backends import zephyr_model as zm

    bundle = workloads.gen_matmul_f32(tempfile.mkdtemp(prefix="selfcopy_wl_"), M=M, N=N, K=K)
    work = Path(tempfile.mkdtemp(prefix="selfcopy_"))
    prepared = zm._prepare_model_mlir(bundle / "model.mlir", work, int8_compute=int8)
    res = lower_model_file(prepared, work / "lower", targets=(), textual=True, vectorize=True,
                           transform_schedule=None, hoist_static_allocs=False,
                           features=frozenset(features))
    return res.ll_path.read_text(encoding="utf-8")


@pytest.mark.skipif(not _m2m(), reason="model2mlir venv missing")
@pytest.mark.parametrize("int8", [True, False])
def test_the_v3_self_copy_is_real_and_the_implied_hygiene_is_what_removes_it(int8):
    """The defect and its fix are IDENTICAL for int8 and f32 — that is what makes it a recipe
    property rather than a dtype special case.

    This test used to read ``assert "memrefCopy" in _lower_ir([_V3])`` — i.e. it encoded the DEFECT as
    the contract. That premise is now false, deliberately: an MR>1 recipe declares
    ``implies={erase_self_copy}`` (``impr_features._tile_epilogue_hygiene``), so naming the recipe is
    enough and the copy never reaches the emitted IR. The reason is measured — bare MR=4 was 2.04x
    SLOWER than MR=1 on spike purely because of this call (PC-histogram attributed: memrefCopy
    +187,520 instructions, exactly the observed cycle delta, in f32 AND int8 alike).

    The original evidence is KEPT rather than deleted, because "the copy is gone" alone cannot
    distinguish a working implication from a lowering that never had a copy. So the implication is
    temporarily removed from the registry and the defect is shown to reappear — which is what makes it
    load-bearing rather than cosmetic. Patching the parent's registry is sufficient even though the
    lowering runs in a subprocess: ``normalize`` and the argv gate are both evaluated parent-side, and
    only the resolved flag crosses the process boundary.
    """
    import dataclasses

    from merlin.llvmlower import impr_features as F

    assert F.get(_V3).implies == frozenset({_ERASE}), "the hygiene must be DECLARED, not incidental"

    saved = F._REGISTRY[_V3]
    F._REGISTRY[_V3] = dataclasses.replace(saved, implies=frozenset())
    try:
        assert "memrefCopy" in _lower_ir([_V3], int8=int8), (
            "without the implication the per-tile self-copy must still be there — if it is not, this "
            "test no longer proves the implication does anything")
        assert "memrefCopy" not in _lower_ir([_V3, _ERASE], int8=int8)
    finally:
        F._REGISTRY[_V3] = saved

    # ...and with the implication in place, naming ONLY the recipe is enough.
    assert "memrefCopy" not in _lower_ir([_V3], int8=int8)


@pytest.mark.skipif(not _m2m(), reason="model2mlir venv missing")
@pytest.mark.parametrize("int8", [True, False])
def test_erase_is_inert_on_the_wholemodel_vf_recipe(int8):
    """REFUTES "erase_self_copy should be default-ON for int8": on the recipe int8 actually shipped
    with, there is no self-copy, so the feature has nothing to erase. The win required switching the
    RECIPE to v3 first — the erase alone would have bought nothing.

    Asserted on the set of external runtime calls, which is what the erase is about and what is
    stable across the extra canonicalize the feature adds to the pipeline. (The IR TEXT is not
    byte-identical — that canonicalize renumbers SSA values — but the compiled K1 object is: 6,864
    bytes with and without, versus 8,440 -> 7,112 on the v3 recipe, and 128^3 board timing moves
    0.03%, i.e. noise.)
    """
    plain, erased = _lower_ir([_WMVF], int8=int8), _lower_ir([_WMVF, _ERASE], int8=int8)
    assert "memrefCopy" not in plain
    decls = lambda ir: sorted(ln for ln in ir.splitlines() if ln.startswith("declare"))  # noqa: E731
    assert decls(plain) == decls(erased)


@pytest.mark.skipif(not _m2m(), reason="model2mlir venv missing")
def test_the_microkernel_knob_block_delivers_the_erase_for_int8_and_f32_alike():
    """End-to-end through the SHARED capability: a package that names the micro-kernel point via the
    target-agnostic `microkernel` knob block gets the hygiene without listing it, whatever its
    dtype_strategy. This is the seam that stopped int8 from being a parallel silo."""
    from merlin.mining.registry import _resolve_features, load_rvv_package

    pkg_dir = Path(repo_root()) / "out/artifacts/targets/rvv/hand_v0_int8"
    if not pkg_dir.is_dir():
        pytest.skip("hand_v0_int8 package not present")
    base = load_rvv_package(pkg_dir)
    assert base.is_int8 and base.compiler_features == []      # the frozen int8 control is unchanged
    feats = _resolve_features({**base.knobs, "microkernel": {"MR": 4, "NR": 16, "KC": 16}},
                              base.manifest)
    assert feats == [_V3, _ERASE]
    assert "memrefCopy" not in _lower_ir(feats, int8=True)
