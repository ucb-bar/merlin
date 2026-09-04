"""`expand_memref_copy` + the post-bufferization stage window every runner must honor.

Two defects are pinned here.

1. THE DROPPED STAGE. ``erase_self_copy`` was routed by the beam seven times and reported inert.
   It was not inert: the act_poly runner variant drove the PassManager itself instead of going
   through ``_run_stages``, so on any fork that ALSO enabled
   ``vectorized_transcendental_activation`` -- which the whole-model proposer enables by default,
   i.e. nearly every fork -- the erase was requested, threaded through argv, and never executed.
   MEASURED on small_llama_int8 before the fix: 17 in-loop ``@memrefCopy`` call sites present with
   the feature ON. A comment cannot detect that, so the runner sources are asserted directly.

2. THE FIXED SPLIT WINDOW. The window was a hardcoded ``k+3`` passes after ``buffer-loop-hoisting``.
   In the RVV pipeline that lands before ``convert-linalg-to-loops``; in the SCALAR pipeline that
   pass sits at ``k+1``, so a rewrite that EMITS linalg (this feature emits ``linalg.copy``) ran
   after its own lowering and the build died in LLVM translation.
"""
from __future__ import annotations

import ctypes
import sys
import types

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import toolchain
from merlin.llvmlower.copy_expand import FEATURE as EXPAND_FEATURE
from merlin.llvmlower.copy_expand import MID_STAGE_SRC, RUNNER_PRELUDE
from merlin.llvmlower.impr_features import known, normalize
from merlin.llvmlower.pipeline import (
    EMIT_TRANSLATE,
    _activation_poly_runner,
    _RUNNER,
    _upstream_pipeline,
)

# A concat: two `tensor.insert_slice` into one buffer. Bufferization gives each a `memref.subview`
# destination with a strided layout, which `finalize-memref-to-llvm` cannot lower to a memcpy -- so
# the baseline emits calls to the rank-generic `@memrefCopy` helper. This is the SHAPE of the 24
# prologue escapes measured in the small_llama int8 whole model, reduced to something a test can run.
CONCAT = """
module {
  func.func @forward(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x16xf32> {
    %e = tensor.empty() : tensor<4x16xf32>
    %0 = tensor.insert_slice %a into %e[0, 0] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<4x16xf32>
    %1 = tensor.insert_slice %b into %0[0, 8] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<4x16xf32>
    return %1 : tensor<4x16xf32>
  }
}
"""


def _scalarize_runner() -> str:
    from merlin.llvmlower.accum_microkernel import run_source
    return run_source().replace("__MERLIN_EMIT__", EMIT_TRANSLATE)


def _runner_variants() -> dict[str, str]:
    return {"plain": _RUNNER,
            "act_poly": _activation_poly_runner(EMIT_TRANSLATE),
            "scalarize": _scalarize_runner()}


def test_feature_is_registered_and_off_by_default():
    assert EXPAND_FEATURE in known()
    assert normalize(None) == frozenset()
    assert normalize([EXPAND_FEATURE]) == frozenset({EXPAND_FEATURE})


@pytest.mark.parametrize("variant", sorted(_runner_variants()))
def test_every_runner_variant_runs_the_post_bufferization_stages(variant):
    """The regression for the dropped stage: no runner may drive the pass pipeline itself."""
    src = _runner_variants()[variant]
    assert "_run_stages(" in src, f"{variant} runner bypasses the post-bufferization stage window"
    assert "_MID_STAGES" in src
    assert "_expand_memref_copies" in src
    assert "_ERASE_SELF_COPY" in src
    assert "_fuse_transpose_b" in src or variant == "scalarize"
    # The whole-pipeline PassManager call is what the act_poly runner used to do instead.
    assert 'PassManager.parse("builtin.module(" + pipeline + ")"' not in src


def test_argv_gate_is_wired_for_every_variant():
    for name, src in _runner_variants().items():
        assert "sys.argv[6]" in src, f"{name} runner never reads the expand gate"
    assert "sys.argv[6]" in MID_STAGE_SRC


def _stage_probe():
    """Exec the runner prelude against a stub PassManager and return (run, calls_of_pipelines)."""
    ns: dict = {}
    from merlin.llvmlower.selfcopy import RUNNER_PRELUDE as SELFCOPY_PRELUDE
    exec("import sys\n" + SELFCOPY_PRELUDE + RUNNER_PRELUDE, ns)  # noqa: S102 - the shipped source
    seen: list[str] = []

    class _PM:
        def __init__(self, text):
            self.text = text

        @classmethod
        def parse(cls, text, _ctx):
            seen.append(text)
            return cls(text)

        def run(self, _op):
            return None

    tm = sys.modules.setdefault("torch_mlir", types.ModuleType("torch_mlir"))
    pm_mod = types.ModuleType("torch_mlir.passmanager")
    pm_mod.PassManager = _PM
    sys.modules["torch_mlir.passmanager"] = pm_mod
    tm.passmanager = pm_mod
    return ns, seen


class _EmptyOp:
    regions: list = []


class _EmptyModule:
    operation = _EmptyOp()


def test_no_feature_runs_the_pipeline_in_one_stage():
    """The baseline invariant: with nothing requested the pass list is not split at all, so the
    emitted object is byte-identical to a build that never heard of these features."""
    ns, seen = _stage_probe()
    ns["_run_stages"](None, _EmptyModule(), _upstream_pipeline(), False, ())
    assert len(seen) == 1
    assert seen[0] == "builtin.module(" + _upstream_pipeline() + ")"


def test_mid_stage_window_stops_before_linalg_is_lowered():
    """A mid rewrite that EMITS linalg must run before convert-linalg-to-loops, in EVERY pipeline.

    The scalar pipeline is the adversarial one: it puts convert-linalg-to-loops immediately after
    buffer-loop-hoisting, so the old fixed k+3 window straddled it."""
    passes = _upstream_pipeline().split(",")
    k = next(i for i, p in enumerate(passes) if "buffer-loop-hoisting" in p)
    j = next(i for i, p in enumerate(passes) if "convert-linalg-to-loops" in p)
    assert j < k + 3, "fixture no longer exercises the clamp; pick another adversarial pipeline"

    ns, seen = _stage_probe()
    calls: list[str] = []
    ns["_run_stages"](None, _EmptyModule(), _upstream_pipeline(), False,
                      [("expand_memref_copy", lambda _c, _m: calls.append("ran") or 0)])
    assert calls == ["ran"], "the mid rewrite never ran"
    assert len(seen) == 2, "the pipeline was not split around the rewrite"
    assert "convert-linalg-to-loops" not in seen[0], \
        "linalg was lowered before the rewrite that emits it"
    assert "convert-linalg-to-loops" in seen[1]


def test_erase_window_is_unchanged_on_the_rvv_pipeline():
    """The clamp must not move the self-copy erase, whose window the RVV pipeline already fits."""
    from merlin.llvmlower.pipeline import build_rvv_pipeline
    from merlin.llvmlower.selfcopy import with_canonicalize

    sched = repo_root() / "out" / "artifacts" / "targets" / "rvv" / "hand_v0_int8" / "schedule.mlir"
    if not sched.is_file():
        pytest.skip("reference RVV package not present")
    pipeline = with_canonicalize(build_rvv_pipeline(sched))
    passes = pipeline.split(",")
    k = next(i for i, p in enumerate(passes) if "buffer-loop-hoisting" in p)

    ns, seen = _stage_probe()
    ns["_run_stages"](None, _EmptyModule(), pipeline, True, ())
    assert len(seen) == 2
    assert seen[0] == "builtin.module(" + ",".join(passes[:k + 3]) + ")"


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
@pytest.mark.parametrize("vectorize", [False, True])
def test_expand_removes_the_runtime_copy_and_keeps_the_numbers(tmp_path, vectorize):
    """End to end on the HOST: the escape disappears from the emitted IR and the output is
    bit-identical. Both pipelines, because they place convert-linalg-to-loops differently."""
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    results = {}
    counts = {}
    for tag, feats in (("off", None), ("on", frozenset({EXPAND_FEATURE}))):
        res = lower_model(CONCAT, tmp_path / f"{tag}_{int(vectorize)}", targets=("host",),
                          textual=True, vectorize=vectorize, features=feats)
        counts[tag] = res.ll_path.read_text(encoding="utf-8").count("@memrefCopy")
        model = HostModel.load(str(res.host_so))
        a = (ctypes.c_float * 32)(*[i * 0.5 for i in range(32)])
        b = (ctypes.c_float * 32)(*[100.0 + i for i in range(32)])
        y = (ctypes.c_float * 64)()
        model([(ctypes.addressof(a), (4, 8)), (ctypes.addressof(b), (4, 8)),
               (ctypes.addressof(y), (4, 16))])
        results[tag] = list(y)

    assert counts["off"] > 0, "fixture no longer produces a rank-generic copy to remove"
    assert counts["on"] == 0, "expand_memref_copy left a @memrefCopy call in the emitted IR"
    assert results["off"] == results["on"], "the expansion changed the numbers"
    assert results["off"][:2] == [0.0, 0.5] and results["off"][8] == 100.0
