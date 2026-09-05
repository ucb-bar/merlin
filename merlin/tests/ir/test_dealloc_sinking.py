"""Where the deallocations land, and the check that a sunk one is still safe.

`ownership-based-buffer-deallocation` frees every buffer at the END of `@forward`: measured on the
deepjscc int8 build, all 112 `free` calls sit after the last compute. Under malloc/free liveness
every intermediate is then live for the whole inference, so the static arena binder
(`llvmlower.arena_bind`) had nothing to reclaim and reported 1.00x reuse on all three int8 models,
while planning the SAME buffers under last-use liveness gives 5.77x / 7.67x / 11.73x.

`func.func(optimize-allocation-liveness)` is upstream's pass for exactly this, and a previous
session recorded that it "moved nothing" and concluded the frees were already tight. The second
half was wrong. `bufferization-lower-deallocations` emits each free inside an `scf.if` guarded by a
(constant-true) ownership flag, and the pass bails unless the dealloc is in the SAME BLOCK as the
alloc -- so it is a guaranteed no-op there, wherever it is placed. The missing piece is the
`canonicalize` upstream's own `buffer-deallocation-pipeline` runs after the lowering
(BufferizationPipelines.cpp:36-38), which folds the statically-true `scf.if` away.

Both halves are pinned here: the no-op WITHOUT the canonicalize (the mis-diagnosis), and the moved
free WITH it. Plus the guard: sinking a free is the one edit in this pipeline whose failure mode is
wrong numbers rather than a failed build, so `pipeline.dealloc_placement_violations` re-checks the
result and MUST reject a free moved before a use -- including a use through an alias.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import pipeline as P
from merlin.llvmlower import toolchain

m2m = pytest.mark.skipif(not toolchain.m2m_python().is_file(),
                         reason="model2MLIR venv (torch-mlir pass registry) not present")

LOWER = "bufferization-lower-deallocations"
SINK = "optimize-allocation-liveness"

#: A three-matmul chain: %alloc feeds matmul 2 and is dead after it, %alloc_0 feeds matmul 3.
#: Nothing else keeps either alive, so a correct sinking frees the first before the last matmul.
CHAIN = """
func.func @forward(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<64x64xf32>
  %f0 = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %e1 = tensor.empty() : tensor<64x64xf32>
  %f1 = linalg.fill ins(%c0 : f32) outs(%e1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %m1 = linalg.matmul ins(%m0, %b : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %e2 = tensor.empty() : tensor<64x64xf32>
  %f2 = linalg.fill ins(%c0 : f32) outs(%e2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %m2 = linalg.matmul ins(%m1, %b : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%f2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %m2 : tensor<64x64xf32>
}
"""

_BUFFERIZE = ("one-shot-bufferize{bufferize-function-boundaries "
              "function-boundary-type-conversion=identity-layout-map},"
              "buffer-results-to-out-params{modify-public-functions hoist-static-allocs}")


def _bufferized_and_deallocated(extra: list[str]) -> str:
    return P.apply_passes(CHAIN, ",".join([_BUFFERIZE] + P._dealloc_passes(sink=False) + extra))


def _stages(text: str) -> list[str]:
    return [s for s in text.split(",") if s]


# --- the pass list ---------------------------------------------------------------------------

def test_the_sinking_stage_is_off_by_default(monkeypatch):
    """An unflagged build must emit the frozen baseline byte for byte, so nothing is added."""
    monkeypatch.delenv("MERLIN_SINK_DEALLOCS", raising=False)
    assert P._dealloc_passes() == ["ownership-based-buffer-deallocation",
                                   "buffer-deallocation-simplification", LOWER]
    for text in (P._upstream_pipeline(), P._parallel_pipeline(),
                 P.build_rvv_pipeline("", hoist_static_allocs=True, features=frozenset())):
        assert SINK not in text


def test_the_env_flag_turns_the_stage_on(monkeypatch):
    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    for text in (P._upstream_pipeline(), P._parallel_pipeline(),
                 P.build_rvv_pipeline("", hoist_static_allocs=True, features=frozenset())):
        assert SINK in text


def test_the_canonicalize_runs_between_the_lowering_and_the_sink(monkeypatch):
    """The ordering IS the fix. Without a canonicalize after `bufferization-lower-deallocations`
    every dealloc is still inside an `scf.if`, and the sinking pass silently moves nothing."""
    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    for name, text in (("upstream", P._upstream_pipeline()),
                       ("parallel", P._parallel_pipeline()),
                       ("rvv", P.build_rvv_pipeline("", hoist_static_allocs=True,
                                                    features=frozenset()))):
        stages = _stages(text)
        lower = stages.index(LOWER)
        sink = next(i for i, s in enumerate(stages) if SINK in s)
        canon = next((i for i in range(lower + 1, sink) if "canonicalize" in stages[i]), None)
        assert canon is not None, f"{name}: no canonicalize between {LOWER} and {SINK}"
        assert lower < canon < sink


# --- what the stage does to the IR -----------------------------------------------------------

@m2m
def test_without_the_sink_every_free_is_at_the_end():
    """The baseline this is here to change: both frees after the last compute, in `scf.if`s."""
    out = _bufferized_and_deallocated([])
    body = [ln.strip() for ln in out.splitlines()]
    last_matmul = max(i for i, ln in enumerate(body) if ln.startswith("linalg.matmul"))
    frees = [i for i, ln in enumerate(body) if ln.startswith("memref.dealloc")]
    assert len(frees) == 2 and min(frees) > last_matmul
    assert any(ln.startswith("scf.if") for ln in body)


@m2m
def test_the_sinking_pass_alone_moves_nothing():
    """The mis-diagnosis, pinned. `optimize-allocation-liveness` after the lowering but WITHOUT the
    canonicalize is a no-op: the dealloc is in the `scf.if`'s block, not the alloc's."""
    out = _bufferized_and_deallocated([f"func.func({SINK})"])
    body = [ln.strip() for ln in out.splitlines()]
    last_matmul = max(i for i, ln in enumerate(body) if ln.startswith("linalg.matmul"))
    assert min(i for i, ln in enumerate(body) if ln.startswith("memref.dealloc")) > last_matmul


@m2m
def test_the_stage_frees_a_buffer_after_its_last_use():
    """With the stage on, the first buffer's free lands before the last matmul -- which is the
    slack the arena binder can then reclaim."""
    out = P.apply_passes(CHAIN, ",".join([_BUFFERIZE] + P._dealloc_passes(sink=True)))
    body = [ln.strip() for ln in out.splitlines()]
    last_matmul = max(i for i, ln in enumerate(body) if ln.startswith("linalg.matmul"))
    frees = [i for i, ln in enumerate(body) if ln.startswith("memref.dealloc")]
    assert len(frees) == 2
    assert min(frees) < last_matmul, "the first buffer is still freed after the last compute"


# --- the guard -------------------------------------------------------------------------------

@m2m
def test_the_placement_check_accepts_what_the_stage_produces():
    out = P.apply_passes(CHAIN, ",".join([_BUFFERIZE] + P._dealloc_passes(sink=True)))
    assert P.dealloc_placement_violations(out) == []


@m2m
def test_the_placement_check_accepts_the_unsunk_baseline():
    """A free left inside an `scf.if` was placed by upstream, not by the sinking stage: it is not
    a violation, or the check would refuse every build that does not sink."""
    assert P.dealloc_placement_violations(_bufferized_and_deallocated([])) == []


#: The stage's own output with ONE free moved back before the buffer's last use -- the mutation that
#: turns a correct program into a use-after-free, and with the arena bound on top of it hands the
#: freed bytes to another buffer while this one is still being read.
MOVED_TOO_EARLY = """
func.func @forward(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
  linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
    outs(%alloc : memref<64x64xf32>)
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<64x64xf32>)
  memref.dealloc %alloc : memref<64x64xf32>
  linalg.matmul ins(%alloc, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
    outs(%alloc_0 : memref<64x64xf32>)
  memref.dealloc %alloc_0 : memref<64x64xf32>
  return
}
"""

#: The same mistake made through an ALIAS: the last direct use of %alloc precedes its free, but a
#: subview of it is read afterwards. A check that only followed the allocation's own SSA value would
#: call this fine.
FREED_BEFORE_AN_ALIAS_IS_READ = """
func.func @forward(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
  %view = memref.subview %alloc[0, 0] [64, 64] [1, 1] :
    memref<64x64xf32> to memref<64x64xf32, strided<[64, 1]>>
  memref.dealloc %alloc : memref<64x64xf32>
  linalg.fill ins(%cst : f32) outs(%view : memref<64x64xf32, strided<[64, 1]>>)
  return
}
"""


@m2m
def test_a_free_moved_before_its_last_use_is_rejected():
    bad = P.dealloc_placement_violations(MOVED_TOO_EARLY)
    assert bad, "a use-after-free passed the placement check"
    assert any("linalg.matmul" in v for v in bad), bad


@m2m
def test_a_free_before_a_read_through_an_alias_is_rejected():
    bad = P.dealloc_placement_violations(FREED_BEFORE_AN_ALIAS_IS_READ)
    assert bad, "a use-after-free through a subview passed the placement check"
    assert any("linalg.fill" in v for v in bad), bad


# --- the guard cannot be skipped -------------------------------------------------------------

def test_every_runner_variant_carries_the_check():
    """The check lives in a `_run_stages` wrapper spliced into the runner source. A variant that
    dropped it would sink every free with nothing verifying the placement."""
    assert P.DEALLOC_CHECK_RUNNER in P._RUNNER_SRC
    assert P.DEALLOC_CHECK_RUNNER in P._activation_poly_runner()


def test_a_sunk_build_whose_runner_never_checked_is_refused(tmp_path, monkeypatch):
    """Fail closed. `accum_microkernel`'s runner drives the PassManager itself for its first stage,
    so it can carry the sinking stage without reaching the wrapper -- and an unchecked sink is the
    silent-corruption case. The parent requires the check's own line in the child's output."""
    class _Proc:
        returncode = 0
        stderr = ""

        def __init__(self, stdout):
            self.stdout = stdout

    def fake(argv, **kw):
        # argv[2] is the runner's output path: create it so only the missing line can fail the build
        from pathlib import Path as _P
        _P(argv[3]).write_text("; empty\n", encoding="utf-8")
        return _Proc(fake.stdout)

    monkeypatch.setattr(P.subprocess, "run", fake)
    sunk = ",".join(["canonicalize"] + P._dealloc_passes(sink=True))

    fake.stdout = "OK\n"
    with pytest.raises(P.PipelineError, match="use-after-free check did not report"):
        P.lower_to_llvm_ir("builtin.module {}", workdir=tmp_path / "a", pipeline=sunk)

    fake.stdout = f"{P.DEALLOC_CHECK_TOKEN} 3 allocations 3 sunk 0 violations\nOK\n"
    P.lower_to_llvm_ir("builtin.module {}", workdir=tmp_path / "b", pipeline=sunk)  # no raise
