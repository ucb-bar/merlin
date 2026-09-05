"""`parallel_grain`: a parallel region is only worth entering if there is work in it.

The multicore RVV pipeline hands every `scf.parallel` to `convert-scf-to-openmp`, which gives each
one its own `omp.parallel` + `__kmpc_fork_call`. Nothing on that path has a cost model, so a loop
that stores 32 floats forks, barriers and joins exactly like a 25-million-element contraction does.
Counted off the whole-model int8 lowerings at 8 harts, an inference enters 23,344 (lstmnetvit) and
5,160 (deepjscc) parallel regions, and 97% / 79% of those entries carry 2.6% / 1.6% of the work --
while the Amdahl serial fraction of the same builds is 1.4% and 0.001%. Parallel COVERAGE is not the
ceiling; the number of region entries is.

This pins the rewrite that drops the cheap ones, and the three properties without which it would be
unsafe or unmeasurable: it must be default-off and leave the pass string and the schedule untouched
(the frozen baseline lowers byte-identically), it must run in the window where `scf.parallel` exists
and `convert-scf-to-openmp` has not yet run (a decision made at the `mid` split point would price
loops that do not exist yet), and it must FAIL CLOSED -- a loop it cannot price, or one carrying a
reduction, keeps its parallelism rather than being serialized on an assumption.
"""
from __future__ import annotations

import subprocess

import pytest

import merlin.llvmlower.lower  # noqa: F401 — the production import that wires the lowering
from merlin.llvmlower import toolchain
from merlin.llvmlower.impr_features import apply_pipeline, apply_schedule, get, normalize
from merlin.llvmlower.parallel_grain import (FEATURE_PREFIX, RUNNER_PRELUDE, feature_name,
                                             threshold_of)

#: One cheap `scf.parallel` (32 stores) and one expensive one (4096 x 64 loads/stores), the two
#: shapes the whole-model lowerings actually contain.
CHEAP_AND_EXPENSIVE = """
module {
  func.func @forward(%a: memref<32xf32>, %b: memref<4096x64xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant 1.000000e+00 : f32
    scf.parallel (%i) = (%c0) to (%c32) step (%c1) {
      memref.store %cst, %a[%i] : memref<32xf32>
      scf.reduce
    }
    scf.parallel (%i, %j) = (%c0, %c0) to (%c4096, %c64) step (%c1, %c1) {
      %v = memref.load %b[%i, %j] : memref<4096x64xf32>
      %w = arith.mulf %v, %cst : f32
      memref.store %w, %b[%i, %j] : memref<4096x64xf32>
      scf.reduce
    }
    return
  }
}
"""

#: A cheap loop whose upper bound is a runtime value. It cannot be priced, so it must be LEFT
#: parallel: guessing a cost is how a rewrite silently serializes the model's hot loop.
DYNAMIC_BOUND = """
module {
  func.func @forward(%a: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    scf.parallel (%i) = (%c0) to (%n) step (%c1) {
      memref.store %cst, %a[%i] : memref<?xf32>
      scf.reduce
    }
    return
  }
}
"""

#: A cheap loop that CARRIES a reduction. Serializing it would mean rewriting its `scf.reduce`
#: terminator as well; refuse instead.
WITH_REDUCTION = """
module {
  func.func @forward(%a: memref<32xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant 0.000000e+00 : f32
    %s = scf.parallel (%i) = (%c0) to (%c32) step (%c1) init (%zero) -> f32 {
      %v = memref.load %a[%i] : memref<32xf32>
      scf.reduce(%v : f32) {
      ^bb0(%lhs: f32, %rhs: f32):
        %r = arith.addf %lhs, %rhs : f32
        scf.reduce.return %r : f32
      }
    }
    return %s : f32
  }
}
"""


def test_a_grain_point_is_registered_from_its_name_and_absent_by_default():
    """The lowering runs in a CHILD process that re-imports the registry: a point registered only in
    the parent fails to resolve there (the failure `_try_lazy_register` exists for). And an unnamed
    build must normalize to the empty set, or the frozen baseline is not byte-identical."""
    name = feature_name(12345)
    assert name == f"{FEATURE_PREFIX}12345"
    assert normalize({name}) == frozenset({name})       # resolves purely from the string
    assert normalize(None) == frozenset()
    with pytest.raises(KeyError):
        get(f"{FEATURE_PREFIX}notanumber")


def test_two_thresholds_at_once_are_refused_rather_than_silently_merged():
    """They describe incompatible grains; picking either would make the build unattributable."""
    assert threshold_of(frozenset()) is None
    assert threshold_of({feature_name(1000), "erase_self_copy"}) == 1000
    with pytest.raises(ValueError):
        threshold_of({feature_name(1000), feature_name(2000)})


def test_it_edits_neither_the_pass_list_nor_the_schedule_nor_the_cflags():
    """It is a module rewrite gated on argv, so a build that names it must still produce the frozen
    baseline's pipeline string — the only thing that may differ is the emitted IR."""
    name = feature_name(10000)
    normalize({name})
    assert get(name).edit_pipeline is None
    assert get(name).edit_schedule is None
    assert get(name).edit_cflags is None
    passes = ["canonicalize", "cse", "scf-forall-to-parallel", "convert-scf-to-openmp"]
    assert apply_pipeline(list(passes), frozenset({name})) == passes
    sched = "module attributes {transform.with_named_sequence} {}\n"
    assert apply_schedule(sched, frozenset({name})) == sched


def test_every_lowering_runner_carries_the_rewrite_and_the_same_argv_gate():
    """`erase_self_copy` read as an inert lever for seven beam rounds because ONE runner variant
    drove the PassManager itself and quietly skipped it. Every variant must splice this prelude and
    read the same argv slot, and the caller must pass that slot."""
    from merlin.llvmlower import accum_microkernel, pipeline

    runners = [pipeline._RUNNER_SRC, pipeline._RUNNER_ACT_POLY_TAIL, accum_microkernel.run_source()]
    for src in runners:
        assert "_parallel_grain" in src, "a runner variant does not carry the rewrite"
        assert "_PARALLEL_GRAIN = int(sys.argv[9])" in src, "a runner variant has its own gate"
        assert "_MID_STAGES, _LATE_STAGES)" in src, "a runner variant never runs the late stage"
    lowering = open(pipeline.__file__, encoding="utf-8").read()
    assert "_concat_dps_gate, _grain_gate]" in lowering, "the argv slot is never passed"


def _run_stages_split(pipeline_text: str, late_labels: tuple[str, ...]) -> list:
    """Drive `selfcopy._run_stages` with a recording stand-in for the PassManager.

    Returns the sequence of events: `('pm', <pass string>)` for each PassManager the function
    builds, and `('late', <label>)` where a late rewrite ran. This is how the frozen-baseline
    invariant is checked at the unit level: with no late stage the pass list must reach ONE
    PassManager, unsplit."""
    import sys as _sys
    import types

    from merlin.llvmlower.selfcopy import RUNNER_PRELUDE as SELFCOPY_PRELUDE

    events: list = []

    class _PM:
        def __init__(self, text):
            inner = text[len("builtin.module("):-1] if text.startswith("builtin.module(") else text
            events.append(("pm", inner))

        @staticmethod
        def parse(text, ctx):
            return _PM(text)

        def run(self, op):
            return None

    module = types.ModuleType("torch_mlir.passmanager")
    module.PassManager = _PM
    parent = types.ModuleType("torch_mlir")
    parent.passmanager = module
    saved = {k: _sys.modules.get(k) for k in ("torch_mlir", "torch_mlir.passmanager")}
    _sys.modules["torch_mlir"] = parent
    _sys.modules["torch_mlir.passmanager"] = module
    try:
        namespace: dict = {"sys": _sys}
        exec(SELFCOPY_PRELUDE, namespace)                       # noqa: S102 — the shipped source
        late = tuple((label, lambda ctx, mod, label=label: events.append(("late", label)) or 0)
                     for label in late_labels)
        namespace["_run_stages"](None, types.SimpleNamespace(operation=None), pipeline_text,
                                 False, (), late)
    finally:
        for k, v in saved.items():
            if v is None:
                _sys.modules.pop(k, None)
            else:
                _sys.modules[k] = v
    return events


#: The multicore tail of `build_rvv_pipeline`, in order.
_PARALLEL_TAIL = ("canonicalize,cse,one-shot-bufferize,func.func(buffer-loop-hoisting),"
                  "scf-forall-to-parallel,func.func(convert-linalg-to-parallel-loops),"
                  "convert-scf-to-openmp,canonicalize,convert-scf-to-cf")


def test_with_no_late_stage_the_pass_list_reaches_one_pass_manager_unsplit():
    """The frozen-baseline invariant, at the level where it is decided: an unflagged build must not
    even see a different pass-manager decomposition."""
    assert _run_stages_split(_PARALLEL_TAIL, ()) == [("pm", _PARALLEL_TAIL)]


def test_the_late_stage_runs_between_the_parallel_loops_and_the_openmp_conversion():
    """A rewrite placed at the `mid` split point would price loops that do not exist yet (nothing is
    an `scf.parallel` until `scf-forall-to-parallel` / `convert-linalg-to-parallel-loops` have run),
    and one placed after `convert-scf-to-openmp` would be too late — the fork is already emitted."""
    events = _run_stages_split(_PARALLEL_TAIL, ("parallel_grain",))
    assert [e[0] for e in events] == ["pm", "late", "pm"]
    before, _, after = events
    assert before[1].endswith("func.func(convert-linalg-to-parallel-loops)")
    assert after[1].startswith("convert-scf-to-openmp")


def test_a_serial_pipeline_still_runs_the_late_stage_instead_of_dropping_it():
    """There is no OpenMP conversion to stop in front of; running the rewrite at the end (where it
    finds nothing and says so) is visible, while dropping it silently is not."""
    events = _run_stages_split("canonicalize,func.func(convert-linalg-to-loops)",
                               ("parallel_grain",))
    assert [e[0] for e in events] == ["pm", "late"]


def _grain(mlir_text: str, threshold: int, tmp_path) -> tuple[str, int, str]:
    """Run the rewrite exactly as the lowering runner does (the m2m venv owns the MLIR bindings).

    Returns (printed module, regions serialized, the runner's report line)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "in.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = tmp_path / "_grain.py"
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        f"_PARALLEL_GRAIN = {int(threshold)}\n" + RUNNER_PRELUDE + "\n"
        "ctx = ir.Context()\n"
        "mod = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
        "n = _parallel_grain(ctx, mod)\n"
        "mod.operation.verify()\n"
        "print('N', n)\n"
        "print('MODULE')\n"
        "print(str(mod.operation))\n", encoding="utf-8")
    proc = subprocess.run([str(toolchain.m2m_python()), str(script), str(src)],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr
    head, _, module = proc.stdout.partition("MODULE\n")
    lines = head.splitlines()
    n = int(next(ln for ln in lines if ln.startswith("N ")).split()[1])
    report = next(ln for ln in lines if ln.startswith("OK parallel_grain"))
    return module, n, report


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_the_cheap_region_is_serialized_and_the_expensive_one_is_kept(tmp_path):
    """The payoff shape: one fork instead of two, and the loop that carries the work keeps its."""
    module, n, report = _grain(CHEAP_AND_EXPENSIVE, 10_000, tmp_path)
    assert n == 1
    assert module.count("scf.parallel") == 1
    assert module.count("scf.for ") == 1, "the serialized 1-D loop should be one scf.for"
    assert "serialized 1 kept 1 unpriceable 0 reducing 0" in report
    # the arithmetic and the stores survive the move
    assert module.count("memref.store") == 2 and module.count("arith.mulf") == 1


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_a_threshold_above_everything_serializes_every_region_and_below_it_serializes_none(tmp_path):
    """The knob has to actually be continuous in both directions, or a search cannot move it."""
    all_serial, n_all, _ = _grain(CHEAP_AND_EXPENSIVE, 10_000_000, tmp_path / "hi")
    assert n_all == 2 and "scf.parallel" not in all_serial
    untouched, n_none, _ = _grain(CHEAP_AND_EXPENSIVE, 1, tmp_path / "lo")
    assert n_none == 0 and untouched.count("scf.parallel") == 2


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_a_loop_it_cannot_price_keeps_its_parallelism(tmp_path):
    """Fail closed. A dynamic bound is not evidence that the loop is small."""
    module, n, report = _grain(DYNAMIC_BOUND, 10_000_000, tmp_path)
    assert n == 0
    assert module.count("scf.parallel") == 1
    assert "unpriceable 1" in report


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_a_region_carrying_a_reduction_is_refused(tmp_path):
    """Its `scf.reduce` terminator would have to be rewritten too; refuse rather than half-do it."""
    module, n, report = _grain(WITH_REDUCTION, 10_000_000, tmp_path)
    assert n == 0
    assert module.count("scf.parallel") == 1
    assert "reducing 1" in report
