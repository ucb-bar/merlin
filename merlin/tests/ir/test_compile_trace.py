"""A compile stops at a named stage with that stage's IR written, and a traced compile indexes every stage.

These drive the REAL lowering routes (the LLVM route through the native pass manager, and the staged
xDSL route) on a one-op module, so what is held is what a user of ``--stop-after`` gets: the IR at the
stage, nothing after it kept, and a trace index whose stage list is the route's own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from merlin.common import compile_trace as T
from merlin.common.paths import python_import_roots

_MODULE = """
func.func @forward(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %e = tensor.empty() : tensor<4xf32>
  %r = linalg.add ins(%a, %b : tensor<4xf32>, tensor<4xf32>) outs(%e : tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
"""


def _lower(work: Path):
    from merlin.llvmlower.lower import lower_model

    return lower_model(_MODULE, work, targets=())


def _index(trace: Path) -> dict:
    return json.loads((trace / T.INDEX).read_text(encoding="utf-8"))


def _stop(tmp_path: Path, stage: str) -> tuple[T.StopAfterStage, Path, Path]:
    trace, work = tmp_path / "trace", tmp_path / "work"
    with pytest.raises(T.StopAfterStage) as stopped:
        with T.session(T.Request(directory=str(trace), stop_after=stage)):
            _lower(work)
    return stopped.value, trace, work


@pytest.mark.parametrize(
    ("stage", "holds"),
    [
        ("xdsl-pruned", "linalg.add"),  # an xDSL rewrite: the module is still on tensors
        ("mlir:one-shot-bufferize", "memref"),  # a native pass: bufferized, still before any LLVM
        ("mlir:cse#2", "llvm.func"),  # the SECOND cse of the pipeline, after LLVM conversion
        ("llvm-final", "define"),  # the LLVM IR the object would be compiled from
    ],
)
def test_stop_after_writes_that_stages_ir_and_nothing_after_it(tmp_path, stage, holds):
    stop, trace, work = _stop(tmp_path, stage)
    name, _ = T.parse_selector(stage)
    assert stop.stage.startswith(name)
    assert stop.files and all(Path(f).is_file() for f in stop.files)
    assert holds in Path(stop.files[0]).read_text(encoding="utf-8")
    index = _index(trace)
    assert index["outcome"] == T.STOPPED
    assert index["stages"][-1]["stage"] == name  # the stop is the last stage reached
    # No final artifact: no object, no shared library, and a native stop discards the pass manager's
    # output (the IR after its LAST pass), so the workdir never holds a complete LLVM module.
    assert not list(work.glob("*.o")) and not list(work.glob("*.so"))
    if stage.startswith(T.MLIR_PREFIX) or stage == "xdsl-pruned":
        assert not (work / "model.ll").exists()


def test_dump_ir_after_all_indexes_every_stage_in_route_order(tmp_path):
    from merlin.llvmlower import lower as L
    from merlin.llvmlower import pipeline as P

    trace = tmp_path / "trace"
    with T.session(T.Request(directory=str(trace), dump_after=(T.ALL,))):
        _lower(tmp_path / "work")
    index = _index(trace)
    assert index["outcome"] == T.COMPLETED
    reached = [e["stage"] for e in index["stages"]]
    # The route's declared stages, in declared order (a serial build reaches llvm-translated, not
    # llvm-dialect), and every native pass of the pipeline the lowering built, in pipeline order.
    declared = [s for s in reached if not s.startswith(T.MLIR_PREFIX)]
    assert declared == [s for s in L.STAGES if s in declared]
    assert set(declared) >= {"input", "xdsl-parsed", "upstream", "upstream-scheduled", "llvm-final"}
    native = [s[len(T.MLIR_PREFIX) :] for s in reached if s.startswith(T.MLIR_PREFIX)]
    assert native == T.split_pipeline(P._upstream_pipeline())
    for event in index["stages"]:
        files = event.get("files") or [event.get("file")]
        assert all((trace / f).is_file() for f in files if f), event
    assert index["timing_seconds"] and (trace / T.PIPELINE).is_file()
    assert index["native_segments"] and "one-shot-bufferize" in index["native_segments"][0]["pipeline"]


def test_dump_ir_before_writes_the_previous_stage_and_the_native_before_dump(tmp_path):
    trace = tmp_path / "trace"
    with T.session(T.Request(directory=str(trace), dump_before=("upstream", "mlir:convert-scf-to-cf"))):
        _lower(tmp_path / "work")
    events = _index(trace)["stages"]
    upstream = next(e for e in events if e["stage"] == "upstream")
    assert upstream["before_is"] == "xdsl-c-interface" and (trace / upstream["before_file"]).is_file()
    native = [e for e in events if e["stage"] == "mlir:convert-scf-to-cf"]
    assert [e["when"] for e in native] == ["before"]
    assert "scf." in (trace / native[0]["files"][0]).read_text(encoding="utf-8")


def test_a_stage_the_route_never_reaches_completes_and_says_so(tmp_path):
    trace = tmp_path / "trace"
    with T.session(T.Request(directory=str(trace), stop_after="contract")):  # a staged-route stage
        _lower(tmp_path / "work")
    assert _index(trace)["outcome"] == T.NOT_REACHED


def test_the_staged_route_records_exactly_its_declared_stages(tmp_path):
    from merlin.compile_core import compile_core_mlir
    from merlin.xdsl_dialects.lowering import pipeline
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module

    trace = tmp_path / "trace"
    with T.session(T.Request(directory=str(trace), dump_after=(T.ALL,))):
        compile_core_mlir(build_input_module(reuse=1, m=2, k=2, n=2), target="toy_npu")
    events = _index(trace)["stages"]
    assert [e["stage"] for e in events] == list(pipeline.STAGES)
    with pytest.raises(T.StopAfterStage) as stopped:
        with T.session(T.Request(directory=str(tmp_path / "stop"), stop_after="schedule")):
            compile_core_mlir(build_input_module(reuse=1, m=2, k=2, n=2), target="toy_npu")
    # The stopped compile's schedule IR is byte-for-byte the full compile's: the stop changes nothing before it.
    full = next(trace / e["file"] for e in events if e["stage"] == "schedule")
    assert Path(stopped.value.files[0]).read_bytes() == full.read_bytes()
    assert [e["stage"] for e in _index(tmp_path / "stop")["stages"]] == list(pipeline.STAGES[:3])


def test_a_child_process_dumps_into_the_trace_and_only_the_owner_stops(tmp_path):
    """A stage reached in a child (a package entrypoint) is dumped under ir/p<pid>/ and recorded as the
    stop; the child is NOT stopped (its caller would read that as a failed group), the owner is."""
    trace = tmp_path / "trace"
    script = textwrap.dedent(
        f"""
        from merlin.llvmlower.lower import lower_model
        lower_model({_MODULE!r}, {str(tmp_path / "child")!r}, targets=())
        print("child finished")
        """
    )
    with pytest.raises(T.StopAfterStage) as stopped:
        with T.session(T.Request(directory=str(trace), stop_after="upstream")):
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in python_import_roots()))
            done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env=env)
            assert done.returncode == 0 and "child finished" in done.stdout, done.stderr
            T.stop_if_reached()
    assert "ir/p" in stopped.value.files[0] and Path(stopped.value.files[0]).is_file()
    assert "reached in process" in stopped.value.note


def test_a_trace_directory_must_be_new():
    with pytest.raises(T.TraceError, match="not empty"):
        with T.session(T.Request(directory=str(Path(__file__).parent))):
            pass


@pytest.mark.parametrize(
    ("text", "passes"),
    [
        ("canonicalize,cse", ["canonicalize", "cse"]),
        ("builtin.module(a,func.func(b,c{x=1 y=2}),d)", ["a", "b", "c", "d"]),
        (
            "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
            ["one-shot-bufferize"],
        ),
        ("transform-interpreter{entry-point=a,b},cse", ["transform-interpreter", "cse"]),
    ],
)
def test_a_pass_pipeline_is_split_by_bracket_depth(text, passes):
    assert T.split_pipeline(text) == passes


def test_selectors_name_known_stages_and_occurrences():
    known = {"input", "contract"}
    assert T.validate(["input,mlir:cse#2", "all"], known, allow_all=True) == ("input", "mlir:cse#2", "all")
    with pytest.raises(T.TraceError, match="unknown stage"):
        T.validate(["not-a-stage"], known, allow_all=True)
    with pytest.raises(T.TraceError, match="one stage"):
        T.validate(["all"], known, allow_all=False)
    with pytest.raises(T.TraceError, match="positive"):
        T.parse_selector("input#0")
    request = T.Request(directory="unused", dump_after=("mlir:cse#2",), stop_after="input")
    assert request.dumps_after("mlir:cse", 2) and not request.dumps_after("mlir:cse", 1)
    assert request.stops_at("input", 1) and not request.stops_at("input", 2)
