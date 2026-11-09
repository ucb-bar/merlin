from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.common import compile_trace as T
from merlin.llvmlower import toolchain

native = pytest.mark.skipif(
    not toolchain.compiler_python().is_file(), reason="the compiler's Python (native MLIR pass manager) is absent"
)
_MODULE = "\nfunc.func @forward(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {\n  %e = tensor.empty() : tensor<4xf32>\n  %r = linalg.add ins(%a, %b : tensor<4xf32>, tensor<4xf32>) outs(%e : tensor<4xf32>) -> tensor<4xf32>\n  return %r : tensor<4xf32>\n}\n"


def _lower(work: Path):
    from merlin.llvmlower.lower import lower_model

    return lower_model(_MODULE, work, targets=())


def _index(trace: Path) -> dict:
    return json.loads((trace / T.INDEX).read_text(encoding="utf-8"))


def _stop(tmp_path: Path, stage: str) -> tuple[T.StopAfterStage, Path, Path]:
    trace, work = (tmp_path / "trace", tmp_path / "work")
    with pytest.raises(T.StopAfterStage) as stopped:
        with T.session(T.Request(directory=str(trace), stop_after=stage)):
            _lower(work)
    return (stopped.value, trace, work)


@pytest.mark.parametrize("stage,holds", [("mlir:cse#2", "memref")])
def test_stop_after_writes_that_stages_ir_and_nothing_after_it(tmp_path, stage, holds):
    stop, trace, work = _stop(tmp_path, stage)
    name, _ = T.parse_selector(stage)
    assert stop.stage.startswith(name)
    assert stop.files and all((Path(f).is_file() for f in stop.files))
    assert holds in Path(stop.files[0]).read_text(encoding="utf-8")
    index = _index(trace)
    assert index["outcome"] == T.STOPPED
    assert index["stages"][-1]["stage"] == name
    assert not list(work.glob("*.o")) and (not list(work.glob("*.so")))
    if stage.startswith(T.MLIR_PREFIX) or stage == "xdsl-pruned":
        assert not (work / "model.ll").exists()


@native
def test_dump_ir_after_all_indexes_every_stage_in_route_order(tmp_path):
    from merlin.compiler.lower.native import pipeline as P
    from merlin.llvmlower import lower as L

    trace = tmp_path / "trace"
    with T.session(T.Request(directory=str(trace), dump_after=(T.ALL,))):
        _lower(tmp_path / "work")
    index = _index(trace)
    assert index["outcome"] == T.COMPLETED
    reached = [e["stage"] for e in index["stages"]]
    declared = [s for s in reached if not s.startswith(T.MLIR_PREFIX)]
    assert declared == [s for s in L.STAGES if s in declared]
    assert set(declared) >= {"input", "xdsl-parsed", "upstream", "upstream-scheduled", "llvm-final"}
    native = [s[len(T.MLIR_PREFIX) :] for s in reached if s.startswith(T.MLIR_PREFIX)]
    boundaries = {P.SPLIT_FRESH_DEALLOCS_MARKER, P.SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER}
    assert not boundaries.intersection(native)
    assert native == [p for segment in index["native_segments"] for p in T.split_pipeline(segment["pipeline"])]
    for event in index["stages"]:
        files = event.get("files") or [event.get("file")]
        assert all(((trace / f).is_file() for f in files if f)), event
    assert index["timing_seconds"] and (trace / T.PIPELINE).is_file()
    assert any(("one-shot-bufferize" in segment["pipeline"] for segment in index["native_segments"]))
