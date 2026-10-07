"""``merlin-compile`` exposes the compile trace: --list-stages, --dump-ir-after/--dump-ir-before,
--stop-after and --trace-dir reach a REAL lowering and report a stop as a stop (exit 0, no artifact).

The workflow under the front door is replaced by a real ``lower_model`` of a one-op module (a whole
captured model is not needed to hold the wiring); everything from the argument parser down to the
native pass printer is the shipped code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin import compile_cli
from merlin.common import compile_trace as T
from merlin.compile import debug

_MODULE = """
func.func @forward(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %e = tensor.empty() : tensor<4xf32>
  %r = linalg.add ins(%a, %b : tensor<4xf32>, tensor<4xf32>) outs(%e : tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
"""


@pytest.fixture
def lowering(monkeypatch, tmp_path):
    """The selected workflow becomes a real lowering into ``tmp_path/work``; ``reached`` says whether it
    returned (a stopped compile never does)."""
    from merlin.llvmlower.lower import lower_model

    state = {"reached": False}

    def workflow(a, run):
        result = lower_model(_MODULE, tmp_path / "work", targets=())
        state["reached"] = True
        return {
            "tool": "merlin-compile",
            "target": a.target,
            "workload": a.workload,
            "status": "compiled",
            "binary": str(result.ll_path),
        }

    monkeypatch.setattr(compile_cli, "_workflow", workflow)
    return state


def _args(*extra: str) -> list[str]:
    return ["--workload", "fixture", "--run", "none", *extra]


def test_list_stages_is_read_off_the_pipelines(capsys):
    from merlin.llvmlower import pipeline as P

    assert compile_cli.main(["--list-stages", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    stages = [row["stage"] for row in rows]
    for decl in T.declared():  # every declared stage of every pipeline, in its declared order
        listed = [row["stage"] for row in rows if row["pipeline"] == decl["pipeline"]]
        assert listed == list(decl["stages"]), decl["pipeline"]
    assert {"capture", "contract", "xdsl-pruned", "llvm-final", "object", "link", "statement"} <= set(stages)
    serial = T.with_ordinals(T.split_pipeline(P._upstream_pipeline()))
    assert all(T.MLIR_PREFIX + name in stages for name in serial)
    assert compile_cli.main(["--list-stages"]) == 0
    assert "mlir:canonicalize#2" in capsys.readouterr().out


def test_an_unknown_stage_is_refused_before_anything_runs(lowering):
    with pytest.raises(SystemExit) as exited:
        compile_cli.main(_args("--stop-after", "not-a-stage"))
    assert exited.value.code == 2 and not lowering["reached"]


def test_stop_after_exits_zero_with_the_ir_and_no_artifact(lowering, tmp_path, capsys):
    trace = tmp_path / "trace"
    assert compile_cli.main(_args("--stop-after", "mlir:one-shot-bufferize", "--trace-dir", str(trace))) == 0
    out = capsys.readouterr().out
    assert "stopped after stage 'mlir:one-shot-bufferize#1'" in out and "no final artifact" in out
    assert not lowering["reached"]  # the workflow never returned a result
    assert not (tmp_path / "work" / "model.ll").exists()
    index = json.loads((trace / T.INDEX).read_text())
    assert index["outcome"] == T.STOPPED and index["command"][:2] == ["merlin-compile", "--workload"]
    assert all((trace / f).is_file() for f in index["stop"]["files"])


def test_dump_ir_after_all_writes_a_trace_index_and_reports_it(lowering, tmp_path, capsys):
    trace = tmp_path / "trace"
    code = compile_cli.main(
        _args("--dump-ir-after", "all", "--dump-ir-before", "llvm-final", "--trace-dir", str(trace), "--json")
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["status"] == "compiled" and lowering["reached"]
    assert result["trace"] == str(trace / T.INDEX)
    index = json.loads((trace / T.INDEX).read_text())
    reached = [e["stage"] for e in index["stages"]]
    assert reached[0] == "input" and reached[-1] == "llvm-final" and "mlir:cse" in reached
    final = next(e for e in index["stages"] if e["stage"] == "llvm-final")
    assert final["before_is"] == "llvm-normalized" and (trace / final["before_file"]).is_file()
    assert (trace / T.PIPELINE).is_file() and index["pass_log"].endswith(T.PASS_LOG)


def test_a_stop_stage_the_route_never_reaches_is_not_a_success(lowering, tmp_path, capsys):
    code = compile_cli.main(_args("--stop-after", "contract", "--trace-dir", str(tmp_path / "trace"), "--json"))
    result = json.loads(capsys.readouterr().out)
    assert code == 1 and result["status"] == "stop_stage_not_reached" and "never reached" in result["reason"]


def test_without_trace_dir_the_trace_goes_under_the_out_root(lowering, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    assert compile_cli.main(_args("--dump-ir-after", "upstream", "--json")) == 0
    trace = Path(json.loads(capsys.readouterr().out)["trace"])
    assert trace.is_relative_to(tmp_path / "out" / "artifacts" / "probes" / "compile-trace" / "rvv")
    assert [e["stage"] for e in json.loads(trace.read_text())["stages"] if e.get("file")] == ["upstream"]


def test_a_launch_config_spells_the_same_request():
    request = debug.request_from_options(
        {"dump_ir_after": "contract,mlir:cse#2", "stop_after": "statement", "trace_dir": "/unused"},
        target="fixture",
        workload="m",
    )
    assert request.dump_after == ("contract", "mlir:cse#2") and request.stop_after == "statement"
    assert debug.request_from_options({}, target="fixture", workload="m") is None
