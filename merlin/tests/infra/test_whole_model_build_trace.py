"""The whole-model builder's stages are compile-trace stages: declared where the build times them, dumped
with the files each wrote, and a point the build stops at; its CLI and the service builder say so.

Nothing here names a target: the build's own collaborators are replaced where a real one would need a
target's toolchain.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from merlin.common import compile_trace as T
from merlin.perf import whole_model_build as W


def test_the_declared_build_stages_are_the_ones_build_times():
    """The declaration IS the clock's vocabulary: every ``with stages("...")`` in :func:`W.build`, in order."""
    tree = ast.parse(inspect.getsource(W.build).strip())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "stages"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    timed = [node.args[0].value for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset))]
    assert sorted(timed, key=list(W.BUILD_STAGES).index) == timed
    assert set(timed) == set(W.BUILD_STAGES)


def test_a_traced_build_stage_keeps_what_it_wrote_and_can_stop_the_build(tmp_path):
    out = tmp_path / "out"
    (out / "lower").mkdir(parents=True)
    (out / "lower" / "old.txt").write_text("before the build")
    request = T.Request(directory=str(tmp_path / "trace"), dump_after=("statement",), stop_after="group_objects")
    clock = W._StageClock()
    clock.root = out
    with pytest.raises(T.StopAfterStage) as stopped:
        with T.session(request):
            with clock("statement"):
                (out / "lower" / "g1.iface.mlir").write_text("module {}")
            with clock("group_objects"):
                (out / "objects").mkdir()
                (out / "objects" / "g1.o").write_bytes(b"\x7fELF")
            with clock("extract"):
                raise AssertionError("the build ran past the stage it was told to stop after")
    events = json.loads((tmp_path / "trace" / T.INDEX).read_text())["stages"]
    assert [e["stage"] for e in events] == ["statement", "group_objects"]
    statement = events[0]
    assert statement["sources"] == [str(out / "lower" / "g1.iface.mlir")]  # not the file that was there before
    assert (tmp_path / "trace" / statement["files"][0]).read_text() == "module {}"
    assert Path(stopped.value.files[0]).read_bytes() == b"\x7fELF"


def test_the_builder_cli_lists_stages_and_reports_a_stop_as_a_stop(monkeypatch, tmp_path, capsys):
    from merlin.perf import whole_model_build_cli as CLI

    assert CLI.main(["--list-stages"]) == 0
    assert "statement" in capsys.readouterr().out

    def fake_build(*args, **kwargs):
        T.artifact("statement", [], pipeline="whole-model")  # the stop stage, reached
        raise AssertionError("the build ran past the stage it was told to stop after")

    monkeypatch.setattr(CLI, "build", fake_build)
    argv = [
        "build",
        "--capsule",
        str(tmp_path / "m"),
        "--target",
        "fixture",
        "--machine",
        "mach",
        "--header",
        "h.h",
        "--stop-after",
        "statement",
        "--trace-dir",
        str(tmp_path / "trace"),
    ]
    assert CLI.main(argv) == 0
    assert "stopped after stage 'statement'" in capsys.readouterr().out
    assert json.loads((tmp_path / "trace" / T.INDEX).read_text())["outcome"] == T.STOPPED


def test_the_service_builder_turns_a_stop_into_a_refusal(monkeypatch, tmp_path):
    """Where a program is what the caller is owed (the measurement service), a stopped build is a refusal."""
    from merlin.perf import whole_model_builder as B
    from merlin.perf import whole_model_open as WO

    def fake_build(package_dir, model_capsule, **kwargs):
        T.artifact("statement", [], pipeline="whole-model")
        raise AssertionError("unreachable")

    monkeypatch.setattr(W, "build", fake_build)
    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: False)
    with pytest.raises(W.WholeModelBuildError, match="stopped after stage 'statement'"):
        B.build(
            "pkg",
            target="fixture",
            out_dir=tmp_path / "out",
            model_capsule="m",
            machine="mach",
            header="h.h",
            trace_dir=str(tmp_path / "trace"),
            stop_after="statement",
        )
