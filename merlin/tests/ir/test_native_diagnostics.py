"""Diagnostic wrappers retain the selected compiler, exact check source and public errors."""

import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower import pipeline


@pytest.mark.parametrize("kind", ["apply", "dealloc"])
def test_diagnostic_uses_actual_tool_and_production_source(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", lambda **kw: str(tmp_path))
    monkeypatch.setattr(pipeline, "m2m_python", lambda: Path("/selected/compiler/python"))
    observed = []

    def execute(command, **kwargs):
        observed.append((command, kwargs))
        assert command[0] == "/selected/compiler/python"
        script = Path(command[1]).read_text()
        assert Path(command[2]).read_text() == "module {}"
        if kind == "apply":
            assert "class _FGPassManager:" in script
            assert command[-1] == "canonicalize,cse"
            Path(command[3]).write_text("printed module")
        else:
            assert pipeline.DEALLOC_CHECK_PRELUDE in script
            assert "_dealloc_placement_violations(module)" in script
        return subprocess.CompletedProcess(command, 0, 'MERLIN_CHECK {"violations": ["diagnostic"]}\n', "")

    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    if kind == "apply":
        assert pipeline.apply_passes("module {}", "canonicalize,cse", timeout=23) == "printed module"
    else:
        assert pipeline.dealloc_placement_violations("module {}", timeout=23) == ["diagnostic"]
    assert observed[0][1] == {"capture_output": True, "text": True, "timeout": 23}


@pytest.mark.parametrize("kind", ["apply", "dealloc"])
def test_diagnostic_failure_keeps_public_error_and_output(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", lambda **kw: str(tmp_path))
    monkeypatch.setattr(pipeline, "m2m_python", lambda: Path("/selected/python"))
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 7, "out", "err")
    )
    with pytest.raises(pipeline.PipelineError, match="out\nerr"):
        if kind == "apply":
            pipeline.apply_passes("module {}", "canonicalize")
        else:
            pipeline.dealloc_placement_violations("module {}")


def test_success_without_apply_output_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", lambda **kw: str(tmp_path))
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", "")
    )
    with pytest.raises(pipeline.PipelineError, match="apply_passes failed"):
        pipeline.apply_passes("module {}", "canonicalize")
