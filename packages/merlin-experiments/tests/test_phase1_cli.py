"""Installed command parsing stays inert until explicit execution inputs exist."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("arguments, expected", [(["--help"], 0), (["--run-id", "missing"], 2)])
def test_cli_parsing_does_not_initialize_context(tmp_path, arguments, expected):
    program = """
import importlib.abc, os, runpy, subprocess, sys
class DenyExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'merlin_experiments.phase1.context', 'merlin_experiments.phase1.controller',
                        '_common', 'run_baseline_qa_loop'}:
            raise AssertionError('execution import during parsing: ' + fullname)
sys.meta_path.insert(0, DenyExecution())
def forbidden(*args, **kwargs): raise AssertionError('process launched during parsing')
subprocess.Popen = forbidden
before = dict(os.environ)
try:
    runpy.run_module('merlin_experiments.phase1', run_name='__main__')
finally:
    assert dict(os.environ) == before
"""
    result = subprocess.run(
        [sys.executable, "-c", program, *arguments],
        cwd=tmp_path,
        env=dict(os.environ, MERLIN_TARGET_EXPERIMENT=str(tmp_path / "missing.yaml")),
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert "--descriptor" in result.stdout + result.stderr


def test_controller_cannot_attribute_foreign_entrypoint_to_package_sources(tmp_path):
    from merlin_experiments.phase1.controller import run

    # Refuse before execution imports, context access or workspace mutation.
    with pytest.raises(ValueError, match="override requires native source verification"):
        run(
            None,
            None,
            bundle_manifest=tmp_path / "input_bundle_manifest.yaml",
            bundle_id="fixture",
            oracle_timing=tmp_path / "timing.json",
            source_entrypoint=tmp_path / "foreign.py",
        )
    assert list(tmp_path.iterdir()) == []
