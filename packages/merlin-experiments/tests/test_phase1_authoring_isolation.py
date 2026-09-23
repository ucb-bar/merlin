"""Extracted authoring has no import-time execution or candidate-visible policy."""

import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.common import access
from merlin.common.paths import python_import_roots
from merlin.targetgen.sandbox import bwrap


def test_authoring_import_is_native_target_and_process_inert(tmp_path):
    program = """
import importlib.abc, os, socket, subprocess, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'_common', '_pbcommon', 'run_agent_experiment',
                        'run_baseline_qa_loop', 'tooling_readiness', 'chia', 'ray'}:
            raise AssertionError('import crossed native/scheduler seam: ' + fullname)
sys.meta_path.insert(0, NoNative())
def refused(*args, **kwargs):
    raise AssertionError('import attempted process or network activity')
subprocess.run = subprocess.Popen = refused
socket.socket = refused
before = dict(os.environ)
from merlin_experiments.phase1 import authoring, audit, runtime_environment, task_staging
from merlin_experiments.phase1.feedback import certification
assert dict(os.environ) == before
assert callable(authoring.execute)
assert callable(audit.AnswerAudit.for_descriptor)
assert callable(runtime_environment.prepare_runtime_environment)
assert callable(task_staging.callbacks)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
        "MERLIN_REPO_ROOT": str(tmp_path),
        "MERLIN_TARGET_EXPERIMENT": str(tmp_path / "missing-descriptor.yaml"),
    }
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "member", ["authoring.py", "audit.py", "runtime_environment.py", "task_staging.py", "feedback/certification.py"]
)
def test_authoring_policy_is_physically_masked_without_masking_public_clients(tmp_path, monkeypatch, member):
    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    monkeypatch.setattr(access, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(surfaces_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces_module, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces_module, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces_module, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    root = tmp_path / "packages/merlin-experiments/src/merlin_experiments/phase1"
    private = root / member
    public = root / "tools/selfcheck.py"
    for path in (private, public):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source\n")
    policy = SimpleNamespace(
        target="synthetic",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    surfaces = surfaces_module.answer_surfaces(policy)
    masked = private.parent if member.startswith("feedback/") else private
    assert any(surface.path == masked and surface.origin == "grader" for surface in surfaces)
    assert not any(surface.path == public or surface.path in public.parents for surface in surfaces)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in bwrap.coverage_gap(unmasked, surfaces)} == {masked}
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(unmasked, surfaces), surfaces) == []
