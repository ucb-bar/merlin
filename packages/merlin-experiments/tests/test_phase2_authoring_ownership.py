"""Installed authoring imports without checkout discovery or native controllers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2 import authoring as AUTHORING
from merlin_experiments.phase2 import candidate_record as RECORD

from merlin.common.paths import python_import_roots
from merlin.targetgen.sandbox import preflight as SANDBOX_PREFLIGHT
from merlin.targetgen.sandbox.answer_surfaces import DroppedDeclaration


def test_authoring_import_requires_no_checkout_or_native_engine(tmp_path):
    program = """
import importlib.abc, inspect, json, subprocess, sys
sys.path[:0] = json.loads(sys.argv[1])
from merlin.common import paths
def forbidden(*args, **kwargs):
    raise AssertionError('authoring import discovered a checkout or launched a process')
paths.repo_root = paths.merlin_dir = forbidden
subprocess.Popen = forbidden
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name in {'perf_agent_stage', '_pbcommon', '_common', 'run_paired_perf_bench',
                    'run_agentic_perf_experiment', 'heldout_gsim_qualification',
                    'produce_gsim_certificate', 'functional_gsim_qualification', 'functional_coverage',
                    'perf_holdout_corpus'}:
            raise AssertionError('native dependency: ' + name)
sys.meta_path.insert(0, NoNative())
from merlin_experiments.phase2 import authoring
from merlin_experiments.phase2 import functional_cohort, functional_coverage, functional_qualification, holdout_corpus
assert callable(functional_qualification.produce_functional_certificate)
assert callable(holdout_corpus.commit_holdout)
parameters = inspect.signature(authoring.run_stage).parameters
for name in ('contract_root', 'source_root', 'suite', 'functional_runs_root', 'stage_root'):
    assert parameters[name].default is inspect.Parameter.empty, name
assert callable(authoring.functional_emission_guard)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, json.dumps([str(path) for path in python_import_roots()])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_authoring_containment_gate_and_sealed_evidence(monkeypatch):
    binary = Path("/synthetic/bwrap")
    target = object()
    usable = SANDBOX_PREFLIGHT.SandboxProbe("ok", "synthetic")
    denied = SANDBOX_PREFLIGHT.SandboxProbe("inoperable", "userns_uid_map_denied")
    monkeypatch.setattr(AUTHORING.SANDBOX_PREFLIGHT, "require_working_sandbox", lambda *a, **k: usable)
    monkeypatch.setattr(AUTHORING, "dropped_declarations", lambda _target: [])
    observed, drops = AUTHORING._require_stage_containment(binary, target)
    assert observed == usable and drops == []
    sealed = {"preflight": observed.as_record(), "dropped_declarations": []}
    RECORD.verify_sandbox_containment_evidence(sealed)
    RECORD.verify_sandbox_containment_evidence({}, require_present=False)  # historical inspection

    monkeypatch.setattr(
        AUTHORING.SANDBOX_PREFLIGHT,
        "require_working_sandbox",
        lambda *a, **k: (_ for _ in ()).throw(SANDBOX_PREFLIGHT.SandboxUnavailable(denied)),
    )
    with pytest.raises(AUTHORING.StageGateError, match="cannot be built"):
        AUTHORING._require_stage_containment(binary, target)
    monkeypatch.setattr(AUTHORING.SANDBOX_PREFLIGHT, "require_working_sandbox", lambda *a, **k: usable)
    monkeypatch.setattr(
        AUTHORING, "dropped_declarations", lambda _target: [DroppedDeclaration("prior_backend", "stale", "absent")]
    )
    with pytest.raises(AUTHORING.StageGateError, match="masked nothing"):
        AUTHORING._require_stage_containment(binary, target)
    for invalid in (
        {"preflight": denied.as_record(), "dropped_declarations": []},
        {"preflight": usable.as_record(), "dropped_declarations": [{"origin": "prior_backend"}]},
        {"preflight": usable.as_record()},
    ):
        with pytest.raises(RECORD.StageGateError):
            RECORD.verify_sandbox_containment_evidence(invalid, require_present=False)
