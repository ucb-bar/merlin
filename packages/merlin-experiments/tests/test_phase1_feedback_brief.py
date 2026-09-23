"""Package-owned feedback publication; synthetic redacted inputs, no agents/oracles."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import module_source_path, python_import_roots


def test_cold_writer_and_recovery_refresh_outside_checkout(tmp_path):
    run, workspace = tmp_path / "run", tmp_path / "workspace"
    history = run / "qa_history"
    history.mkdir(parents=True)
    notes = workspace / "submission/docs/iteration_notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("our own finding\n" * 400)
    for index, passing in enumerate((True, False)):
        (history / f"verdict_round_{index:02d}.json").write_text(
            json.dumps(
                {
                    "n_passed": int(passing),
                    "n_capsules": 1,
                    "first_failure_planes": {} if passing else {"numeric": 1},
                    "per_capsule": [
                        {
                            "capsule": "public-case",
                            "status": "pass" if passing else "fail",
                            "failure_plane": "numeric",
                            "mismatch_count": 0 if passing else 3,
                        }
                    ],
                }
            )
        )
    (history / "verdict_round_invalid.json").write_text("invalid JSON")
    (history / "verdict_fast_999.json").write_text('{"n_passed": "FAST_SENTINEL"}')
    private = run / "hidden/golden.yaml"
    private.parent.mkdir()
    private.write_text("answer: PRIVATE_SENTINEL\n")
    program = r"""
import importlib.abc, json, os, pathlib, sys
sys.path[:0] = json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'_common', 'round_brief', 'run_baseline_qa_loop', 'resume_on_quota'}:
            raise AssertionError('native import: ' + fullname)
sys.meta_path.insert(0, NoNative())
before = dict(os.environ)
from merlin_experiments.phase1.feedback import brief
from merlin_experiments.phase1 import recovery
assert dict(os.environ) == before
run, ws = map(pathlib.Path, sys.argv[2:])
reads = []
original_read = pathlib.Path.read_text
def read(path, *args, **kwargs):
    reads.append(path)
    return original_read(path, *args, **kwargs)
pathlib.Path.read_text = read
path = brief.write(run, ws, 1)
assert path == ws / 'qa/round_brief.md'
first = path.read_text()
assert 'BELOW your best round' in first and 'public-case' in first
assert 'mismatch_count 3' in first
assert 'did NOT update' not in first
brief.write(run, ws, 1)
assert 'did NOT update' in path.read_text()
stamp = (ws / 'qa/.notes_hash').read_bytes()
recovery.prepend_resume_note(ws, recovery.REASON_TIMEOUT)
(run / 'ERRATA.md').write_text('## RETRACTION: re-derive the earlier advice\n')
brief.refresh_before_launch(run, ws, 1)
refreshed = path.read_text()
assert refreshed.startswith('> ## RESUME') and refreshed.count('> ## RESUME') == 1
assert refreshed.count('RETRACTION') == 1
assert refreshed.index('RETRACTION') < refreshed.index('Your iteration_notes.md')
assert 'did NOT update' not in refreshed
assert (ws / 'qa/.notes_hash').read_bytes() == stamp
assert 'PRIVATE_SENTINEL' not in refreshed and 'FAST_SENTINEL' not in refreshed
assert all('hidden' not in path.parts for path in reads)
brief.refresh_before_launch(run, ws, 1)
assert path.read_text() == refreshed
brief.write(run, ws, 2)
assert not path.read_text().startswith('> ## RESUME')
print('brief-publication-qualified')
"""
    dependencies = [p for p in sys.path if p and Path(p).is_dir() and "site-packages" in p]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            json.dumps([*map(str, python_import_roots()), *dependencies]),
            str(run),
            str(workspace),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "brief-publication-qualified"
    assert {p.name for p in (workspace / "qa").iterdir()} == {"round_brief.md", ".notes_hash"}


def test_brief_implementation_is_masked_not_staged_as_candidate_sdk(tmp_path, monkeypatch):
    from merlin.common import access
    from merlin.targetgen.sandbox import bwrap

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    private = tmp_path / "packages/merlin-experiments/src/merlin_experiments/phase1/feedback/brief.py"
    private.parent.mkdir(parents=True)
    private.write_text("# trusted host feedback construction\n")
    monkeypatch.setattr(access, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(surfaces_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces_module, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces_module, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces_module, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    policy = SimpleNamespace(
        target="fixture",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    surfaces = surfaces_module.answer_surfaces(policy)
    grants = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert any(private.is_relative_to(surface.path) for surface in bwrap.coverage_gap(grants, surfaces))
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(grants, surfaces), surfaces) == []


def test_source_receipt_binds_actual_brief_bytes(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    package = tmp_path / "phase1"
    shutil.copytree(
        module_source_path("merlin_experiments.phase1").parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    original = SI._source

    def copied_source(module):
        if module == "merlin_experiments.phase1":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.phase1."):
            path = package.joinpath(*module.split(".")[2:])
            return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        return original(module)

    monkeypatch.setattr(SI, "_source", copied_source)
    inputs = {"repo": tmp_path, "entrypoint": tmp_path / "synthetic_transport.py"}
    record = SI.record(**inputs)
    member = package / "feedback/brief.py"
    assert record["inputs"]["phase1:source:feedback/brief.py"]["path"] == str(member)
    SI.verify(record, **inputs)
    member.write_text(member.read_text() + "# changed feedback owner\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **inputs)
