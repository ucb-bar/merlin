"""Package-owned conformance replay uses persisted evidence, without a native target."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import module_source_path, python_import_roots


def _environment():
    return dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, python_import_roots())),
        MERLIN_TARGET_EXPERIMENT="/absent/conformance-test-descriptor.yaml",
    )


@pytest.mark.parametrize("case", ["complete", "missing-result", "failed-isa", "regex", "invalid-python"])
def test_canonical_cli_replays_persisted_evidence_outside_checkout(tmp_path, case):
    run = tmp_path / "recorded-run"
    rounds = run / "rounds"
    rounds.mkdir(parents=True)
    submission = run / "submission"
    submission.mkdir()
    code = {
        "regex": "import re\nre.compile('example')\n",
        "invalid-python": "def broken(:\n",
    }.get(case, "import ast\n")
    (submission / "backend.py").write_text(code)
    (run / "run_manifest.yaml").write_text(json.dumps({"arm": "merlin_assisted", "endpoint_kind": "external_backend"}))
    # The saved tool set, not the historical arm spelling, selects the actual checks.
    (run / "environment.yaml").write_text(json.dumps({"resolved_tools": ["xdsl_kit", "isa_tools"]}))
    calls = [
        ("isa", "python isa_tools.py asm", "assembled"),
        ("selfcheck", "python agent_selfcheck.py --capsules all", '{"n_capsules": 2, "n_passed": 0}'),
    ]
    events = [{"type": "system", "subtype": "init", "driver": "codex"}]
    for name, command, output in calls:
        events.append(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": name, "name": "bash", "input": {"command": command}}]
                },
            }
        )
        if case == "missing-result" and name == "isa":
            continue
        events.append(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": name,
                            "content": output,
                            # A real full selfcheck may fail numerically and still prove workflow.
                            "is_error": name == "selfcheck" or case == "failed-isa",
                        }
                    ]
                },
            }
        )
    transcript = rounds / "round_03.transcript.jsonl"
    transcript.write_text("".join(json.dumps(event) + "\n" for event in events))
    before = {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    result = subprocess.run(
        [sys.executable, "-m", "merlin_experiments.phase1.conformance", str(run)],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Historical replay CLI reports the verdict, not a process-exit grading status.
    assert result.returncode == 0, result.stderr
    verdict = json.loads(result.stdout)
    assert verdict["conformant"] is (case == "complete")
    assert verdict["checks"]["full_selfcheck"] is True
    assert verdict["checks"]["cca_used"] is None
    assert verdict["resolved_tools"] == ["isa_tools", "xdsl_kit"]
    if case in {"missing-result", "failed-isa"}:
        assert verdict["checks"]["isa_tools_used"] is False
        assert verdict["checks"]["asm_used"] is False
    if case == "missing-result":
        assert verdict["tool_evidence"]["n_missing_results"] == 1
    if case == "regex":
        assert verdict["checks"]["no_regex_ok"] is False
        assert verdict["regex_hits"]
    if case == "invalid-python":
        assert verdict["checks"]["no_regex_ok"] is False
        assert "regex_scan_error" in verdict
    assert ("NOT CONFORMANT" in result.stderr) is (case != "complete")
    assert before == {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}


def test_import_and_help_are_inert_without_native_or_descriptor(tmp_path):
    program = """
import importlib.abc, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in {'_common', 'conformance', 'run_baseline_qa_loop',
                    'merlin_experiments.phase1.context', 'merlin.common.regex_scan'}:
            raise AssertionError('unexpected import: ' + name)
sys.meta_path.insert(0, NoNative())
from merlin_experiments.phase1 import conformance
assert callable(conformance.compute)
conformance.main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Dev-conformance verdict" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_conformance_owner_has_nonvacuous_candidate_mask(tmp_path, monkeypatch):
    from merlin.common import access
    from merlin.targetgen.sandbox import bwrap

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    private = tmp_path / "packages/merlin-experiments/src/merlin_experiments/phase1/conformance.py"
    private.parent.mkdir(parents=True)
    private.write_text("# private treatment grading\n")
    context = private.with_name("context.py")
    context.write_text("# inert context\n")
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
    assert "merlin_experiments.phase1.conformance" in access.declared_modules("grader")
    surfaces = surfaces_module.answer_surfaces(policy)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in bwrap.coverage_gap(unmasked, surfaces)} == {private}
    assert all(surface.path != context and surface.path not in context.parents for surface in surfaces)
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize("changed_owner", ["conformance", "scanner"])
def test_source_receipt_binds_conformance_and_actual_scanner(tmp_path, monkeypatch, changed_owner):
    from merlin_experiments.phase1 import source_inputs
    from merlin_experiments.spec import SpecError

    package = tmp_path / "phase1"
    shutil.copytree(
        module_source_path("merlin_experiments.phase1").parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    scanner = tmp_path / "regex_scan.py"
    shutil.copyfile(module_source_path("merlin.common.regex_scan"), scanner)
    original = source_inputs._source

    def copied_source(module):
        if module == "merlin.common.regex_scan":
            return scanner
        if module == "merlin_experiments.phase1":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.phase1."):
            path = package.joinpath(*module.split(".")[2:])
            return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        return original(module)

    monkeypatch.setattr(source_inputs, "_source", copied_source)
    inputs = {"repo": tmp_path, "entrypoint": tmp_path / "synthetic_transport.py"}
    record = source_inputs.record(**inputs)
    assert record["inputs"]["phase1:source:conformance.py"]["path"] == str(package / "conformance.py")
    assert record["inputs"]["phase1:startup:python_regex_scanner"]["path"] == str(scanner)
    source_inputs.verify(record, **inputs)
    member = package / "conformance.py" if changed_owner == "conformance" else scanner
    member.write_text(member.read_text() + "# changed grading implementation\n")
    with pytest.raises(SpecError, match="source identity changed"):
        source_inputs.verify(record, **inputs)
