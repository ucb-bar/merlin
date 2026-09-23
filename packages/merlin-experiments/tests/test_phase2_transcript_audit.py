"""Installed transcript policy audits observations without loading a native controller."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import transcript_audit as audit


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text("entrypoints:\n  tool: target-opt\n")
    (candidate / "target-opt").write_text("candidate bytes; never executed")
    monkeypatch.setattr(
        audit,
        "audit_tokens",
        lambda target: {
            "answer": ["golden.yaml"],
            "grader": [],
            "oracle_subpath": [],
        },
    )
    monkeypatch.setattr(
        audit.TC,
        "required_tool_probes",
        lambda target: [
            SimpleNamespace(cmd="fixture-opt --version", label="fixture-opt"),
        ],
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("audit launched a process"))
    return SimpleNamespace(target="synthetic"), candidate, (SimpleNamespace(name="candidate-parse"),)


def transcript(path, command, native):
    if native:
        item = {"id": "command-1", "type": "command_execution", "command": command}
        rows = [
            {"type": "thread.started", "thread_id": "synthetic-thread"},
            {"type": "item.started", "item": {**item, "status": "in_progress"}},
            {"type": "item.completed", "item": {**item, "status": "completed", "exit_code": 0}},
        ]
    else:
        rows = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Bash", "input": {"command": command}},
                    ]
                },
            }
        ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "command,hit",
    [
        ("python3 submission/target-opt input.mlir", "candidate_execution_outside_broker"),
        ("fixture-opt input.mlir", "target_tool_outside_broker"),
        ("cat hidden/golden.yaml", "answer_reconnaissance"),
        ("python3 /perf-control/perf_tool.py candidate-parse input_mlir=input.mlir", None),
    ],
)
def test_native_and_translated_observations_enforce_same_policy(inputs, tmp_path, native, command, hit):
    report = audit.audit_codex_transcript(transcript(tmp_path / "events.jsonl", command, native), *inputs)
    assert report["commands_seen"] == 1
    assert report["clean"] is (hit is None)
    if hit is None:
        assert len(report["broker_invocations"]) == 1
    else:
        assert hit in [record["kind"] for record in report["hits"]]


@pytest.mark.parametrize(
    "tokens",
    [
        {},
        {"answer": [], "grader": []},
        {"answer": "golden.yaml", "grader": [], "oracle_subpath": []},
        {"answer": [""], "grader": [], "oracle_subpath": []},
        {"answer": [4], "grader": [], "oracle_subpath": []},
    ],
)
def test_malformed_explicit_token_sets_refused(inputs, tmp_path, tokens):
    path = transcript(tmp_path / "events.jsonl", "ls submission", False)
    with pytest.raises(audit.StageGateError, match="token set is malformed"):
        audit.audit_codex_transcript(path, *inputs, audit_token_set=tokens)


def test_heredoc_bodies_are_source_not_additional_commands(inputs, tmp_path):
    command = "python3 - <<'PY'\nexec(open('submission/target-opt').read())\nPY"
    report = audit.audit_codex_transcript(transcript(tmp_path / "events.jsonl", command, False), *inputs)
    assert report["commands_seen"] == 1
    assert not report["clean"]
    assert "candidate_execution_outside_broker" in [record["kind"] for record in report["hits"]]


@pytest.mark.parametrize(
    "label,command,blocked",
    [
        ("fixture-opt: version probe", "fixture-opt input.mlir", True),
        ("FIXTURE-OPT (version)", "fixture-opt input.mlir", True),
        ("fixture.opt+2_extra --version", "fixture.opt+2_extra input.mlir", True),
        ("fixture-opté", "fixture-opt input.mlir", True),
        (" fixture-opt", "fixture-opt input.mlir", False),
        ("éfixture-opt", "fixture-opt input.mlir", False),
        ("", "fixture-opt input.mlir", False),
        ("python3: interpreter", "python3 -c 'print(1)'", False),
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_probe_label_ascii_prefix_policy(inputs, tmp_path, monkeypatch, label, command, blocked, native):
    monkeypatch.setattr(
        audit.TC,
        "required_tool_probes",
        lambda target: [
            SimpleNamespace(cmd="python3 --version", label=label),
        ],
    )
    report = audit.audit_codex_transcript(transcript(tmp_path / "events.jsonl", command, native), *inputs)
    hits = [record["kind"] for record in report["hits"]]
    assert ("target_tool_outside_broker" in hits) is blocked
    assert report["commands_seen"] == 1


def test_owner_import_is_independent_of_native_checkout(tmp_path):
    from merlin.common.paths import python_import_roots

    program = """import importlib.abc, pathlib, subprocess, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"perf_agent_stage", "_common", "run_baseline_qa_loop"}:
            raise AssertionError("native controller import: " + fullname)
sys.meta_path.insert(0, NoNative())
def forbidden(*args, **kwargs):
    raise AssertionError("import launched a process")
subprocess.Popen = forbidden
from merlin_experiments.phase2 import transcript_audit
assert callable(transcript_audit.audit_codex_transcript)
assert not any("gemmini_perf_bench" in str(getattr(module, "__file__", "")) for module in sys.modules.values())
"""
    env = {
        **os.environ,
        "MERLIN_REPO_ROOT": str(tmp_path),
        "PYTHONPATH": os.pathsep.join(map(str, python_import_roots())),
    }
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
