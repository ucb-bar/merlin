"""Execute actual controller selection and final-conformance nodes over real evidence."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import recovery as ROQ

from merlin.common.paths import module_source_path


def _transcript(path, *, dead=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if dead:
        events = [{"type": "system", "subtype": "init"}, {"type": "result", "is_error": True}]
    else:
        events = [
            {
                "type": "assistant",
                "message": {
                    "model": "model",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "id": "call",
                            "input": {"command": "echo authoring"},
                        }
                    ],
                },
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "call", "content": "authoring"}]},
            },
            {"type": "result", "is_error": False},
        ]
    path.write_text("\n".join(map(json.dumps, events)))
    return path


@pytest.fixture
def controller(tmp_path):
    tree = ast.parse((module_source_path("merlin_experiments.phase1.authoring")).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    body = functions["execute"].body

    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

    start = next(i for i, node in enumerate(body) if assigned(node, "_saved_authoring_transcripts"))
    selections = [node for node in ast.walk(functions["execute"]) if assigned(node, "_latest_authoring_tpath")]
    ordinary = next(
        node for node in selections if "tpath]" in ast.unparse(node) and "_fix_tpath" not in ast.unparse(node)
    )
    fix = next(node for node in selections if "_fix_tpath" in ast.unparse(node))
    final = next(
        node
        for node in body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_latest_authoring_tpath is not None"
    )
    seal = next(node for node in body if isinstance(node, ast.If) and ast.unparse(node.test) == "a.seal_current")
    seal_end = next(i for i, node in enumerate(seal.body) if assigned(node, "_latest_authoring_tpath"))
    seal_prefix = ast.If(test=seal.test, body=seal.body[: seal_end + 1], orelse=[])
    scope = {
        "Path": Path,
        "Mapping": Mapping,
        "ROQ": ROQ,
        "run_dir": tmp_path,
        "ws": tmp_path / "workspace",
        "arm": "baseline",
        "_endpoint_kind": "",
        "resolved_tools": lambda: (),
        "a": SimpleNamespace(seal_current=False),
    }

    def execute(nodes):
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "actual-controller-evidence", "exec"
            ),
            scope,
        )

    execute([functions["_workflow_conformance"]])
    return (
        scope,
        lambda: execute(body[start : start + 2]),
        lambda: execute([ordinary]),
        lambda: execute([fix]),
        lambda: execute([final]),
        lambda: execute([seal_prefix]),
    )


@pytest.mark.parametrize("mode", ["fresh", "resume", "same_index_retry", "l3_fix"])
@pytest.mark.parametrize("failure", ["dead", "daily", "five_hour", "weekly"])
def test_actual_completion_keeps_live_evidence_after_dead_launch(tmp_path, controller, mode, failure):
    scope, initialize, ordinary, fix, finish, _ = controller
    if mode == "fresh":
        initialize()
    live = _transcript(tmp_path / "rounds/round_02.transcript.jsonl")
    latest = tmp_path / "rounds/round_03.transcript.jsonl"
    if mode == "same_index_retry":
        _transcript(latest)
        initialize()
        assert scope["_latest_authoring_tpath"] == latest
    _transcript(latest, dead=True)
    if failure != "dead":
        event = {"type": "result", "result": "429 daily quota limit"}
        if failure != "daily":
            event = {
                "type": "rate_limit_event",
                "rate_limit_info": {
                    "status": "rejected",
                    "rateLimitType": "seven_day" if failure == "weekly" else failure,
                },
            }
        latest.write_text(
            "\n".join(
                map(
                    json.dumps,
                    [
                        {"type": "assistant", "message": {"model": "model", "content": []}},
                        event,
                    ],
                )
            )
        )
    # Finalize is explicitly not authoring evidence, even if it contains tool work.
    _transcript(tmp_path / "rounds/finalize.transcript.jsonl")
    if mode == "resume":
        initialize()
    else:
        scope.update(tpath=latest, _fix_tpath=latest)
        (fix if mode == "l3_fix" else ordinary)()
    finish()
    assert scope["_latest_authoring_tpath"] == live
    assert scope["final_conformance"]["tool_evidence"]["n_calls"] == 1


def test_actual_completion_fails_closed_when_every_launch_is_dead(tmp_path, controller):
    scope, initialize, _, _, finish, _ = controller
    _transcript(tmp_path / "rounds/round_02.transcript.jsonl", dead=True)
    initialize()
    finish()
    assert scope["_latest_authoring_tpath"] is None
    assert scope["workflow_conformant"] is False
    assert scope["final_conformance"] == {"conformant": False, "error": "no authoring transcript"}


@pytest.mark.parametrize("malformed_tail", [False, True])
def test_operator_seal_uses_checkpoint_authority_not_live_tail(tmp_path, controller, malformed_tail):
    scope, initialize, _, _, finish, seal = controller
    completed = _transcript(tmp_path / "rounds/round_02.transcript.jsonl", dead=True)
    tail = _transcript(tmp_path / "rounds/round_03.transcript.jsonl")
    if malformed_tail:
        tail.write_text("{unaudited partial tail")
    submission = scope["ws"] / "submission"
    submission.mkdir(parents=True)
    (submission / "manifest.yaml").write_text("{}\n")
    scope.update(a=SimpleNamespace(seal_current=True), rounds_summary=[{"round": 2}])
    initialize()
    assert scope["_latest_authoring_tpath"] is None
    seal()
    finish()
    assert scope["_latest_authoring_tpath"] == completed
    assert scope["final_conformance"]["tool_evidence"]["n_calls"] == 0


def test_productive_late_failure_replaces_older_authoring_evidence(tmp_path, controller):
    scope, initialize, ordinary, _, finish, _ = controller
    _transcript(tmp_path / "rounds/round_02.transcript.jsonl")
    initialize()
    latest = _transcript(tmp_path / "rounds/round_03.transcript.jsonl")
    latest.write_text(
        latest.read_text()
        + "\n"
        + json.dumps(
            {
                "type": "result",
                "is_error": True,
                "result": "authentication failed after tool work",
            }
        )
    )
    scope["tpath"] = latest
    ordinary()
    finish()
    assert scope["_latest_authoring_tpath"] == latest
    assert scope["final_conformance"]["tool_evidence"]["n_calls"] == 1
