"""Execution parity and local readiness only; these tests do not qualify a compiler."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1 import construction_actions as A
from merlin_experiments.phase1 import construction_study as S
from merlin_experiments.phase1.options import parse_options


def row(name, *, kind="command", after=(), argv=None, capability="projection"):
    return {
        "id": name,
        "kind": kind,
        "boundary": "isa_lowering",
        "capability": capability,
        "argv": argv or [sys.executable, "-c", 'print("constructed")'],
        "after": list(after),
    }


def plan(*rows):
    return {"schema": A.SCHEMA, "actions": list(rows)}


def run(tmp_path, value):
    return A.execute(value, output=tmp_path / "output", timeout_s=10, plan_sha256="test")


def test_failed_capability_does_not_block_independent_construction(tmp_path):
    result = run(
        tmp_path,
        plan(
            row("broken", argv=[sys.executable, "-c", "raise SystemExit(1)"]),
            row("related-test", after=["broken"], kind="selfcheck", argv=["--capsules", "x"]),
            row("missing-tool", argv=[str(tmp_path / "absent-tool")]),
            row("other-capability", capability="convolution"),
        ),
    )
    assert result["actions"] == {
        "broken": "failed",
        "related-test": "deferred",
        "missing-tool": "failed",
        "other-capability": "completed",
    }
    assert result["capsule_passes"] is None
    assert result["qualification_authority"] is False


def test_pending_test_does_not_pause_construction_and_replays_exact_argv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Real subprocess client shim at the same CLI boundary, no pretend numerical evidence.
    (tmp_path / "agent_selfcheck.py").write_text("""import json, sys
from pathlib import Path
if '--attach' not in sys.argv:
    print(json.dumps({'status': 'in_progress', 'request_id': 'r1'}))
    raise SystemExit(3)
assert Path('other-built').is_file(), 'construction waited for unrelated test'
print(json.dumps({'selfcheck_request_id':'r1', 'submission_sha256':'exact-source', 'all_pass': True}))
""")
    value = plan(
        row("test", kind="selfcheck", argv=["--capsules", "small", "--sim", "spike"]),
        row(
            "build",
            capability="other",
            argv=[sys.executable, "-c", "from pathlib import Path; Path('other-built').write_text('done')"],
        ),
        row("next", after=["test"]),
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(value))
    served = tmp_path / "construction_actions.py"
    served.write_bytes(Path(A.__file__).read_bytes())
    replay = subprocess.run(
        [sys.executable, str(served), str(plan_path), "--out", "replayed", "--timeout", "10"],
        capture_output=True,
        text=True,
    )
    assert replay.returncode == 0, replay.stderr
    result = json.loads((tmp_path / "replayed/result.json").read_text())
    assert result["actions"] == {"test": "completed", "build": "completed", "next": "completed"}
    events = [json.loads(line) for line in (tmp_path / "replayed/events.jsonl").read_text().splitlines()]
    checks = [event for event in events if event["action_id"] == "test"]
    assert [event["state"] for event in checks] == ["running", "pending", "completed"]
    assert checks[0]["argv_sha256"] == checks[-1]["argv_sha256"]
    assert checks[-1]["feedback"]["submission_sha256"] == "exact-source"
    # Automatic author imports/executes this exact copied client and command constructor.
    assert served.read_bytes() == Path(A.__file__).read_bytes()
    assert A.command(value["actions"][0]) == [
        sys.executable,
        "agent_selfcheck.py",
        "--capsules",
        "small",
        "--sim",
        "spike",
        "--wait-budget",
        "1",
    ]


@pytest.mark.parametrize(
    "bad", [plan(row("a", after=["missing"])), plan(row("a"), row("a")), plan(row("a", kind="grant-private-access"))]
)
def test_malformed_plan_is_not_executable(bad):
    with pytest.raises(ValueError):
        A.validate(bad)


def test_explicit_study_selection_and_staging(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan(row("build"))))
    options = parse_options(
        ["--run-id", "study", "--offline-actions", str(path), "--fresh-backend", "--max-wall-s", "10"], environ={}
    )
    S.validate(options)
    for change in (
        {"resume": True},
        {"seed_submission": "retained"},
        {"fresh_backend": False},
        {"sandbox": "none"},
        {"preflight_only": True},
    ):
        with pytest.raises(ValueError):
            S.validate(dataclasses.replace(options, **change))
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()
    S.stage(ws, run_dir, options)
    assert (ws / S.CLIENT).read_bytes() == Path(A.__file__).read_bytes()
    assert (ws / S.PLAN).read_bytes() == path.read_bytes()
    assert S.selection(options)["provider_started"] is False
    S.stage(ws, run_dir, options)  # idempotent exact client; read-only source remains preserved


def test_controller_dispatch_preserves_ordinary_prepared_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin_experiments.phase1 import controller

    token = object()
    timing = tmp_path / "timing"
    seen = []
    monkeypatch.setattr(S, "execute", lambda prepared, *, timing_file: seen.append((prepared, timing_file)) or 0)
    options = SimpleNamespace(preflight_only=False, codex_canary=False, offline_actions="public-plan")
    assert controller._execute_prepared(token, options, None, "selected", timing, None) == 0
    assert seen == [(token, timing)]


def test_native_study_uses_existing_sandbox_and_brokers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin_experiments.phase1.feedback import lifecycle as FL
    from merlin_experiments.phase1.providers import execution as EX

    from merlin.targetgen.sandbox import bwrap as BW

    source = tmp_path / "actions.json"
    source.write_text(json.dumps(plan(row("construction"))))
    options = parse_options(
        ["--run-id", "offline", "--offline-actions", str(source), "--fresh-backend", "--max-wall-s", "10"], environ={}
    )
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()
    S.stage(ws, run_dir, options)
    context = SimpleNamespace(repo=tmp_path)
    calls = []
    prepared = SimpleNamespace(
        request=SimpleNamespace(options=options, context=context, resolved_tools=lambda: ("selected",)),
        workspace=ws,
        run_dir=run_dir,
        bundle={"compiler_tool_profile": {}},
        environment={"run_config": {"construction_study": S.selection(options)}},
        public_root=tmp_path / "public",
        policy_root=tmp_path / "policy",
        contract_root=tmp_path / "contract",
        verify_inputs=lambda: calls.append("verify"),
    )
    monkeypatch.setattr(BW, "frozen_selected_rtl_facts", lambda *a, **k: tmp_path / "facts")
    monkeypatch.setattr(
        FL, "start_brokers", lambda workspace, config: calls.append(("brokers", workspace, config)) or "owned"
    )
    monkeypatch.setattr(FL, "stop_brokers", lambda workspace, broker, **kw: calls.append(("stop", broker)))

    def sandbox(inner, workspace, bundle, **kwargs):
        calls.append(("sandbox", inner, workspace, bundle, kwargs))
        # Only the composition is tested here; this harmless command is not isolation evidence.
        return inner

    monkeypatch.setattr(EX, "sandbox_command", sandbox)
    assert S.execute(prepared, timing_file=tmp_path / "timing") == 0
    assert calls[0] == calls[-2] == "verify"
    assert calls[1][2].compiler_bundle_manifest == run_dir / "input_bundle_manifest.yaml"
    assert calls[1][2].capsules_root == tmp_path / "public"
    assert calls[2][0] == "sandbox"
    assert calls[-1] == ("stop", "owned")
    result = json.loads((run_dir / "construction_study_result.json").read_text())
    assert result["status"] == "actions_completed"
    assert result["provider_started"] is False
    assert result["qualification_authority"] is False


def test_long_command_does_not_block_independent_construction(tmp_path):
    signal_path = tmp_path / "second-built"
    wait_script = (
        "from pathlib import Path; import time; p=Path(" + repr(str(signal_path)) + "); "
        "deadline=time.monotonic()+2\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "assert p.exists(), 'independent work blocked behind long build'"
    )
    result = run(
        tmp_path,
        plan(
            row("long", argv=[sys.executable, "-c", wait_script]),
            row(
                "independent",
                argv=[sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(signal_path)) + ").touch()"],
            ),
        ),
    )
    assert set(result["actions"].values()) == {"completed"}
