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


def test_terminal_evidence_dependency_preserves_failed_verdict(tmp_path):
    marker = tmp_path / "finished"
    first = row("failed-check", argv=[sys.executable, "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('done'); raise SystemExit(1)"])
    gather = row("gather", argv=[sys.executable, "-c",
        f"from pathlib import Path; assert Path({str(marker)!r}).read_text() == 'done'"])
    gather["after_finished"] = ["failed-check"]
    result = run(tmp_path, plan(first, row("blocked", after=["failed-check"]), gather))
    assert result["actions"] == {"failed-check": "failed", "blocked": "deferred", "gather": "completed"}
    assert result["qualification_authority"] is False
    assert result["capsule_passes"] is None


def test_terminal_dependency_cannot_reference_future_action():
    first = row("first")
    first["after_finished"] = ["future"]
    with pytest.raises(ValueError, match="earlier actions"):
        A.validate(plan(first, row("future")))


def test_action_bound_covers_construction_and_individual_corpus_checks():
    A.validate(plan(*(row(f"action-{n}") for n in range(300))))
    with pytest.raises(ValueError, match="512"):
        A.validate(plan(*(row(f"action-{n}") for n in range(513))))


@pytest.mark.parametrize("numerical_pass", [True, False])
def test_partial_tier_report_distinguishes_numerics_without_changing_gates(tmp_path, monkeypatch, numerical_pass):
    monkeypatch.chdir(tmp_path)
    # Real CLI completion with the native partial-check return code. These are
    # synthetic report projections, never numerical qualification evidence.
    response = {
        "selfcheck_request_id": "partial",
        "submission_sha256": "snapshot",
        "all_pass": False,
        "certified_complete": False,
        "n_capsules": 1,
        "per_capsule": [{
            "capsule": "tiny",
            "numerical_execution_passed": numerical_pass,
            "numeric_consistency_passed": numerical_pass,
            "structural_qualification": {"status": "pass", "violations": []},
        }],
    }
    (tmp_path / "agent_selfcheck.py").write_text(
        "import json\nprint(json.dumps(" + repr(response) + "))\nraise SystemExit(1)\n"
    )
    result = run(tmp_path, plan(
        row("screen", kind="selfcheck", argv=["--capsules", "tiny", "--sim", "spike"]),
        row("dependent", after=["screen"]),
        row("independent"),
    ))
    assert result["actions"] == {"screen": "failed", "dependent": "deferred", "independent": "completed"}
    report = result["selfcheck_reports"]["screen"]
    assert report["completed_report"] is True
    assert report["requested_numerical_screen"] == ("pass" if numerical_pass else "fail")
    assert report["structural_qualification"] == "pass"
    assert report["full_certification"] is report["grader_all_pass"] is False
    assert result["selfcheck_summary"]["completed_reports"] == 1
    assert result["selfcheck_summary"]["requested_numerical_screen_pass"] == int(numerical_pass)
    assert result["selfcheck_summary"]["requested_numerical_screen_fail"] == int(not numerical_pass)
    assert result["selfcheck_summary"]["full_certification_pass"] == 0
    assert result["capsule_passes"] is None
    events = [json.loads(line) for line in (tmp_path / "output/events.jsonl").read_text().splitlines()]
    final = next(event for event in events if event["action_id"] == "screen" and "report" in event)
    assert final["report"] == report and final["feedback"]["all_pass"] is False


def test_scalability_screen_never_infers_structural_or_full_certification():
    response = {
        "selfcheck_request_id": "connected",
        "all_pass": False,
        "complete": False,
        "requested_cases": ["region"],
        "cases": [{"capsule": "region", "status": "pass", "tiers": {"L0": "pass", "L1": "pass", "L2": "pass"}}],
    }
    report = A.report_projection(response)
    assert report["completed_report"] is True
    assert report["requested_numerical_screen"] == "pass"
    assert report["structural_qualification"] == "unknown"
    assert report["full_certification"] is None
    response["cases"][0]["tiers"].pop("L2")
    assert A.report_projection(response)["requested_numerical_screen"] == "unknown"
    response["cases"][0]["tiers"]["L2"] = "pass"
    response["requested_cases"].append("missing")
    assert A.report_projection(response)["requested_numerical_screen"] == "unknown"


def test_structurally_unqualified_numerical_screen_is_not_full_success():
    response = {
        "selfcheck_request_id": "group",
        "n_capsules": 1,
        "all_pass": False,
        "certified_complete": False,
        "per_capsule": [{
            "capsule": "group",
            "numerical_execution_passed": True,
            "structural_qualification": {"status": "unavailable_fields", "unavailable_requirements": ["reuse"]},
        }],
    }
    report = A.report_projection(response)
    assert report["requested_numerical_screen"] == "pass"
    assert report["structural_qualification"] == "unknown"
    assert report["cases"][0]["structural_qualification"]["unavailable_requirements"] == ["reuse"]
    assert report["full_certification"] is False
    response["n_capsules"] = 2
    assert A.report_projection(response)["requested_numerical_screen"] == "unknown"
    assert A.report_projection({"selfcheck_request_id": "r", "all_pass": True})["requested_numerical_screen"] == "unknown"


@pytest.mark.parametrize("status", ["fail", "failed"])
def test_structural_failure_preserves_numerical_result(status):
    response = {
        "selfcheck_request_id": "structural-failure",
        "n_capsules": 1,
        "per_capsule": [{
            "capsule": "convolution",
            "numerical_execution_passed": True,
            "structural_qualification": {"status": status, "violations": ["missing accumulation"]},
        }],
    }
    report = A.report_projection(response)
    assert report["requested_numerical_screen"] == "pass"
    assert report["structural_qualification"] == "fail"
    assert report["cases"][0]["structural_qualification"] == response["per_capsule"][0]["structural_qualification"]
    assert report["full_certification"] is None


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


def test_scripted_joint_selfcheck_starts_native_clock_before_submission(tmp_path, monkeypatch):
    """Real public-client IPC/native routing; synthetic evaluator, no simulator claim."""
    import threading
    import time
    from types import SimpleNamespace

    from merlin_experiments.phase1 import joint_control as JC
    from merlin_experiments.phase1.feedback import lifecycle as FL
    from merlin_experiments.phase1.providers import execution as EX
    from merlin_experiments.phase1.tools import selfcheck

    from merlin.targetgen.sandbox import bwrap as BW

    state = tmp_path / "joint_campaign.json"
    state.write_text(json.dumps({"started_at_epoch": None, "policy": {"duration_s": 3600, "drain_s": 1}}))
    monkeypatch.setenv(JC.ENV, str(state))
    with pytest.raises(RuntimeError, match="clock has not started"):
        JC.remaining()
    source = tmp_path / "actions.json"
    source.write_text(
        json.dumps(
            plan(
                row(
                    "check",
                    kind="selfcheck",
                    argv=["--capsules", "synthetic-public", "--sim", "spike", "--timeout", "5"],
                )
            )
        )
    )
    options = parse_options(
        ["--run-id", "joint-study", "--offline-actions", str(source), "--fresh-backend", "--max-wall-s", "10"],
        environ={},
    )
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()
    (ws / "submission").mkdir()
    (ws / "agent_selfcheck.py").write_bytes(Path(selfcheck.__file__).read_bytes())
    S.stage(ws, run_dir, options)
    prepared = SimpleNamespace(
        request=SimpleNamespace(options=options, context=SimpleNamespace(repo=tmp_path), resolved_tools=lambda: ()),
        workspace=ws,
        run_dir=run_dir,
        bundle={},
        public_root=None,
        policy_root=None,
        contract_root=None,
        environment={"run_config": {"construction_study": S.selection(options)}},
        verify_inputs=lambda: None,
    )
    monkeypatch.setattr(BW, "frozen_selected_rtl_facts", lambda *a, **k: None)
    monkeypatch.setattr(EX, "sandbox_command", lambda inner, *a, **k: inner)
    observed = []

    def start_brokers(workspace, config):
        channel = workspace / ".qa_channel"
        channel.mkdir()

        def serve():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                requests = list(channel.glob("req_*.json"))
                if requests:
                    request = json.loads(requests[0].read_text())
                    response = {
                        "selfcheck_protocol": 3,
                        "selfcheck_request_id": request["request_id"],
                        "selfcheck_submission": request["submission"],
                        "submission_sha256": request["submission_sha256"],
                    }
                    try:
                        remaining = JC.remaining()
                        forwarded = []
                        routed = JC.selfcheck_policy(lambda argv: forwarded.extend(argv) or 0)(
                            ["--capsules", request["capsules"], "--sim", request["sim"], "--timeout", "5"]
                        )
                        observed.append((remaining, routed, forwarded))
                        response["all_pass"] = routed == 0
                    except RuntimeError as error:
                        response.update(all_pass=False, error=str(error))
                    (channel / f"resp_{request['request_id']}.json").write_text(json.dumps(response))
                    (channel / f"done_{request['request_id']}").touch()
                    return
                time.sleep(0.01)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        return worker

    monkeypatch.setattr(FL, "start_brokers", start_brokers)
    monkeypatch.setattr(FL, "stop_brokers", lambda ws, worker, **kw: worker.join(timeout=2))
    assert S.execute(prepared, timing_file=tmp_path / "timing") == 0
    assert len(observed) == 1 and 0 < observed[0][0] <= 3600
    assert observed[0][1] == 0 and "synthetic-public" in observed[0][2]
    assert json.loads(state.read_text())["started_at_epoch"] is not None


def test_bounded_admission_keeps_construction_and_collection_rolling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'pending').mkdir()
    # A real subprocess protocol stub checks admission and fairness only.
    # The slow request cannot finish before a later request has been collected.
    (tmp_path / 'agent_selfcheck.py').write_text('''import json,sys
from pathlib import Path
pending=Path('pending')
if '--attach' not in sys.argv:
    name=sys.argv[sys.argv.index('--capsules')+1]
    assert len(list(pending.iterdir())) < 2, 'client window overflow'
    (pending/name).touch()
    print(json.dumps({'request_id':name,'status':'in_progress'}))
    raise SystemExit(3)
name=sys.argv[sys.argv.index('--attach')+1]
assert Path('constructed').exists(), 'pending checks blocked construction'
if name=='slow' and not Path('fast-collected').exists():
    print(json.dumps({'request_id':name,'status':'in_progress'}))
    raise SystemExit(3)
if name!='slow': Path('fast-collected').touch()
(pending/name).unlink()
print(json.dumps({'selfcheck_request_id':name,'all_pass':True}))
''')
    value = plan(
        row('slow', kind='selfcheck', argv=['--capsules', 'slow']),
        row('fast', kind='selfcheck', argv=['--capsules', 'fast']),
        row('third', kind='selfcheck', argv=['--capsules', 'third']),
        row('construct', argv=[sys.executable, '-c', "from pathlib import Path; Path('constructed').touch()"]),
    )
    result = A.execute(value, output=tmp_path/'output', timeout_s=10, plan_sha256='bounded',
                       command_workers=1, selfcheck_window=2)
    assert set(result['actions'].values()) == {'completed'}
    assert result['selfcheck_window'] == 2
    assert not list((tmp_path/'pending').iterdir())
    events = [json.loads(line) for line in (tmp_path/'output/events.jsonl').read_text().splitlines()]
    finished = [row['action_id'] for row in events if row['state']=='completed']
    assert finished.index('construct') < finished.index('fast') < finished.index('slow')
