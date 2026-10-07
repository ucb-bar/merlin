"""One authoring round of a measured run, driven by a dummy agent through the real broker, shim,
transcript audit and edit authority, with a fake machine: the agent's request reaches the screen, the
receipts join the audited transcript, an authored round's bytes become the run's candidate and are
committed, and a round that is not authored keeps the previous candidate while the run goes on."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2 import transcript_audit as audit
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import ledger as L
from merlin_experiments.phase2.whole_model_measured import objective as O
from merlin_experiments.phase2.whole_model_measured import rounds as R
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured import sessions as SES
from merlin_experiments.phase2.whole_model_measured import worker as W
from merlin_experiments.phase2.whole_model_measured.identity import package_digest, read_json

from merlin.common import oot_repo
from merlin.perf import whole_model_verdict as V

TOKENS = {"answer": ["golden.yaml"], "grader": [], "oracle_subpath": []}
TOOL = "python3 /perf-control/perf_tool.py"


@pytest.fixture
def run(tmp_path, monkeypatch):
    dispatched: list[Path] = []

    def spawn(argv, **kw):
        dispatched.append(Path(argv[-1]))
        return SimpleNamespace(pid=999_999_999)

    monkeypatch.setattr(S, "spawn", spawn)
    monkeypatch.setattr(S, "alive", lambda pid, owner: pid == 999_999_999)
    monkeypatch.setattr(audit.TC, "required_tool_probes", lambda target: [])
    run_dir = tmp_path / "run"
    seed = FX.package(tmp_path / "seed_src", "pkg")
    for where in (run_dir / "seed" / "submission", run_dir / "workspace"):
        where.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-r", str(seed), str(where)], check=True)
    phase1 = tmp_path / "phase1_oot"
    oot_repo.init(phase1)
    frozen = oot_repo.commit_candidate(phase1, seed, label="round 1", when="20260929T000000Z")
    oot_repo.tag(phase1, oot_repo.FROZEN_TAG, frozen.commit)
    repo = oot_repo.init_from(run_dir / "oot", phase1, ref=oot_repo.FROZEN_TAG)
    spec, pin = FX.write_builder(tmp_path)
    screen = S.MeasurementService(
        tmp_path / "store", target="toy", builder=spec, builder_sha256=pin, machine=FX.spike_machine(tmp_path)
    )
    objective = O.WholeModelObjective(screen=screen, screen_reference=None, repeats_on_best=1)
    stamps = iter(f"2026093{d}T0{h}0000Z" for d in range(10) for h in range(10))
    objective.ledger = L.OotLedger(
        repo,
        run_id="run",
        records=run_dir / L.ITERATIONS,
        sandbox_roots=(run_dir / "workspace", run_dir / "stage" / "agent_workspaces"),
        clock=lambda: next(stamps),
    )
    return SimpleNamespace(run_dir=run_dir, objective=objective, repo=repo, dispatched=dispatched, tmp=tmp_path)


def _transcript(path: Path, commands: list[str]) -> Path:
    rows = [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": c}}]}}
        for c in commands
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def dummy_agent(*, edit=None, commands=(("measure",),), extra=(), rc=0, outputs=None):
    """An agent that edits the candidate, calls the broker through its shim exactly as the transcript
    says it did, and reports ``rc``."""

    def agent(*, workspace, candidate, control_dir, prompt, round_index, timeout_s, stage_root):
        assert "submission/" in prompt and "lower" not in prompt.lower()
        if edit is not None:
            edit(Path(candidate))
        said = []
        for words in commands:
            proc = subprocess.run(
                [sys.executable, str(Path(control_dir) / "perf_tool.py"), *words],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if outputs is not None:
                outputs.append(proc)
            said.append(" ".join([TOOL, *words]))
        transcript = Path(stage_root) / "rounds" / f"round_{round_index:02d}.transcript.jsonl"
        return rc, _transcript(transcript, [*said, *extra])

    return agent


def faster(candidate: Path) -> None:
    spec = json.loads((candidate / "program.json").read_text())
    spec["groups"]["2"]["cycles"] = 150
    (candidate / "program.json").write_text(json.dumps(spec, sort_keys=True))


def _driver(run, agent) -> R.RoundDriver:
    profile = {"round_seconds": 120, "iteration_seconds": 60, "max_tool_calls": 20, "max_sessions": 3}
    driver = R.round_driver(
        profile=profile,
        run_dir=run.run_dir,
        objective=run.objective,
        agent=agent,
        target_experiment=SimpleNamespace(target="synthetic"),
    )
    driver.audit_token_set = TOKENS
    return driver


def test_a_dummy_agent_round_reaches_the_screen_and_its_bytes_become_the_candidate(run):
    outputs: list = []
    driver = _driver(run, dummy_agent(edit=faster, outputs=outputs))
    outcome = driver(session=1, stage_root=run.run_dir / "stage")
    assert outcome["status"] == "authored", outcome
    assert outputs[0].returncode == 0 and '"package_sha256"' in outputs[0].stdout
    record = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    assert record["receipts"] == {"receipts": 1, "invocations": 1, "joined": True, "reason": None}
    assert record["audit"]["clean"] is True and record["carried_forward"] is True
    # The workspace now holds the round's bytes, which the agent requested and the harness requested again.
    digest = package_digest(run.run_dir / "workspace")
    assert record["requested"] == [digest] == [record["final_package_sha256"]]
    assert json.loads((run.run_dir / "workspace" / "program.json").read_text())["groups"]["2"]["cycles"] == 150
    # The request was screened and dispatched to the fake machine; running it measures the bytes.
    job = read_json(run.objective.screen.root / digest / "job.json")
    assert job["state"] == J.RUNNING and run.dispatched == [run.objective.screen.root / digest]
    result = W.work(run.dispatched[0])
    assert result["timing_status"] == V.TIMING_MEASURED and result["objective_cycles"] == 250
    # The harness committed the requested bytes to the run's OOT history; the credential left with the round.
    rows = [json.loads(line) for line in (run.run_dir / L.ITERATIONS).read_text().splitlines()]
    assert {row["package_sha256"] for row in rows if row["kind"] == "candidate"} == {digest}
    assert not (run.run_dir / "stage" / "control" / "round_00" / ".perf_broker.json").exists()


def test_a_round_with_an_audit_finding_keeps_the_previous_candidate_and_the_run_goes_on(run):
    before = package_digest(run.run_dir / "workspace")
    agents = iter(
        [
            dummy_agent(edit=faster, extra=("cat golden.yaml",)),
            dummy_agent(edit=faster),
        ]
    )
    driver = _driver(run, None)

    def run_round(*, session, stage_root):
        driver.agent = next(agents)
        return driver(session=session, stage_root=stage_root)

    document = SES.run_sessions(
        run.objective,
        run_round=run_round,
        stage_root=run.run_dir / "stage",
        run="run",
        max_sessions=2,
        total_seconds=1e9,
    )
    assert [row["status"] for row in document["sessions"]] == ["refused", "authored"]
    first = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    assert first["audit"]["clean"] is False and first["carried_forward"] is False
    second = json.loads((run.run_dir / "stage" / "rounds" / "round_01.round.json").read_text())
    assert second["carried_forward"] is True and package_digest(run.run_dir / "workspace") != before


def test_bytes_measured_in_a_round_that_was_not_authored_never_become_the_best(run):
    """The refused round's bytes are measured, and FASTER than the seed, yet never the best, never
    tagged best and never a champion; the same bytes earned by a later authored round are."""
    seed = run.objective.measure(run.run_dir / "workspace", label="seed", seed=True)["package_sha256"]
    for job_dir in list(run.dispatched):
        W.worker_main(job_dir)
    driver = _driver(run, dummy_agent(edit=faster, extra=("cat golden.yaml",)))
    assert driver(session=1, stage_root=run.run_dir / "stage")["status"] == "refused"
    record = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    (digest,) = record["requested"]
    assert record["attribution"] == {digest: J.ATTRIBUTION_UNAUTHORED}
    W.worker_main(run.objective.screen.root / digest)
    assert run.objective.screen.result(digest)["objective_cycles"] == 250  # faster than the seed's 300
    run.objective.poll()
    assert run.objective.summary()["best"]["package_sha256"] == seed
    assert run.objective.screen.best()["package_sha256"] == seed
    # The store and the run's history both say who asked, and why the bytes are not attributable.
    assert run.objective.screen.attribution(digest)["state"] == J.ATTRIBUTION_UNAUTHORED
    assert [r["attribution"] for r in run.objective.screen.history() if r["package_sha256"] == digest] == ["unauthored"]
    rows = [json.loads(line) for line in (run.run_dir / L.ITERATIONS).read_text().splitlines()]
    assert {(r["kind"], r.get("state")) for r in rows if r.get("package_sha256") == digest} >= {
        ("attribution", J.ATTRIBUTION_UNAUTHORED)
    }
    run.objective.ledger.sync(run.objective)
    assert oot_repo.tags(run.repo).get(oot_repo.BEST_TAG) != _commit_of(run, digest)
    # Even a forced best tag on those bytes is refused at export.
    oot_repo.tag(run.repo, oot_repo.BEST_TAG, _commit_of(run, digest), move=True)
    with pytest.raises(L.LedgerError, match="unauthored"):
        L.export_best(run.objective, run.objective.ledger, target="toy", package_id="x", roles=())
    # The same bytes, earned by an authored round, are attributable and become the best.
    driver.agent = dummy_agent(edit=faster)
    assert driver(session=2, stage_root=run.run_dir / "stage")["status"] == "authored"
    assert run.objective.screen.attribution(digest)["state"] == J.ATTRIBUTION_AUTHORED
    run.objective.poll()
    assert run.objective.summary()["best"]["package_sha256"] == digest


def test_harness_and_authored_bytes_are_never_downgraded(run):
    screen = run.objective.screen
    seed = screen.request(run.run_dir / "workspace", label="seed")["package_sha256"]
    screen.attribute(seed, {"state": J.ATTRIBUTION_UNAUTHORED, "round": 0})
    assert screen.attributable(seed) and screen.attribution(seed)["history"][-1]["state"] == "unauthored"
    other = run.tmp / "other"
    subprocess.run(["cp", "-r", str(run.run_dir / "workspace"), str(other)], check=True)
    faster(other)
    pending = {"state": J.ATTRIBUTION_PENDING, "round": 1}
    digest = screen.request(other, label="agent", attribution=pending)["package_sha256"]
    assert not screen.attributable(digest)  # pending from the moment it is requested
    screen.attribute(digest, {"state": J.ATTRIBUTION_AUTHORED, "round": 1})
    screen.attribute(digest, {"state": J.ATTRIBUTION_UNAUTHORED, "round": 2})
    assert screen.attribution(digest)["state"] == J.ATTRIBUTION_AUTHORED


def _commit_of(run, digest: str) -> str:
    rows = [json.loads(line) for line in (run.run_dir / L.ITERATIONS).read_text().splitlines()]
    return next(r["commit"]["commit"] for r in rows if r["kind"] == "candidate" and r["package_sha256"] == digest)


def test_bytes_that_change_the_manifest_controls_are_refused_before_any_build(run):
    def entry_point(candidate: Path) -> None:
        (candidate / "manifest.yaml").write_text("name: fake\nentrypoints: {tool: other}\n")

    outputs: list = []
    driver = _driver(run, dummy_agent(edit=entry_point, outputs=outputs))
    outcome = driver(session=1, stage_root=run.run_dir / "stage")
    assert outputs[0].returncode == 125 and "manifest's controls" in outputs[0].stderr
    assert outcome["status"] == "refused" and run.dispatched == []
    assert run.objective.screen.jobs() == []
    record = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    assert record["edits"]["status"] == "refused" and record["carried_forward"] is False


def test_a_receipt_the_transcript_does_not_show_refuses_the_round(run):
    """The agent called the broker twice but its transcript shows one call: the join refuses."""
    driver = _driver(run, dummy_agent(edit=faster, commands=(("status",), ("measure",))))
    original = driver.agent

    def agent(**kw):
        rc, transcript = original(**kw)
        lines = Path(transcript).read_text().splitlines()
        Path(transcript).write_text(lines[1] + "\n")
        return rc, transcript

    driver.agent = agent
    outcome = driver(session=1, stage_root=run.run_dir / "stage")
    record = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    assert outcome["status"] == "refused" and record["receipts"]["joined"] is False


def test_the_round_driver_is_the_cli_default_and_names_an_unknown_edit_authority(run):
    from merlin_experiments.phase2.whole_model_measured.cli import DEFAULT_ROUND_DRIVER
    from merlin_experiments.phase2.whole_model_measured.identity import load_builder

    assert load_builder(DEFAULT_ROUND_DRIVER) is R.round_driver
    with pytest.raises(Exception, match="unknown edit authority"):
        R.round_driver(
            profile={"edit_authority": "anything", "round_seconds": 1, "iteration_seconds": 1},
            run_dir=run.run_dir,
            objective=run.objective,
            agent=lambda **kw: None,
            target_experiment=SimpleNamespace(target="synthetic"),
        )


def test_the_fast_tiers_are_broker_actions_on_the_current_bytes(run):
    outputs: list = []
    driver = _driver(
        run, dummy_agent(edit=faster, commands=(("structure",), ("group-check", "group=2")), outputs=outputs)
    )
    assert driver(session=1, stage_root=run.run_dir / "stage")["status"] == "authored"
    assert all(proc.returncode == 0 for proc in outputs), [proc.stderr for proc in outputs]
    assert '"state": "running"' in outputs[0].stdout and "never the measured objective" in outputs[0].stdout
    tiers = sorted(p.parent.name for p in (run.objective.screen.root / "fast").glob("*/*/spec.json"))
    digest = package_digest(run.run_dir / "workspace")
    assert tiers == [digest, f"{digest}__g2"]  # the exact bytes the round ended on


def test_the_codex_agent_runs_the_real_executable_path_never_a_bare_name(tmp_path, monkeypatch):
    """A bare `codex` resolved inside the sandbox's system PATH exits 127 before a turn starts."""
    from merlin_experiments.phase1.providers import codex_agent as CA
    from merlin_experiments.phase2.contracts import StageGateError

    real = tmp_path / "pkg" / "bin" / "codex"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    link = tmp_path / "bin" / "codex"
    link.parent.mkdir()
    link.symlink_to(real)
    monkeypatch.setenv("PATH", f"{link.parent}:/usr/bin:/bin")
    monkeypatch.delenv("CODEX_BIN", raising=False)
    seen = {}
    monkeypatch.setattr(CA, "run_round", lambda *a, **kw: seen.update(kw) or (0, tmp_path / "t.jsonl"))
    agent = R.codex_agent({"model": "m"}, target_experiment=SimpleNamespace(target="synthetic"))
    agent(
        workspace=tmp_path,
        candidate=tmp_path,
        control_dir=tmp_path,
        prompt="p",
        round_index=0,
        timeout_s=1,
        stage_root=tmp_path,
    )
    assert seen["codex_binary"] == real.resolve()
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(StageGateError, match="Codex executable is absent"):
        R.codex_agent({"model": "m"}, target_experiment=SimpleNamespace(target="synthetic"))


def test_bytes_first_requested_by_a_correctness_check_carry_the_rounds_attribution(run):
    """`correctness` creates the same job `measure` does.  Created WITHOUT the round's attribution, the
    agent's bytes read as the harness's -- attributable -- and a refused round's bytes could be the best."""
    seed = run.objective.measure(run.run_dir / "workspace", label="seed", seed=True)["package_sha256"]
    for job_dir in list(run.dispatched):
        W.worker_main(job_dir)
    agent = dummy_agent(edit=faster, commands=(("correctness",), ("measure",)), extra=("cat golden.yaml",))
    assert _driver(run, agent)(session=1, stage_root=run.run_dir / "stage")["status"] == "refused"
    record = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    digest = record["requested"][0]
    assert record["attribution"] == {digest: J.ATTRIBUTION_UNAUTHORED}
    assert run.objective.screen.attribution(digest)["state"] == J.ATTRIBUTION_UNAUTHORED
    assert not run.objective.screen.attributable(digest)
    W.worker_main(run.objective.screen.root / digest)
    run.objective.poll()
    assert run.objective.screen.best()["package_sha256"] == seed


def test_the_task_states_the_execution_rule_the_audit_enforces():
    """The audit refuses a round that runs package code or chains a broker call; the task says so."""
    objective = SimpleNamespace(rule=lambda: {"prohibited_roles": ["loop_descriptor"]})
    text = R.render_task(objective, round_index=0, rounds=1, round_seconds=60)
    assert "do not execute any file under `submission/`" in text and "stands alone" in text


def test_the_agent_sees_derived_mechanisms_without_target_specific_hints():
    objective = SimpleNamespace(
        rule=lambda: {},
        config={"mechanism_derivation": {"sections": {"screen": {"model_closure": "closed"}}}},
    )
    text = R.render_task(objective, round_index=0, rounds=1, round_seconds=60)
    assert "package-declared whole-model passes and fused regions are available" in text
    assert "Gemmini" not in text


def test_receipts_join_in_the_order_calls_started_not_the_order_they_finished(tmp_path):
    """A blocking `wait-for-result` finishes after the calls the agent made while it waited."""
    rows = [("measure", "a", 0), ("status", "b", 2), ("wait-for-result", "c", 1)]
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        "".join(json.dumps({"index": i, "action": a, "bindings_command_sha256": d}) + "\n" for a, d, i in rows)
    )
    started = [{"action": a, "bindings_sha256": d} for a, d, _ in sorted(rows, key=lambda r: r[2])]
    assert R.join_receipts(receipts, {"broker_invocations": started})["joined"] is True
    missing = tmp_path / "missing.jsonl"
    missing.write_text(
        "".join(json.dumps({"index": i, "action": a, "bindings_command_sha256": d}) + "\n" for a, d, i in rows[:2])
    )
    assert R.join_receipts(missing, {"broker_invocations": started})["joined"] is False


# ------------------------------------------------------------------- a killed agent or driver
def test_a_killed_agent_is_a_refused_round_and_the_next_session_starts_from_the_previous_candidate(run):
    """SIGKILL leaves no evidence of its own: the round is refused (never authored), its requests are
    unauthored, and the run goes on from the bytes it had."""
    before = package_digest(run.run_dir / "workspace")
    agents = iter([dummy_agent(edit=faster, rc=-9), dummy_agent(edit=faster)])
    driver = _driver(run, None)

    def run_round(*, session, stage_root):
        driver.agent = next(agents)
        return driver(session=session, stage_root=stage_root)

    document = SES.run_sessions(
        run.objective,
        run_round=run_round,
        stage_root=run.run_dir / "stage",
        run="run",
        max_sessions=2,
        total_seconds=1e9,
    )
    assert [row["status"] for row in document["sessions"]] == ["refused", "authored"]
    killed = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    assert killed["agent_signal"] == 9 and killed["carried_forward"] is False
    assert set(killed["attribution"].values()) == {J.ATTRIBUTION_UNAUTHORED}
    assert package_digest(run.run_dir / "workspace") != before  # the second, authored round carried
    assert not list((run.run_dir / "stage" / "rounds").glob(f"*{R.OPEN_SUFFIX}"))


def test_an_agent_driver_that_raises_is_recorded_and_its_requests_resolved(run):
    """A driver that dies (no transcript at all) is the ROUND's failure, recorded with its requests
    attributed -- never an exception that leaves them pending forever."""

    def agent(**kw):
        dummy_agent(edit=faster)(**kw)  # it asked the screen for its bytes ...
        raise RuntimeError("agent process killed (signal 9)")  # ... and then the driver died

    before = package_digest(run.run_dir / "workspace")
    outcome = _driver(run, agent)(session=1, stage_root=run.run_dir / "stage")
    assert outcome["status"] == SES.ROUND_FAILED and "signal 9" in outcome["failure"]
    record = json.loads((run.run_dir / "stage" / "rounds" / "round_00.round.json").read_text())
    assert record["status"] == SES.ROUND_FAILED and record["agent_exit_code"] is None
    (digest,) = record["requested"]
    assert run.objective.screen.attribution(digest)["state"] == J.ATTRIBUTION_UNAUTHORED
    assert package_digest(run.run_dir / "workspace") == before


def test_a_driver_killed_mid_round_is_closed_by_the_next_start(run):
    """The marker names what the round asked for; nothing alive owns it, so its requests become
    unauthored and the round is recorded as killed."""
    seen = {}

    def agent(**kw):
        dummy_agent(edit=faster)(**kw)
        marker = Path(kw["stage_root"]) / "rounds" / f"round_00{R.OPEN_SUFFIX}"
        seen["marker"] = json.loads(marker.read_text())
        raise SystemExit("the launcher was killed")  # not an Exception: nothing of the round runs after it

    with pytest.raises(SystemExit):
        _driver(run, agent)(session=1, stage_root=run.run_dir / "stage")
    (digest,) = seen["marker"]["requested"]
    assert run.objective.screen.attribution(digest)["state"] == J.ATTRIBUTION_PENDING
    stage = run.run_dir / "stage"
    assert R.next_session(stage) == 2  # a relaunched start continues after the killed round
    # The marker records this process as the driver; the next start runs in another one.
    marker = stage / "rounds" / f"round_00{R.OPEN_SUFFIX}"
    marker.write_text(json.dumps({**seen["marker"], "pid": 2**22 + 1}))
    (record,) = R.recover_killed_rounds(stage, attribute=run.objective.attribute, run_name="run")
    assert record["status"] == R.ROUND_KILLED and record["attribution"] == {digest: J.ATTRIBUTION_UNAUTHORED}
    assert run.objective.screen.attribution(digest)["state"] == J.ATTRIBUTION_UNAUTHORED
    assert not marker.exists() and (stage / "rounds" / "round_00.round.json").is_file()
    assert R.recover_killed_rounds(stage, attribute=run.objective.attribute, run_name="run") == []


# ------------------------------------------------------------------- judging a recorded round again
def test_a_recorded_round_is_judged_again_from_its_own_evidence(run):
    """The replay reruns the round driver's own owners over the round's sealed evidence; a recorded
    status the evidence no longer supports is reported, never rewritten."""
    from merlin_experiments.phase2.whole_model_measured import round_audit as AUD

    stage = run.run_dir / "stage"
    target = SimpleNamespace(target="synthetic")
    driver = _driver(run, dummy_agent(edit=faster))
    assert driver(session=1, stage_root=stage)["status"] == "authored"
    document = AUD.audit_round(run.run_dir, 0, target_experiment=target, audit_token_set=TOKENS)
    assert document["agrees"] is True and document["replayed"]["status"] == "authored"
    assert document["replayed"]["receipts"]["joined"] is True and document["replayed"]["edits"]["status"] == "allowed"
    driver.agent = dummy_agent(edit=faster, extra=("cat golden.yaml",))
    assert driver(session=2, stage_root=stage)["status"] == "refused"
    refused = AUD.audit_round(run.run_dir, 1, target_experiment=target, audit_token_set=TOKENS)
    assert refused["agrees"] is True and refused["replayed"]["audit_clean"] is False
    numbers = sorted({n for lines in refused["replayed"]["hits"].values() for n in lines})
    assert any("golden.yaml" in text for text in AUD.transcript_lines(run.run_dir, 1, numbers).values())
    assert "DISAGREES" not in AUD.format_audit(refused)
    path = stage / "rounds" / "round_00.round.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, "status": "refused"}))
    assert AUD.audit_round(run.run_dir, 0, target_experiment=target, audit_token_set=TOKENS)["agrees"] is False
    assert json.loads(path.read_text())["status"] == "refused"  # the replay wrote nothing
    with pytest.raises(AUD.RoundAuditError, match="no record of round 7"):
        AUD.audit_round(run.run_dir, 7, target_experiment=target, audit_token_set=TOKENS)
