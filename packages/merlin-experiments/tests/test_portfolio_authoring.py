"""Installed sequence evidence with synthetic round transport, never paid authoring."""

import hashlib
import importlib.util
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import campaign as C
from merlin_experiments.phase2 import contracts as D
from merlin_experiments.phase2 import portfolio_checkpoint as PC
from merlin_experiments.phase2.global_experiment import GlobalPerfExperiment
from merlin_experiments.phase2.portfolio_authoring import PortfolioAuthoring

from merlin.perf.execution_policy import GLOBAL_AUTHORING_ROUND_MAX_SECONDS
from merlin.targetgen.sandbox import toolchain as TC


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("authoring evidence tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def case(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "portfolio_authoring_fixture", Path(__file__).with_name("portfolio_analysis_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    seed = helper.build_case(tmp_path / "seed", monkeypatch)
    inputs = seed.inputs
    experiment = GlobalPerfExperiment(
        baseline=inputs.baseline,
        baseline_sha256=inputs.baseline_sha256,
        sentinel=inputs.sentinel,
        portfolio_sentinels=inputs.portfolio_sentinels[1:],
        target=inputs.target,
        target_sha256=inputs.target_sha256,
        compiler_shared_source_root=inputs.compiler_shared_source_root,
        contract_root=inputs.contract_root,
        controller_source=inputs.controller_source,
        prior_shared_source_relative=Path("compiler"),
        prior_shared_source_fallback=inputs.compiler_shared_source_root,
        output=tmp_path / "experiment",
        analyzer=seed.analyzer,
    )
    paths = TC.ToolchainPaths(tmp_path, *(str(tmp_path / name) for name in ("venv", "llvm", "compat", "clang", "uv")))
    selection = C.PackageSandboxInputs(paths, TC.SimToolchain(), "", ())
    root = tmp_path / "agent"
    root.mkdir()
    payload = b"public"
    (root / "interface").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = root / "agent_input_manifest.json"
    manifest.write_bytes(
        D.canonical_json({"files": [{"path": "interface", "sha256": digest, "n_bytes": len(payload)}]})
    )
    aggregate = hashlib.sha256(f"interface\0{digest}\0{len(payload)}\n".encode()).hexdigest()
    agent = AW.AgentInputSnapshot(root, manifest, D.sha256_file(manifest), aggregate, 1, len(payload))
    declared = {"status": "synthetic_declared", "instructions": [{"name": "sample", "encoding": 37}]}
    owner = PortfolioAuthoring(
        experiment,
        target_experiment=SimpleNamespace(target="synthetic"),
        stage_root=tmp_path / "stage",
        agent_inputs=agent,
        frozen_functional=None,
        frozen_corpus_manifest=manifest,
        sandbox_inputs=selection,
        model="synthetic",
        resolved_model="synthetic-resolved",
        effort="low",
        codex_binary=tmp_path / "never-executed",
        max_tool_calls=4,
        declared_instruction_evidence=declared,
    )
    return SimpleNamespace(
        owner=owner,
        experiment=experiment,
        candidate=seed.candidate,
        analyzer=seed.analyzer,
        root=tmp_path,
        agent=agent,
        declared=declared,
    )


def consume(case, row):
    path = Path(row["path"])
    assert D.sha256_file(path) == row["sha256"]
    assert not path.stat().st_mode & 0o222
    return PC.consume_round_checkpoint(
        path,
        context=PC.CheckpointVerificationContext(
            host_policy=case.experiment.inputs.host_policy,
            compiler_shared_source_root=case.experiment.inputs.compiler_shared_source_root,
        ),
    )


@pytest.mark.parametrize("failure", [None, "author", "capacity"])
def test_sequence_consumes_checkpoints_and_recovers_exact_prior_bytes(case, monkeypatch, failure):
    calls = []

    def round_(candidate, *, round_index, round_timeout_s):
        calls.append((round_index, round_timeout_s, (candidate / "version.txt").read_text(), candidate))
        (candidate / "version.txt").write_text(str(round_index + 1))
        case.experiment.analysis.analyze(candidate, hypothesis="synthetic authored change")
        if failure and round_index == 1:
            case.experiment._write(
                "agent_round_0001.json",
                {
                    "agent_exit_code": 1 if failure == "capacity" else -15,
                    "audit": {"clean": True, "broker_invocations": []},
                    "broker_evidence": {"all_required_succeeded": failure == "author"},
                },
            )
            if failure == "capacity":
                path = case.root / "stage/rounds/round_01.codex_summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "exit_code": 1,
                            "timed_out": False,
                            "turns_started": 1,
                            "turns_usage_reported": 0,
                            "errors": ["Selected model is at capacity."],
                        }
                    )
                )
            raise ValueError("synthetic transport failure")
        return {"status": "authored"}

    monkeypatch.setattr(case.owner, "run_round", round_)
    result = case.owner.run_sequence(
        case.candidate,
        max_rounds=4,
        total_authoring_seconds=70,
        round_seconds=30,
        on_round_failure="resume-last-checkpoint",
    )
    assert [(i, budget, version) for i, budget, version, _ in calls] == [
        (0, 30, "0"),
        (1, 30, "1"),
        (2, 10, "1" if failure else "2"),
    ]
    assert len({path for *_, path in calls}) == 3
    assert result["authoring_seconds_reserved"] == 70
    assert result["global_speedup_proven"] is False
    assert result["checkpoints"][0]["role"] == "initial_verified_seed_not_an_authored_result"
    for checkpoint in result["checkpoints"]:
        assert consume(case, checkpoint)["candidate_sha256"] == checkpoint["candidate_sha256"]
    if failure:
        assert result["failures"][0]["recovery"] == "next_budgeted_round_from_consumed_checkpoint"
        assert result["failures"][0]["live_handle_restarted"] is False
        assert result["failures"][0]["retryable_capacity_failure"] is (failure == "capacity")
    assert json.loads((case.experiment.output / "agent_sequence.json").read_text()) == result


@pytest.mark.parametrize("regression", [False, True])
def test_blocked_repair_and_ready_member_regression_have_distinct_authority(case, monkeypatch, regression):
    if not regression:
        case.analyzer.fail_member = case.experiment.inputs.portfolio_sentinels[-1].capsule

    def round_(candidate, **kwargs):
        (candidate / "version.txt").write_text("1")
        case.analyzer.fail_member = case.experiment.inputs.sentinel.capsule if regression else None
        case.experiment.analysis.analyze(candidate, hypothesis="synthetic portfolio repair or regression")
        return {"status": "authored"}

    monkeypatch.setattr(case.owner, "run_round", round_)
    options = dict(max_rounds=1, total_authoring_seconds=30, round_seconds=30)
    if regression:
        with pytest.raises(ValueError, match="regressed previously verified portfolio members"):
            case.owner.run_sequence(case.candidate, **options)
        assert not (case.experiment.output / "round_0000_candidate.json").exists()
    else:
        result = case.owner.run_sequence(case.candidate, **options)
        initial = consume(case, result["checkpoints"][0])
        assert initial["schema"] == "global_authoring_checkpoint_v1"
        assert initial["portfolio_members_ready"] == 1
        assert result["promotion_ready"] is True
        assert consume(case, result["last_good_checkpoint"])["schema"] == "global_perf_candidate_v1"


@pytest.mark.parametrize("drift", ["baseline", "controller", "checkpoint", "audit"])
def test_recovery_never_bypasses_input_checkpoint_or_audit_refusal(case, monkeypatch, drift):
    calls = []

    def round_(candidate, **kwargs):
        calls.append(candidate)
        if drift == "baseline":
            (case.experiment.inputs.baseline / "version.txt").write_text("tampered")
        elif drift == "controller":
            case.experiment.inputs.controller_source.write_text("changed source")
        elif drift == "checkpoint":
            checkpoint = case.experiment.output / "initial_seed_candidate.json"
            checkpoint.chmod(0o644)
            checkpoint.write_text("{}")
        case.experiment._write(
            "agent_round_0000.json",
            {
                "agent_exit_code": -15,
                "audit": {"clean": drift != "audit"},
                "broker_evidence": {"all_required_succeeded": True},
            },
        )
        raise ValueError("synthetic refused round")

    monkeypatch.setattr(case.owner, "run_round", round_)
    with pytest.raises(ValueError, match="synthetic refused round"):
        case.owner.run_sequence(
            case.candidate,
            max_rounds=2,
            total_authoring_seconds=60,
            round_seconds=30,
            on_round_failure="resume-last-checkpoint",
        )
    assert len(calls) == 1
    failure = json.loads((case.experiment.output / "continuation_failure_0000.json").read_text())
    assert failure["recovery"] == "stop"


@pytest.mark.parametrize(
    "options",
    [
        dict(max_rounds=0),
        dict(total_authoring_seconds=0),
        dict(round_seconds=GLOBAL_AUTHORING_ROUND_MAX_SECONDS + 1),
        dict(on_round_failure="retry-live"),
    ],
)
def test_invalid_bounds_refuse_before_analysis(case, options):
    kwargs = dict(max_rounds=1, total_authoring_seconds=30, round_seconds=30)
    kwargs.update(options)
    with pytest.raises(ValueError):
        case.owner.run_sequence(case.candidate, **kwargs)
    assert not case.analyzer.calls


def test_actual_round_refuses_missing_phase1_before_transport(case):
    with pytest.raises(ValueError, match="Phase-1 qualification"):
        case.owner.run_round(case.candidate, round_index=0, round_timeout_s=30)
    assert not case.analyzer.calls


def test_actual_round_verifies_public_inputs_before_phase1(case):
    (case.agent.root / "interface").write_text("tampered")
    with pytest.raises(D.StageGateError, match="input changed"):
        case.owner.run_round(case.candidate, round_index=0, round_timeout_s=30)
    assert not case.analyzer.calls


def test_declared_instruction_evidence_is_detached_from_callers(case):
    original = json.loads(json.dumps(case.declared))
    case.declared["instructions"][0]["encoding"] = 99
    assert case.owner.declared_instruction_evidence == original
