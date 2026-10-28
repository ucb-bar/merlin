"""Exercise autonomous retries and custody without launching an agent or hardware."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace

import pytest

from merlin.mining.reconstruction_learning import (
    REQUIRED_CHECKS,
    CheckResult,
    GateEvidence,
    GeneralizationPartition,
    LearningHooks,
    LearningTask,
    learning_tasks,
    run_learning,
)
from merlin.mining.reconstruction_proposal import (
    CompilerPatch,
    CompilerProposalAttempt,
    CompilerSurface,
    propose_compiler_patch,
    stage_candidate,
)
from merlin.transforms.evaluated_exploration import (
    CompletedObservation,
    EvaluationRequest,
    ExplorationIdentity,
    exploration_context_identity,
)
from merlin.transforms.pass_exploration import EmittedProgram, ExplorationGeometry
from merlin.transforms.pipeline_state import PipelineContext


def sha(value):
    return hashlib.sha256(value).hexdigest()


@dataclass
class Compilation:
    program: EmittedProgram
    satisfied: bool

    def summary(self):
        return dict(
            goals=("retention",),
            candidates=[dict(satisfied=self.satisfied)],
            gaps=[]
            if self.satisfied
            else [
                dict(
                    kind="adapter_gap",
                    goals=("retention",),
                    reason="producer decision is not consumed",
                    next_action="connect adapter",
                    witnesses=(),
                )
            ],
        )


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "source"
    (root / "src/passes").mkdir(parents=True)
    (root / "src/representations").mkdir()
    (root / "src/passes/lowering.py").write_text(
        "from representations.plan import ENABLED\nFACTOR = 2\ndef emit():\n    return FACTOR if ENABLED else 1\n"
    )
    (root / "src/representations/plan.py").write_text("ENABLED = False\n")
    patch = CompilerPatch(
        (
            (
                "src/passes/lowering.py",
                "from representations.plan import ENABLED\nFACTOR = 4\n"
                "def emit():\n    return FACTOR if ENABLED else 1\n",
            ),
            ("src/representations/plan.py", "ENABLED = True\n"),
        )
    )
    surfaces = (CompilerSurface("src/passes", "pass"), CompilerSurface("src/representations", "representation"))

    def geometry(n):
        return ExplorationGeometry("dense", ((n, 8), (8, 16)))

    partition = GeneralizationPartition((geometry(8),), (geometry(9),), (geometry(11),))
    task = LearningTask("retention-adapter", "adapter_gap", ("retention",), context={"selected_reference": "expert-a"})
    calls = []

    def reconstruct(candidate, task, budget):
        proc = candidate.run([sys.executable, "-c", "from passes.lowering import emit; print(emit())"])
        assert proc.returncode == 0, proc.stderr
        program = EmittedProgram(proc.stdout.encode(), "test-emission", "1" * 64)
        calls.append(("compile", candidate.compiler_sha256))
        return Compilation(program, proc.stdout.strip() == "4")

    def verify(candidate, result, task, split):
        calls.append(("verify", split))
        checks = tuple(
            CheckResult(
                name,
                "pass" if name != "mechanism" or result.satisfied else "fail",
                ("independent injected test owner",),
                "goal absent",
            )
            for name in REQUIRED_CHECKS
        )
        return GateEvidence(candidate.compiler_sha256, sha(result.program.data), checks)

    def requests(candidate, result, split):
        context = PipelineContext(library=candidate.compiler_sha256)
        identity = ExplorationIdentity(
            *("1" * 64 for _ in range(9)), exploration_context_identity(context), result.program.format, "modeled_roi"
        )
        return (EvaluationRequest(result.program, {"after": {"context": asdict(context)}}, identity),)

    def evaluate(program, receipt, identity):
        calls.append(("evaluate", sha(program.data)))
        return CompletedObservation(identity, sha(program.data), "2" * 64, "pass", "complete", True, 20)

    hooks = LearningHooks(reconstruct, verify, requests, evaluate)
    args = dict(
        compiler_root=root,
        source_roots=("src",),
        surfaces=surfaces,
        partition=partition,
        hooks=hooks,
        workspace=tmp_path / "run",
        environment_sha256="a" * 64,
    )
    return task, patch, args, calls


def test_multifile_candidate_fresh_process_automatic_retry_and_pinned_resume(experiment):
    task, patch, args, calls = experiment
    turns = []

    def proposer(task_card, **kw):
        turns.append(kw)
        assert task_card["route"] == "adapter_or_preservation"
        assert "held_out" not in json.dumps(task_card)
        if len(turns) == 1:
            return CompilerProposalAttempt(CompilerPatch((patch.files[0],)))  # missing representation consumer
        assert kw["feedback"][0]["stage"] == "reconstruction"
        return CompilerProposalAttempt(patch)

    result = run_learning(task, **args, proposer=proposer).summary()
    assert result["status"] == "accepted"
    assert len(turns) == 2 and result["attempts"][0]["rejection"]["stage"] == "reconstruction"
    assert len(result["validation_evaluation"]["ranking"]) == 1
    assert result["held_out"]["gate"] is None
    assert calls.count(("verify", "held_out")) == 1
    assert (args["compiler_root"] / "src/representations/plan.py").read_text() == "ENABLED = False\n"
    before = list(calls)
    resumed = run_learning(task, **args, proposer=lambda *a, **k: pytest.fail("cached run launched agent"))
    assert resumed.summary()["reused"] and calls == before
    with pytest.raises(ValueError, match="resume identity"):
        run_learning(task, **{**args, "environment_sha256": "b" * 64}, proposer=proposer)


def test_partial_roi_and_missing_cycles_remain_unranked(experiment):
    task, patch, args, calls = experiment
    previous = args["hooks"]
    hooks = replace(
        previous,
        evaluate=lambda p, r, i: CompletedObservation(i, sha(p.data), "2" * 64, "pass", "complete", False, None),
    )
    result = run_learning(
        task, **{**args, "hooks": hooks}, proposer=lambda *a, **k: CompilerProposalAttempt(patch)
    ).summary()
    assert result["status"] == "accepted"
    assert result["attempts"][0]["evaluation"]["ranking"] == []
    assert result["held_out"]["evaluation"]["completions"][0]["status"] == "unranked"
    assert result["performance_qualification"] is False


@pytest.mark.parametrize("failure", ["scheduling_dialect", "storage", "numerical", "emission", "compiler_identity"])
def test_missing_executable_gate_never_accepts_and_retries_automatically(experiment, failure):
    task, patch, args, calls = experiment
    previous = args["hooks"]

    def verify(candidate, result, task, split):
        answer = previous.verify(candidate, result, task, split)
        if candidate.patch_sha256 is not None:
            if failure == "compiler_identity":
                return replace(answer, compiler_sha256="0" * 64)
            return replace(
                answer,
                checks=tuple(replace(row, status="unknown") if row.name == failure else row for row in answer.checks),
            )
        return answer

    result = run_learning(
        task,
        **{**args, "hooks": replace(previous, verify=verify)},
        proposer=lambda *a, **k: CompilerProposalAttempt(patch),
    ).summary()
    assert result["status"] == "proposal_budget_exhausted" and len(result["attempts"]) == 3
    assert all(row["rejection"]["stage"] == failure for row in result["attempts"])
    assert not any(row[0] == "evaluate" for row in calls)


def test_unchanged_emission_is_inert_even_if_agent_claims_a_mechanism(experiment):
    task, patch, args, calls = experiment
    previous = args["hooks"]

    def reconstruct(candidate, task, budget):
        previous.reconstruct(candidate, task, budget)
        return Compilation(
            EmittedProgram(b"unchanged\n", "test-emission", "1" * 64), candidate.patch_sha256 is not None
        )

    result = run_learning(
        task,
        **{**args, "hooks": replace(previous, reconstruct=reconstruct)},
        proposer=lambda *a, **k: CompilerProposalAttempt(patch),
    ).summary()
    assert all(row["rejection"]["stage"] == "noninert" for row in result["attempts"])


def test_holdout_failure_or_exception_does_not_reopen_authoring(experiment):
    task, patch, args, calls = experiment
    previous = args["hooks"]
    turns = []

    def verify(candidate, result, task, split):
        if split == "held_out":
            raise RuntimeError("withheld owner transport unavailable")
        return previous.verify(candidate, result, task, split)

    def propose(*a, **kw):
        turns.append(kw)
        return CompilerProposalAttempt(patch)

    result = run_learning(task, **{**args, "hooks": replace(previous, verify=verify)}, proposer=propose).summary()
    assert result["status"] == "held_out_error" and len(turns) == 1
    assert result["patch_sha256"] == patch.sha256


def test_unknown_evidence_is_not_a_request_to_invent_a_pass(experiment):
    task, patch, args, calls = experiment
    result = run_learning(
        replace(task, gap_kind="missing_evidence"),
        **args,
        proposer=lambda *a, **k: pytest.fail("evidence gap launched proposer"),
    ).summary()
    assert result["status"] == "evidence_or_legality_required"
    routed = learning_tasks(
        {
            "goals": ("reuse",),
            "gaps": [dict(kind="bounded_search_exhausted", goals=("reuse",), reason="truncated", witnesses=())],
        }
    )[0].to_dict()
    assert routed["route"] == "library_parameters_or_order" and not routed["missing_pass_certainty"]


def test_partition_aliases_and_patch_escape_are_rejected(experiment, tmp_path):
    task, patch, args, calls = experiment
    partition = args["partition"]
    with pytest.raises(ValueError, match="overlap"):
        GeneralizationPartition(partition.training, partition.validation, partition.training)
    for path in ("../src/passes/x.py", "/src/passes/x.py", "src/passes/../x.py", "src/passes/tests/gate.py"):
        with pytest.raises(ValueError):
            CompilerPatch(((path, "x = 1\n"),)).validate(args["surfaces"])
    with pytest.raises(ValueError, match="compiler surfaces"):
        CompilerSurface("src/merlin/verify", "pass")
    staged = stage_candidate(
        args["compiler_root"], args["source_roots"], tmp_path / "isolated", patch=patch, surfaces=args["surfaces"]
    )
    assert staged.patch_sha256 == patch.sha256
    assert all(not path.is_symlink() for path in staged.root.rglob("*"))


def test_production_proposer_multifile_file_protocol_uses_existing_sandbox(experiment, tmp_path, monkeypatch):
    task, patch, args, calls = experiment
    from merlin.mining import pass_agent

    monkeypatch.setattr(pass_agent, "sandbox_argv", lambda ws: ["isolated-test-wrapper"])
    seen = []

    def runner(**kw):
        seen.append(kw)
        card = kw["workspace"] / "TASK.md"
        assert "scheduling" in card.read_text() and "selected_reference" in card.read_text()
        assert not (kw["workspace"] / "held_out").exists()
        (kw["workspace"] / "proposal.json").write_text(
            json.dumps(
                dict(files=[dict(path=p, source=s) for p, s in patch.files], rationale="connect producer and consumer")
            )
        )
        return {"usage": {"input_tokens": 10}}

    attempted = propose_compiler_patch(
        task.to_dict(),
        compiler_root=args["compiler_root"],
        surfaces=args["surfaces"],
        workspace=tmp_path / "proposer",
        runner=runner,
    )
    assert attempted.patch == replace(patch, rationale="connect producer and consumer")
    assert attempted.sandboxed and seen[0]["sandbox"] == ["isolated-test-wrapper"]
    monkeypatch.setattr(pass_agent, "sandbox_argv", lambda ws: None)
    monkeypatch.setattr(pass_agent, "_sandbox_condition", lambda: "unavailable-test")
    refused = propose_compiler_patch(
        task.to_dict(),
        compiler_root=args["compiler_root"],
        surfaces=args["surfaces"],
        workspace=tmp_path / "refused",
        runner=lambda **k: pytest.fail("unsafe launch"),
    )
    assert refused.patch is None and "sandbox_unavailable" in refused.error
