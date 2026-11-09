"""Automatic reference-guided compiler learning through trusted executable gates.

Selected expert evidence is an explicit input to this separate mode. Existing
campaigns and their reference visibility are unaffected. Agents propose compiler
sources; the ordinary compiler/evaluation owner reconstructs, verifies and times
isolated candidates. Gate failures feed bounded retries automatically. Held-out
results are collected only after the final candidate is frozen and never feed a
proposer. No similarity score or static instruction count substitutes for cycles.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from merlin.transforms.evaluated_exploration import EvaluationRequest, evaluate_emitted
from merlin.transforms.pass_exploration import ExplorationGeometry, ObservedMetrics, rank_performance_variants

from .reconstruction_proposal import (
    CompilerPatch,
    CompilerProposalAttempt,
    CompilerSurface,
    digest,
    propose_compiler_patch,
    source_inventory,
    stage_candidate,
)

ROUTES = {
    "bounded_search_exhausted": "library_parameters_or_order",
    "bounded_search_exhaustion": "library_parameters_or_order",
    "unbound_library_requirement": "library_parameters_or_order",
    "forward_execution_failure": "inspect_execution_witnesses",
    "legality_or_capability_restriction": "evidence_or_legality",
    "bounded_search": "library_parameters_or_order",
    "search_exhaustion": "library_parameters_or_order",
    "adapter_gap": "adapter_or_preservation",
    "adapter": "adapter_or_preservation",
    "composition_gap": "composition",
    "composition": "composition",
    "representation_gap": "new_representation_or_transformation",
    "representation": "new_representation_or_transformation",
    "transformation_gap": "new_representation_or_transformation",
    "transformation": "new_representation_or_transformation",
    "missing_evidence": "evidence_or_legality",
    "evidence": "evidence_or_legality",
    "semantic_mismatch": "evidence_or_legality",
    "legality": "evidence_or_legality",
    "capability_restriction": "evidence_or_legality",
}
REQUIRED_CHECKS = (
    "source_semantics",
    "numerical",
    "storage",
    "target",
    "scheduling_dialect",
    "emission",
    "mechanism",
    "frozen_baseline",
    "noninert",
)


@dataclass(frozen=True)
class LearningBudget:
    max_candidates: int = 8
    max_search_steps: int = 4096
    proposal_attempts: int = 3

    def __post_init__(self):
        if (
            type(self.max_candidates) is not int
            or not 1 <= self.max_candidates <= 256
            or type(self.max_search_steps) is not int
            or not 1 <= self.max_search_steps <= 65536
            or type(self.proposal_attempts) is not int
            or not 1 <= self.proposal_attempts <= 16
        ):
            raise ValueError("learning requires bounded candidate, search and proposal budgets")


@dataclass(frozen=True)
class LearningTask:
    task_id: str
    gap_kind: str
    goals: tuple[str, ...]
    witnesses: tuple[dict, ...] = ()
    context: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            type(self.task_id) is not str
            or not self.task_id
            or self.gap_kind not in ROUTES
            or type(self.goals) is not tuple
            or not self.goals
            or any(type(goal) is not str or not goal for goal in self.goals)
            or type(self.witnesses) is not tuple
            or any(type(row) is not dict for row in self.witnesses)
            or type(self.context) is not dict
        ):
            raise ValueError("learning task needs typed goals, diagnosed gap and selected evidence context")
        digest(asdict(self))

    def to_dict(self):
        return dict(
            schema="merlin.reconstruction_learning_task.v1",
            **asdict(self),
            route=ROUTES[self.gap_kind],
            missing_pass_certainty=False,
        )


def learning_tasks(result, *, context=None):
    """Route actual reconstruction gaps without equating truncation with absence."""
    summary = result.summary() if hasattr(result, "summary") else result
    tasks = []
    for gap in summary.get("gaps", ()):
        kind = gap["kind"]
        if kind not in ROUTES:
            raise ValueError(f"unregistered reconstruction gap kind: {kind}")
        tasks.append(
            LearningTask(
                digest(gap)[:24],
                kind,
                tuple(gap.get("goals") or summary["goals"]),
                tuple(gap.get("witnesses", ())),
                dict(context or {}, diagnosis=gap.get("reason", ""), next_action=gap.get("next_action", "")),
            )
        )
    return tuple(tasks)


@dataclass(frozen=True)
class GeneralizationPartition:
    training: tuple[ExplorationGeometry, ...]
    validation: tuple[ExplorationGeometry, ...]
    held_out: tuple[ExplorationGeometry, ...]

    def __post_init__(self):
        identities = []
        for split in (self.training, self.validation, self.held_out):
            if type(split) is not tuple or not split or any(type(row) is not ExplorationGeometry for row in split):
                raise ValueError("freeze nonempty development, validation and held-out geometries")
            identities.extend(row.identity for row in split)
        if len(set(identities)) != len(identities):
            raise ValueError("geometry aliases must not overlap generalization partitions")

    def manifest(self):
        return {name: [asdict(row) for row in getattr(self, name)] for name in ("training", "validation", "held_out")}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    evidence: tuple[str, ...]
    reason: str = ""

    def __post_init__(self):
        if (
            not self.name
            or self.status not in ("pass", "fail", "unknown")
            or type(self.evidence) is not tuple
            or not self.evidence
            or any(type(row) is not str or not row for row in self.evidence)
        ):
            raise ValueError("executable checks need explicit status and independently collected witnesses")


@dataclass(frozen=True)
class GateEvidence:
    compiler_sha256: str
    program_sha256: str | None
    checks: tuple[CheckResult, ...]

    def __post_init__(self):
        for value in (self.compiler_sha256,) + (() if self.program_sha256 is None else (self.program_sha256,)):
            if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("gate must pin the actual candidate compiler and emitted program")
        if (
            type(self.checks) is not tuple
            or any(type(row) is not CheckResult for row in self.checks)
            or len({row.name for row in self.checks}) != len(self.checks)
        ):
            raise ValueError("gate checks must be distinct owner-collected results")


@dataclass(frozen=True)
class LearningHooks:
    """Trusted adapter, never supplied by proposed compiler code.

    reconstruct(candidate, task, budget) must use fresh imports, ordinarily through
    candidate.run(). verify(candidate, result, task, split) performs independent
    checks over all frozen geometries in that split; source membership and
    scheduling-dialect consumption must be checked in actual pipeline receipts.
    evaluation_requests(candidate, result, split) returns exact EvaluationRequest
    objects. evaluate is the ordinary evaluated_exploration provider callback.
    Goldens and held-out inputs stay in the owner, outside proposer workspaces.
    """

    reconstruct: Callable
    verify: Callable
    evaluation_requests: Callable
    evaluate: Callable

    def __post_init__(self):
        if any(not callable(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("all independent compiler/evaluator hooks are required")


@dataclass(frozen=True)
class LearningResult:
    record: dict

    def summary(self):
        return json.loads(json.dumps(self.record))


def _summary(result):
    return result.summary() if hasattr(result, "summary") else result


def _satisfied(result):
    return any(row.get("satisfied") is True for row in _summary(result).get("candidates", ()))


def _gate(candidate, evidence, *, baseline=None):
    if type(evidence) is not GateEvidence:
        return dict(stage="unverifiable", reason="owner did not return typed executable gate evidence")
    evidence.__post_init__()
    if evidence.compiler_sha256 != candidate.compiler_sha256:
        return dict(stage="compiler_identity", reason="verification ran a different compiler/overlay")
    if evidence.program_sha256 is None:
        return dict(stage="emission", reason="no actual emitted artifact was verified")
    checks = {check.name: check for check in evidence.checks}
    for name in REQUIRED_CHECKS:
        check = checks.get(name)
        if check is None or check.status != "pass":
            return dict(
                stage=name,
                reason="missing or unknown executable check" if check is None else check.reason,
                check=None if check is None else asdict(check),
            )
    if (
        baseline is not None
        and baseline.program_sha256 is not None
        and evidence.program_sha256 == baseline.program_sha256
    ):
        return dict(stage="noninert", reason="proposal emitted the unchanged control program")
    return None


def _evaluate(candidate, result, split, hooks, budget):
    requests = tuple(hooks.evaluation_requests(candidate, result, split))
    if len(requests) > budget.max_candidates or any(type(row) is not EvaluationRequest for row in requests):
        raise ValueError("owner must supply bounded exact emitted evaluation requests")
    observations = []
    completions = []
    for completed in evaluate_emitted(requests, hooks.evaluate, max_requests=budget.max_candidates):
        completions.append(completed.record())
        if completed.observation is not None:
            observations.append(completed.observation)
    domains = {}
    for row in observations:
        if row.eligible:
            domains.setdefault(row.identity.digest(), []).append(
                (
                    row.program_sha256,
                    ObservedMetrics(row.identity.basis, row.identity.digest(), (("cycles", row.cycles),)),
                )
            )
    return dict(
        completions=completions,
        ranking=[rank_performance_variants(tuple(rows)) for rows in domains.values()],
        policy="complete same-domain ROI cycles only; no private cycle-model calibration",
    )


def _write(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _final_holdout(record, candidate, reconstructed, task, hooks, budget):
    record["status"] = "accepted"
    try:
        record["validation_evaluation"] = _evaluate(candidate, reconstructed, "validation", hooks, budget)
        heldout = hooks.verify(candidate, reconstructed, task, "held_out")
        failure = _gate(candidate, heldout)
        record["held_out"] = dict(
            gate=failure,
            evaluation=_evaluate(candidate, reconstructed, "held_out", hooks, budget) if failure is None else None,
        )
        if failure is not None:
            record["status"] = "held_out_failed"
    except Exception as error:
        record["status"] = "held_out_error"
        record["held_out"] = dict(exception=type(error).__name__, reason=str(error))
    return record


def run_learning(
    task,
    *,
    compiler_root,
    source_roots,
    surfaces,
    partition,
    hooks,
    workspace,
    environment_sha256,
    proposer=None,
    budget=LearningBudget(),
):
    """Reconstruct, propose, verify, retry and certify automatically within bounds.

    Source roots and surfaces are selected from the trusted target/library
    inventory. The environment pin covers the adapter, evaluator, evidence and
    numerical/ROI contracts; changing any of those must invalidate resume.
    Accepted patches remain isolated reviewable artifacts, usable directly by
    following experiment rounds. They are never written into the live checkout.
    """
    if (
        type(task) is not LearningTask
        or type(partition) is not GeneralizationPartition
        or type(hooks) is not LearningHooks
        or type(budget) is not LearningBudget
        or type(source_roots) is not tuple
        or not source_roots
        or type(surfaces) is not tuple
        or not surfaces
        or any(type(row) is not CompilerSurface for row in surfaces)
        or type(environment_sha256) is not str
        or len(environment_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in environment_sha256)
    ):
        raise ValueError("explicit pinned task, surfaces, partition, trusted hooks and environment required")
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    identity = digest(
        dict(
            task=task.to_dict(),
            sources=source_inventory(compiler_root, source_roots),
            surfaces=[asdict(row) for row in surfaces],
            partition=partition.manifest(),
            environment_sha256=environment_sha256,
            budget=asdict(budget),
        )
    )
    manifest_path = ws / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        raise ValueError("resume identity differs in source, task, partition, evaluator or budget")
    _write(
        manifest_path,
        dict(
            schema="merlin.reconstruction_learning_run.v1",
            identity=identity,
            task=task.to_dict(),
            budget=asdict(budget),
            partition=partition.manifest(),
            environment_sha256=environment_sha256,
        ),
    )
    final_path = ws / "learning-result.json"
    if final_path.exists():
        record = json.loads(final_path.read_text())
        patch_path = record.get("accepted_patch")
        if patch_path:
            content = json.loads((ws / patch_path).read_text())
            if digest(dict(files=tuple(tuple(row) for row in content["files"]))) != record["patch_sha256"]:
                raise ValueError("retained accepted patch bytes changed")
        if record["status"] == "held_out_pending":
            if not patch_path:
                raise ValueError("pending certification has no frozen patch")
            patch = CompilerPatch(tuple(tuple(row) for row in content["files"]), content["rationale"])
            candidate = stage_candidate(
                compiler_root, source_roots, ws / ("resume-" + uuid.uuid4().hex), patch=patch, surfaces=surfaces
            )
            if candidate.compiler_sha256 != record["compiler_sha256"]:
                raise ValueError("resumed compiler differs from frozen accepted source")
            reconstructed = hooks.reconstruct(candidate, task, budget)
            _final_holdout(record, candidate, reconstructed, task, hooks, budget)
            _write(final_path, record)
        return LearningResult(dict(record, reused=True))
    events_path = ws / "attempts.json"
    attempts = json.loads(events_path.read_text()) if events_path.exists() else []
    control = stage_candidate(compiler_root, source_roots, ws / ("control-" + uuid.uuid4().hex))
    initial = hooks.reconstruct(control, task, budget)
    record = dict(
        schema="merlin.reconstruction_learning_result.v1",
        identity=identity,
        task_id=task.task_id,
        route=ROUTES[task.gap_kind],
        baseline=_summary(initial),
        attempts=attempts,
        status="unresolved",
        performance_qualification=False,
        reused=False,
    )
    baseline = hooks.verify(control, initial, task, "training")
    if type(baseline) is not GateEvidence or baseline.compiler_sha256 != control.compiler_sha256:
        raise ValueError("control requires exact actual emitted-program/compiler identity")
    if _satisfied(initial):
        rejection = _gate(control, baseline)
        if rejection is None:
            record.update(status="existing_library_satisfied", validation=[], held_out=[])
            for split in ("validation", "held_out"):
                check = hooks.verify(control, initial, task, split)
                failure = _gate(control, check)
                record[split] = dict(
                    gate=None if failure is None else failure,
                    evaluation=_evaluate(control, initial, split, hooks, budget) if failure is None else None,
                )
                if failure is not None:
                    record["status"] = "generalization_failed"
            _write(final_path, record)
            return LearningResult(record)
    if ROUTES[task.gap_kind] == "evidence_or_legality":
        record["status"] = "evidence_or_legality_required"
        _write(final_path, record)
        return LearningResult(record)
    from .pass_slot import scan_cheats

    for number in range(len(attempts), budget.proposal_attempts):
        turn = ws / f"attempt-{number + 1:03d}-{uuid.uuid4().hex}"
        # Task card contains selected development evidence, not partitions or evaluator records.
        try:
            attempt = (proposer or propose_compiler_patch)(
                task.to_dict(),
                compiler_root=compiler_root,
                surfaces=surfaces,
                workspace=turn / "proposer",
                feedback=tuple(row["rejection"] for row in attempts),
            )
        except Exception as error:
            attempt = CompilerProposalAttempt(None, f"{type(error).__name__}: {error}")
        entry = dict(number=number + 1, patch_sha256=None, rejection=None)
        if type(attempt) is not CompilerProposalAttempt or attempt.patch is None:
            entry["rejection"] = dict(stage="proposal", reason=getattr(attempt, "error", "invalid proposer result"))
        else:
            patch = attempt.patch
            entry["patch_sha256"] = patch.sha256
            try:
                patch.validate(surfaces)
                findings = {path: scan_cheats(source) for path, source in patch.files if path.endswith(".py")}
                findings = {path: value for path, value in findings.items() if value}
                if findings:
                    entry["rejection"] = dict(stage="source", reason="forbidden behavioral source", findings=findings)
                else:
                    candidate = stage_candidate(
                        compiler_root, source_roots, turn / "compiler", patch=patch, surfaces=surfaces
                    )
                    reconstructed = hooks.reconstruct(candidate, task, budget)
                    entry["reconstruction"] = _summary(reconstructed)
                    if not _satisfied(reconstructed):
                        entry["rejection"] = dict(
                            stage="reconstruction",
                            reason="emitted goals remain unresolved",
                            gaps=_summary(reconstructed).get("gaps", ()),
                        )
                    else:
                        for split in ("training", "validation"):
                            check = hooks.verify(candidate, reconstructed, task, split)
                            entry["rejection"] = _gate(
                                candidate, check, baseline=baseline if split == "training" else None
                            )
                            if entry["rejection"] is not None:
                                entry["rejection"]["split"] = split
                                break
                        if entry["rejection"] is None:
                            entry["evaluation"] = _evaluate(candidate, reconstructed, "training", hooks, budget)
                            patch_file = turn / "accepted-patch.json"
                            _write(patch_file, dict(files=patch.files, rationale=patch.rationale))
                            record.update(
                                status="accepted",
                                accepted_patch=str(patch_file.relative_to(ws)),
                                patch_sha256=patch.sha256,
                                compiler_sha256=candidate.compiler_sha256,
                                candidate_root=str(candidate.root),
                            )
            except Exception as error:  # candidate failure is retained and fed to the next bounded turn
                entry["rejection"] = dict(stage="candidate_error", exception=type(error).__name__, reason=str(error))
        attempts.append(entry)
        _write(events_path, attempts)
        if entry["rejection"] is None:
            break
    record["attempts"] = attempts
    if record["status"] == "accepted":
        # Freeze before touching holdouts. Even infrastructure failures here may
        # not reopen authoring or disclose their details in another task card.
        _write(final_path, dict(record, status="held_out_pending"))
        _final_holdout(record, candidate, reconstructed, task, hooks, budget)
    if record["status"] == "unresolved":
        record["status"] = "proposal_budget_exhausted"
    _write(final_path, record)
    return LearningResult(record)
