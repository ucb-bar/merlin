"""Bounded real pass exploration joined to externally completed cycle feedback.

The ordinary caller supplies source/correctness/provider custody and target pass
factories. This module neither runs a simulator nor invents a schedule, score or
verification grant. Selection returns the already emitted evaluated product.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields

from .pass_exploration import ObservedMetrics, explore_candidates, generate_candidates, rank_performance_variants


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def exploration_context_identity(context):
    """Bind the actual shared PipelineContext without inspecting a target name."""
    from .pipeline_state import PipelineContext

    if type(context) is not PipelineContext:
        raise ValueError("explicit ordinary pipeline context required")
    return _sha(json.dumps(asdict(context), sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class ExplorationIdentity:
    source_sha256: str
    abi_sha256: str
    inputs_sha256: str
    reference_sha256: str
    facts_sha256: str
    library_sha256: str
    provider_sha256: str
    scope_sha256: str
    roi_sha256: str
    context_sha256: str
    artifact_format: str
    basis: str

    def __post_init__(self):
        if (any(not _pin(getattr(self, field.name)) for field in fields(self) if field.name.endswith("_sha256"))
                or self.basis not in ("modeled_roi", "measured_roi")
                or type(self.artifact_format) is not str or not self.artifact_format or len(self.artifact_format) > 64):
            raise ValueError("exploration needs complete source/ABI/input/reference/context/provider/ROI pins")

    def digest(self):
        return _sha(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class CompletedObservation:
    identity: ExplorationIdentity
    program_sha256: str
    feedback_sha256: str
    numerical_status: str
    timing_status: str
    complete_roi: bool
    cycles: int | None

    def __post_init__(self):
        if (type(self.identity) is not ExplorationIdentity or not _pin(self.program_sha256)
                or not _pin(self.feedback_sha256) or self.numerical_status not in ("pass", "fail", "unknown")
                or self.timing_status not in ("complete", "unknown", "failed") or type(self.complete_roi) is not bool
                or self.cycles is not None and (type(self.cycles) is not int or self.cycles <= 0)):
            raise ValueError("external observation requires exact identities and explicit numerical/timing status")
        self.identity.__post_init__()

    @property
    def eligible(self):
        return (self.numerical_status == "pass" and self.timing_status == "complete"
                and self.complete_roi and self.cycles is not None)


@dataclass(frozen=True)
class EvaluatedExploration:
    candidates: tuple
    observations: tuple[CompletedObservation, ...]
    ranking: dict
    winner: object | None


def explore_evaluated(module, families, facts, *, output, emission, evaluate, identity,
                      before=(), max_candidates=16, max_search_steps=4096, **session_options):
    """Run actual registered passes, deduplicate emitted bytes, consume feedback.

    evaluate(EmittedProgram, pipeline_receipt, ExplorationIdentity) returns a
    CompletedObservation or None for pending feedback. The callback belongs to
    the ordinary tool owner; supplied hashes detect mismatched joins, not forged
    evidence. No missing cycles, partial ROI or numerical failure can win.
    """
    if type(identity) is not ExplorationIdentity or not callable(evaluate):
        raise ValueError("explicit exploration identity and external evaluation adapter required")
    identity.__post_init__()
    space = generate_candidates(families, output=output, before=before,
                                max_candidates=max_candidates, max_search_steps=max_search_steps)
    observations = {}

    def feedback(program, receipt):
        context = receipt["after"]["context"]
        if (_sha(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()) != identity.context_sha256
                or program.format != identity.artifact_format):
            raise ValueError("actual emitted pipeline context/format differs from the selected evaluation domain")
        digest = _sha(program.data)
        observed = evaluate(program, receipt, identity)
        if observed is not None:
            if type(observed) is not CompletedObservation:
                raise ValueError("evaluation must return an explicit completed observation or pending None")
            observed.__post_init__()
            if observed.identity != identity or observed.program_sha256 != digest:
                raise ValueError("evaluation does not match this exact source/program/provider/ROI")
            observations[digest] = observed
        return {"program_sha256": digest, "evaluation_identity": identity.digest(),
                "status": "pending" if observed is None else "rankable" if observed.eligible else "unranked",
                "feedback_sha256": None if observed is None else observed.feedback_sha256}

    candidates = tuple(explore_candidates(module, space, facts, output=output,
        emission=emission, feedback=feedback, **session_options))
    unique = {_sha(item.emission.data): item for item in candidates
              if item.emission is not None and item.duplicate_of is None}
    metrics = tuple((digest, ObservedMetrics(identity.basis, identity.digest(), (("cycles", observation.cycles),)))
                    for digest, observation in observations.items() if observation.eligible)
    ranking = rank_performance_variants(metrics) if metrics else {
        "ranked": (), "decision_metric": "complete_same_roi_cycles", "performance_qualification": False}
    ranking = {**ranking, "unranked_programs": tuple(digest for digest in unique
               if digest not in observations or not observations[digest].eligible),
               "policy": "no static-count, partial-ROI or UNKNOWN fallback"}
    winner = unique[ranking["ranked"][0][0]] if ranking["ranked"] else None
    return EvaluatedExploration(candidates, tuple(observations.values()), ranking, winner)
